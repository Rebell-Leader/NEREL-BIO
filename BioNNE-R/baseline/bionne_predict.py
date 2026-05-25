#!/usr/bin/env python3
"""
BioNNE-R prediction script.

Loads a checkpoint produced by bionne_train.py, runs inference on a
JSON-lines file, and writes a CodaBench-compatible TSV.

Usage:
    python bionne_predict.py \\
        --data  data/eng_dev.txt \\
        --ckpt  outputs/biomedbert_en.pt \\
        -o      outputs/eng_pred.tsv

    # evaluate against gold labels afterwards:
    python score.py --pred outputs/eng_pred.tsv --gold ../data/en/dev/eng-dev-rel.tsv
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from bionne_dataset import BioNNEDataset, get_typed_marker_tokens, pad_collate_fn
from bionne_model import BioNNEModel


# ---------------------------------------------------------------------------
# Core inference — load model once, return raw probabilities
# ---------------------------------------------------------------------------

def _run_inference(
    data_path: str,
    ckpt_path: str,
    max_length: int = 512,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[list[dict], torch.Tensor, dict[str, int]]:
    """Load model, run one forward pass, return (instances, probs, rel2id).

    probs: float32 tensor of shape [N, num_classes] on CPU.
    Callers apply thresholds or argmax on the returned probs — no re-inference needed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_name: str        = ckpt["model_name"]
    rel2id: dict[str, int] = ckpt["rel2id"]
    num_classes: int       = ckpt["num_classes"]
    use_nesting_flag: bool = ckpt.get("use_nesting_flag", True)

    print(f"Backbone:  {model_name}")
    print(f"Checkpoint macro F1 at save: {ckpt.get('macro_f1', float('nan')):.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": get_typed_marker_tokens()}
    )

    dataset = BioNNEDataset(data_path, tokenizer, rel2id, max_length, pad_to_max_length=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate_fn,
    )
    print(f"Running inference on {len(dataset):,} instances...")

    model = BioNNEModel(model_name, num_classes, use_nesting_flag=use_nesting_flag)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    actual_device = next(model.parameters()).device
    print(f"Model device: {actual_device}")
    if device.type == "cuda" and actual_device.type != "cuda":
        raise RuntimeError(
            "CUDA is available but model is on CPU — CUDA context is corrupted. "
            "Restart the Colab runtime (Runtime → Restart runtime) and re-run."
        )

    all_probs: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference", unit="batch"):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                h_pos=batch["h_pos"].to(device),
                t_pos=batch["t_pos"].to(device),
                nesting_flag=batch["nesting_flag"].to(device),
            )
            all_probs.append(torch.softmax(logits, dim=-1).cpu())
            # Detection failsafe from dataset
            all_valid.append(batch["valid_markers"].cpu())

    return dataset.instances, torch.cat(all_probs, dim=0), torch.cat(all_valid, dim=0), rel2id


def _probs_to_df(
    instances: list[dict],
    all_probs: torch.Tensor,
    all_valid: torch.Tensor,
    rel2id: dict[str, int],
    threshold: float | dict[str, float] = 0.0,
    valid_triplets: set[tuple[str, str, str]] | None = None,
    existence_filter: bool = False,
) -> pd.DataFrame:
    """Apply threshold/triplet filtering and build predictions DataFrame.

    existence_filter: if True, use 1 - p(no_relation) as the filter score instead
        of max(positive_class_probs).  When the model spreads uncertainty across
        multiple positive classes (common for rare relations), this criterion is
        more sensitive — it asks "does any relation exist?" rather than "is the
        model confident about the specific type?".  The predicted type is still
        argmax over positive classes.  Only used when threshold > 0.
    """
    id2rel = {v: k for k, v in rel2id.items()}
    no_rel_id = rel2id["no_relation"]
    N = all_probs.size(0)

    # 1. Apply confidence thresholds
    if isinstance(threshold, dict):
        # Per-class thresholds
        thresh_vec = torch.tensor([threshold.get(id2rel[i], 0.0) for i in range(all_probs.size(1))])
        # Mask probabilities that don't meet their class-specific threshold
        mask = all_probs >= thresh_vec
        masked_probs = all_probs * mask
        top_prob, pred_ids_tensor = masked_probs.max(dim=-1)
        # If max is 0 (didn't meet threshold), set to no_relation
        pred_ids_tensor[top_prob == 0] = no_rel_id
    elif threshold > 0:
        if existence_filter:
            # Filter on total positive mass (1 - p_no_rel); predict argmax of positive classes
            exist_score = 1.0 - all_probs[:, no_rel_id]
            pos_probs = all_probs.clone()
            pos_probs[:, no_rel_id] = 0.0
            _, top_cls = pos_probs.max(dim=-1)
            pred_ids_tensor = torch.where(
                exist_score >= threshold,
                top_cls,
                torch.full_like(top_cls, no_rel_id),
            )
        else:
            top_prob, top_cls = all_probs.max(dim=-1)
            pred_ids_tensor = torch.where(
                top_prob >= threshold,
                top_cls,
                torch.full_like(top_cls, no_rel_id),
            )
    else:
        _, pred_ids_tensor = all_probs.max(dim=-1)

    # 2. Truncation Failsafe: override invalid markers to no_relation
    pred_ids_tensor[~all_valid] = no_rel_id
    
    pred_ids = pred_ids_tensor.tolist()

    rows = []
    for inst, pred_id in zip(instances, pred_ids):
        rel_name = id2rel[pred_id]
        
        # 3. Type Pruning: Hallucination check
        if valid_triplets and rel_name != "no_relation":
            triplet = (inst["head_type"], inst["tail_type"], rel_name)
            if triplet not in valid_triplets:
                rel_name = "no_relation"

        rows.append({
            "document_id": inst["doc_id"],
            "relation":    rel_name,
            "head_text":   inst["h"]["name"],
            "head_span":   inst["head_span"],
            "head_type":   inst["head_type"],
            "tail_text":   inst["t"]["name"],
            "tail_span":   inst["tail_span"],
            "tail_type":   inst["tail_type"],
        })
    df = pd.DataFrame(rows)
    return df[df["relation"] != "no_relation"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    data_path: str,
    ckpt_path: str,
    output_path: str,
    max_length: int = 512,
    batch_size: int = 256,
    threshold: float = 0.0,
    num_workers: int = 0,
    existence_filter: bool = False,
) -> str:
    """Run inference and write CodaBench-compatible TSV.

    threshold: minimum score required to predict a positive relation.
        0.0  = argmax (default; fine for labelled dev/train data).
        >0.0 = only emit a relation if score >= threshold.
               Tune with find_threshold() on the dev set in blind mode first.

    existence_filter: if True, score = 1 - p(no_relation) instead of max positive prob.

    Returns the output path for convenience.
    """
    if isinstance(threshold, dict) or threshold > 0:
        mode = "existence (1-p_no_rel)" if existence_filter else "max-class"
        print(f"Confidence threshold: {threshold}  mode: {mode}")

    instances, all_probs, all_valid, rel2id = _run_inference(
        data_path, ckpt_path, max_length, batch_size, num_workers
    )
    valid_triplets = load_valid_triplets()
    pred_df = _probs_to_df(instances, all_probs, all_valid, rel2id, threshold, valid_triplets,
                           existence_filter=existence_filter)

    total = len(instances)
    kept = len(pred_df)
    print(f"Kept {kept:,} positive predictions (filtered {total - kept:,} no_relation)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, sep="\t", index=False)
    print(f"Predictions written to: {output_path}")
    return output_path


def find_threshold(
    blind_dev_data_path: str,
    ckpt_path: str,
    gold_tsv_path: str,
    thresholds: list[float] | None = None,
    max_length: int = 512,
    batch_size: int = 256,
    existence_filter: bool = False,
) -> float:
    """Find the confidence threshold that maximises blind-mode dev macro F1.

    Runs inference ONCE, then sweeps thresholds on the stored probabilities
    (each sweep is O(N) CPU-only — takes milliseconds).

    blind_dev_data_path: JSON-lines prepared from entity TSV (all pairs).
    gold_tsv_path:       gold relation TSV (eng-dev-rel.tsv / rus-dev-rel.tsv).
    existence_filter:    if True, sweep over 1-p(no_relation) scores instead of max-class.
    """
    from score import evaluate, load_gold

    if thresholds is None:
        thresholds = [i/100 for i in range(101)]

    gold_df = load_gold(Path(gold_tsv_path))

    # ---- single inference pass ----
    instances, all_probs, all_valid, rel2id = _run_inference(
        blind_dev_data_path, ckpt_path, max_length, batch_size
    )

    valid_triplets = load_valid_triplets()

    mode_str = "existence(1-p_no_rel)" if existence_filter else "max-class"
    print(f"\nMode: {mode_str}")
    print(f"\n{'Threshold':>10}  {'#Preds':>8}  {'MacroF1':>8}  {'Precision':>10}  {'Recall':>10}")
    print("-" * 58)

    best_f1, best_t = -1.0, 0.0
    for t in thresholds:
        pred_df = _probs_to_df(instances, all_probs, all_valid, rel2id, t, valid_triplets,
                               existence_filter=existence_filter)
        results = evaluate(pred_df, gold_df)
        f1 = results["macro_f1"]
        mp = results["micro_precision"]
        mr = results["micro_recall"]
        marker = "  ◄" if f1 > best_f1 else ""
        print(f"{t:>10.2f}  {len(pred_df):>8,}  {f1:>8.4f}  {mp:>10.4f}  {mr:>10.4f}{marker}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest threshold: {best_t}  (blind dev macro F1 = {best_f1:.4f})")
    return best_t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BioNNE-R prediction")
    parser.add_argument("--data",  required=True,               help="JSON-lines input file")
    parser.add_argument("--ckpt",  required=True,               help="Checkpoint (.pt)")
    parser.add_argument("-o", "--output", default="outputs/pred.tsv")
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--threshold",   type=float, default=0.0,
                        help="Min softmax prob to emit a positive relation (0=argmax). "
                             "Use >0 for blind test to reduce false positives.")
    args = parser.parse_args()

    predict(
        data_path=args.data,
        ckpt_path=args.ckpt,
        output_path=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()


def load_valid_triplets() -> set[tuple[str, str, str]]:
    """Load valid (HeadType, TailType, Relation) triplets from file."""
    path = Path(__file__).parent / "data/valid_triplets.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {tuple(t) for t in data}
