#!/usr/bin/env python3
"""
BioNNE-R improved training script.

Improvements over the OpenNRE baseline:
  1. Backbone swap  – any HuggingFace model (default: BiomedBERT for EN,
                      mDeBERTa-v3-base for RU/bilingual)
  2. Typed entity markers  – <H:TYPE> / <T:TYPE> start-marker representations
  3. Nesting flag          – binary feature concatenated before classifier head
  4. Class-weighted loss   – inverse-frequency weights via CrossEntropyLoss

Usage:
    # English (Subtask 1)
    python bionne_train.py \\
        --train data/eng_train.txt --dev data/eng_dev.txt \\
        --rel2id data/rel2id.json \\
        --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \\
        --ckpt outputs/biomedbert_en.pt

    # Russian (Subtask 2)
    python bionne_train.py \\
        --train data/rus_train.txt --dev data/rus_dev.txt \\
        --rel2id data/rel2id.json \\
        --model microsoft/mdeberta-v3-base \\
        --ckpt outputs/mdeberta_ru.pt

    # Bilingual (Subtask 3)  — concatenate EN+RU JSON-lines into one file first
    python bionne_train.py \\
        --train data/bilingual_train.txt --dev data/bilingual_dev.txt \\
        --rel2id data/rel2id.json \\
        --model microsoft/mdeberta-v3-base \\
        --ckpt outputs/mdeberta_bilingual.pt
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # suppress "weights not initialized" noise

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Allow running from any working directory (e.g. Colab after cd baseline/)
sys.path.insert(0, str(Path(__file__).parent))

from bionne_dataset import BioNNEDataset, get_typed_marker_tokens, pad_collate_fn
from bionne_model import BioNNEModel


def log(msg: str) -> None:
    """Print with immediate flush — works correctly inside Colab cells."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_macro_f1(
    preds: list[int],
    labels: list[int],
    num_classes: int,
    ignore_class: int | None = None,
) -> float:
    """Macro-averaged F1 over classes that have at least one gold instance.

    `ignore_class` (typically the no_relation id) is excluded from the average,
    matching the official BioNNE-R evaluation metric.
    """
    f1_scores: list[float] = []
    for c in range(num_classes):
        if c == ignore_class:
            continue
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        if tp + fn == 0:          # class absent from gold → skip
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


@torch.no_grad()
def evaluate(
    model: BioNNEModel,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    no_rel_id: int,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            h_pos=batch["h_pos"].to(device),
            t_pos=batch["t_pos"].to(device),
            nesting_flag=batch["nesting_flag"].to(device),
        )
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

    macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes, ignore_class=no_rel_id)
    return macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Training entry point (importable by Colab notebook)
# ---------------------------------------------------------------------------

def train(
    train_path: str,
    dev_path: str,
    rel2id_path: str,
    ckpt_path: str = "outputs/model.pt",
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    max_length: int = 512,
    batch_size: int = 16,
    lr: float = 2e-5,
    epochs: int = 10,
    warmup_steps: int = 300,
    use_class_weights: bool = True,
    use_nesting_flag: bool = True,
    seed: int = 42,
    gdrive_dir: str | None = None,
    resume: bool = False,
    use_fp16: bool = False,
    gradient_checkpointing: bool = False,
) -> float:
    """Train a BioNNE-R model and save the best checkpoint.

    Returns the best dev macro F1 achieved during training.

    Resume behaviour
    ----------------
    Every completed epoch writes a ``<ckpt_path>.resume`` file containing the
    current model weights, optimiser and scheduler state, and the best F1 so
    far.  With ``resume=True`` that file is loaded automatically so training
    continues from epoch N+1 with exactly the same LR schedule.  If the file
    is absent, training starts from the HuggingFace pretrained weights as usual.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Backbone: {model_name}")

    # ---- rel2id ----
    with open(rel2id_path) as f:
        rel2id: dict[str, int] = json.load(f)
    num_classes = len(rel2id)
    no_rel_id = rel2id.get("no_relation", -1)

    # ---- tokenizer ----
    log("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = get_typed_marker_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    log(f"Vocabulary size after adding markers: {len(tokenizer)}")

    # ---- datasets ----
    log("Loading datasets…")
    train_ds = BioNNEDataset(train_path, tokenizer, rel2id, max_length, pad_to_max_length=False)
    dev_ds   = BioNNEDataset(dev_path,   tokenizer, rel2id, max_length, pad_to_max_length=False)
    log(f"Train: {len(train_ds)} instances | Dev: {len(dev_ds)} instances")
    log(f"Train distribution: {train_ds.relation_distribution()}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, collate_fn=pad_collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

    # ---- class weights ----
    class_weights = None
    if use_class_weights:
        class_weights = train_ds.get_class_weights().to(device)
        id2rel = {v: k for k, v in rel2id.items()}
        weight_log = {id2rel[i]: f"{w:.2f}" for i, w in enumerate(class_weights.tolist())}
        log(f"Class weights: {weight_log}")

    # ---- model ----
    log("Loading model…")
    model = BioNNEModel(
        model_name, num_classes,
        use_nesting_flag=use_nesting_flag,
        gradient_checkpointing=gradient_checkpointing,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    if gradient_checkpointing:
        log("Gradient checkpointing: enabled")

    # ---- optimiser & scheduler ----
    # total_steps is always based on the full run so the LR schedule is identical
    # whether we start fresh or resume mid-way.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if use_fp16 and device.type == "cuda" else None
    if use_fp16:
        log(f"Mixed precision: fp16 ({'enabled' if scaler else 'skipped – CPU'})")

    # ---- resume ----
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_path + ".resume"
    start_epoch = 1
    best_macro_f1 = 0.0

    if resume and os.path.exists(resume_path):
        log(f"Loading resume checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch   = state["epoch"] + 1
        best_macro_f1 = state["best_macro_f1"]
        log(f"Resuming from epoch {start_epoch}  (best F1 so far: {best_macro_f1:.4f})")
    elif resume:
        log(f"No resume file found at {resume_path} — starting fresh.")

    if start_epoch > epochs:
        log(f"All {epochs} epochs already completed.")
        return best_macro_f1

    # ---- training loop ----
    log(f"\nStarting training: epochs {start_epoch}–{epochs} × {len(train_loader)} steps")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step, batch in enumerate(pbar, 1):
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    h_pos=batch["h_pos"].to(device),
                    t_pos=batch["t_pos"].to(device),
                    nesting_flag=batch["nesting_flag"].to(device),
                )
                loss = criterion(logits, batch["label"].to(device))

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()


            if step % 500 == 0:
                resume_state = {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_macro_f1": best_macro_f1,
                }
                if scaler is not None:
                    resume_state["scaler_state_dict"] = scaler.state_dict()
                torch.save(resume_state, resume_path)
                # print(f"  [Step {step}] Intermediate resume state saved.")
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / step:.4f}"})

        avg_loss = running_loss / len(train_loader)
        macro_f1, _, _ = evaluate(model, dev_loader, device, num_classes, no_rel_id)
        best_marker = " ← best" if macro_f1 > best_macro_f1 else ""
        log(f"Epoch {epoch:>2}  loss={avg_loss:.4f}  dev_macro_f1={macro_f1:.4f}{best_marker}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "model_name":       model_name,
                    "rel2id":           rel2id,
                    "num_classes":      num_classes,
                    "use_nesting_flag": use_nesting_flag,
                    "macro_f1":         macro_f1,
                },
                ckpt_path,
            )
            if gdrive_dir:
                os.makedirs(gdrive_dir, exist_ok=True)
                drive_dest = os.path.join(gdrive_dir, Path(ckpt_path).name)
                shutil.copy2(ckpt_path, drive_dest)
                log(f"           → mirrored to Drive: {drive_dest}")

        # Save resume state every epoch so a future restart can pick up here.
        epoch_resume_state = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_f1":       best_macro_f1,
        }
        if scaler is not None:
            epoch_resume_state["scaler_state_dict"] = scaler.state_dict()
        torch.save(epoch_resume_state, resume_path)
        if gdrive_dir:
            os.makedirs(gdrive_dir, exist_ok=True)
            shutil.copy2(resume_path, os.path.join(gdrive_dir, Path(resume_path).name))

    log(f"\nTraining complete.  Best dev macro F1: {best_macro_f1:.4f}")
    log(f"Checkpoint: {ckpt_path}")
    return best_macro_f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioNNE-R training: typed markers + class weights"
    )
    parser.add_argument("--train",       required=True,  help="Training JSON-lines file")
    parser.add_argument("--dev",         required=True,  help="Dev JSON-lines file")
    parser.add_argument("--rel2id",      required=True,  help="rel2id.json")
    parser.add_argument("--ckpt",        default="outputs/model.pt")
    parser.add_argument(
        "--model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        help=(
            "HuggingFace model id.  Recommended:\n"
            "  EN:           microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext\n"
            "  RU/bilingual: microsoft/mdeberta-v3-base"
        ),
    )
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--warmup_steps", type=int,   default=300)
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Disable inverse-frequency class weighting")
    parser.add_argument("--no_nesting_flag",  action="store_true",
                        help="Disable nesting flag feature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <ckpt>.resume if it exists")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable fp16 mixed-precision training (recommended for mDeBERTa)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory (recommended for mDeBERTa)")
    args = parser.parse_args()

    train(
        train_path=args.train,
        dev_path=args.dev,
        rel2id_path=args.rel2id,
        ckpt_path=args.ckpt,
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        use_class_weights=not args.no_class_weights,
        use_nesting_flag=not args.no_nesting_flag,
        seed=args.seed,
        resume=args.resume,
        use_fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
