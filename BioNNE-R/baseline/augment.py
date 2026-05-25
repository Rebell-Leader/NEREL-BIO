#!/usr/bin/env python3
"""
LLM data augmentation for BioNNE-R rare relation types.

Reads the JSON-lines training file (output of prepare_data.py), samples
examples from rare classes, and calls the OpenAI API to generate paraphrases.
Augmented examples are written as additional JSON-lines in the same format,
ready to be appended to the training file before re-running bionne_train.py.

Target classes (from EN training distribution):
    APPLIED_TO      43 examples  → target 150
    USED_IN         89 examples  → target 150
    ABBREVIATION    97 examples  → target 150
    TREATED_USING  101 examples  → target 150

Strategy:
    - Instruct the LLM to rephrase only the *surrounding context*, keeping the
      entity texts character-for-character identical.
    - Locate entities in the paraphrase via exact string search.
    - Discard any paraphrase where entity text cannot be found (noise filter).
    - Optionally validate with back-prediction using a trained model checkpoint.

Usage:
    export OPENAI_API_KEY=sk-...
    python augment.py \\
        --data data/eng_train.txt \\
        --output data/augmented_en.jsonl \\
        --target 150 \\
        [--ckpt outputs/biomedbert_en.pt]   # optional back-validation

    # Then merge and retrain:
    cat data/eng_train.txt data/augmented_en.jsonl > data/eng_train_aug.txt
    python bionne_train.py --train data/eng_train_aug.txt ...
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

RARE_CLASSES = ["APPLIED_TO", "USED_IN", "ABBREVIATION", "TREATED_USING"]

SYSTEM_PROMPT = """\
You are a biomedical NLP expert specialising in relation extraction.
Your task is to paraphrase sentences from biomedical abstracts.
Rules you MUST follow:
  1. Keep ALL entity texts EXACTLY as given — do not change a single character.
  2. Rephrase only the surrounding words and sentence structure.
  3. The paraphrase must remain grammatically correct English.
  4. The paraphrase must still clearly express the given relation between the entities.
  5. Return ONLY the paraphrased sentence — no explanation, no extra text."""

USER_TEMPLATE = """\
Relation type: {relation}
Head entity ({head_type}): "{head_text}"
Tail entity ({tail_type}): "{tail_text}"

Original sentence:
{sentence}

Write {n} paraphrases (one per line, numbered 1. 2. 3. …):"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_instances(data_path: str) -> list[dict]:
    instances = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def find_span(text: str, entity_text: str) -> tuple[int, int] | None:
    """Return (start, end) of the first exact occurrence of entity_text in text."""
    idx = text.find(entity_text)
    if idx == -1:
        return None
    return idx, idx + len(entity_text)


def call_openai(
    client,
    relation: str,
    head_text: str,
    head_type: str,
    tail_text: str,
    tail_type: str,
    sentence: str,
    n: int = 3,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Call OpenAI and return a list of paraphrase strings (may be empty on error)."""
    prompt = USER_TEMPLATE.format(
        relation=relation,
        head_text=head_text,
        head_type=head_type,
        tail_text=tail_text,
        tail_type=tail_type,
        sentence=sentence,
        n=n,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.8,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  OpenAI error: {e}", file=sys.stderr)
        return []

    # Parse numbered list:  "1. sentence\n2. sentence\n..."
    paraphrases = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading number + dot/paren
        if line and line[0].isdigit():
            dot = line.find(".")
            paren = line.find(")")
            sep = min(x for x in [dot, paren] if x > 0) if (dot > 0 or paren > 0) else -1
            if sep > 0:
                line = line[sep + 1:].strip()
        if line:
            paraphrases.append(line)
    return paraphrases


def make_augmented_instance(inst: dict, paraphrase: str) -> dict | None:
    """Build a new JSON-lines instance from a paraphrase.

    Returns None if either entity cannot be located in the paraphrase.
    """
    head_text = inst["h"]["name"]
    tail_text = inst["t"]["name"]

    h_span = find_span(paraphrase, head_text)
    t_span = find_span(paraphrase, tail_text)

    if h_span is None or t_span is None:
        return None

    # Avoid degenerate cases where both entities map to the same span
    if h_span == t_span:
        return None

    return {
        "text":      paraphrase,
        "h":         {"name": head_text,  "pos": list(h_span)},
        "t":         {"name": tail_text,  "pos": list(t_span)},
        "relation":  inst["relation"],
        "doc_id":    inst["doc_id"] + "_aug",
        "head_span": f"{h_span[0]}-{h_span[1]}",
        "tail_span": f"{t_span[0]}-{t_span[1]}",
        "head_type": inst["head_type"],
        "tail_type": inst["tail_type"],
    }


# ---------------------------------------------------------------------------
# Optional back-validation with a trained checkpoint
# ---------------------------------------------------------------------------

def load_validator(ckpt_path: str):
    """Load model + tokenizer for back-validation (returns None if unavailable)."""
    try:
        import torch
        from transformers import AutoTokenizer
        from bionne_model import BioNNEModel
        from bionne_dataset import get_typed_marker_tokens, insert_typed_markers, is_nested

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_name = ckpt["model_name"]
        rel2id     = ckpt["rel2id"]
        id2rel     = {v: k for k, v in rel2id.items()}

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": get_typed_marker_tokens()}
        )

        model = BioNNEModel(model_name, ckpt["num_classes"],
                            use_nesting_flag=ckpt.get("use_nesting_flag", True))
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return {"model": model, "tokenizer": tokenizer, "id2rel": id2rel,
                "rel2id": rel2id, "insert": insert_typed_markers, "nested": is_nested}
    except Exception as e:
        print(f"Back-validation unavailable: {e}", file=sys.stderr)
        return None


def validate_instance(validator: dict, inst: dict) -> bool:
    """Return True if the model predicts the correct relation for inst."""
    import torch
    model     = validator["model"]
    tokenizer = validator["tokenizer"]
    id2rel    = validator["id2rel"]
    rel2id    = validator["rel2id"]

    h_start, h_end = inst["h"]["pos"]
    t_start, t_end = inst["t"]["pos"]

    marked = validator["insert"](
        inst["text"], h_start, h_end, inst["head_type"],
        t_start, t_end, inst["tail_type"],
    )
    enc = tokenizer(marked, max_length=256, truncation=True,
                    padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    h_marker_id = tokenizer.convert_tokens_to_ids(f"<H:{inst['head_type']}>")
    t_marker_id = tokenizer.convert_tokens_to_ids(f"<T:{inst['tail_type']}>")
    ids = input_ids[0].tolist()
    h_pos = ids.index(h_marker_id) if h_marker_id in ids else 0
    t_pos = ids.index(t_marker_id) if t_marker_id in ids else 0

    nested = validator["nested"](h_start, h_end, t_start, t_end)
    nesting_flag = torch.tensor([[float(nested)]])

    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attn_mask,
            h_pos=torch.tensor([h_pos]),
            t_pos=torch.tensor([t_pos]),
            nesting_flag=nesting_flag,
        )
    pred_rel = id2rel[logits.argmax(-1).item()]
    return pred_rel == inst["relation"]


# ---------------------------------------------------------------------------
# Main augmentation loop
# ---------------------------------------------------------------------------

def augment(
    data_path: str,
    output_path: str,
    target_per_class: int = 150,
    rare_classes: list[str] | None = None,
    paraphrases_per_call: int = 3,
    openai_model: str = "gpt-4o-mini",
    ckpt_path: str | None = None,
    seed: int = 42,
    rate_limit_sleep: float = 0.5,
) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not found. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()   # reads OPENAI_API_KEY from environment
    rng    = random.Random(seed)
    target_classes = rare_classes or RARE_CLASSES

    # ---- load and group training data ----
    instances = load_instances(data_path)
    by_class: dict[str, list[dict]] = {}
    for inst in instances:
        rel = inst.get("relation", "no_relation")
        if rel in target_classes:
            by_class.setdefault(rel, []).append(inst)

    for rel, exs in by_class.items():
        print(f"  {rel}: {len(exs)} existing examples (target: {target_per_class})")

    # ---- optional back-validator ----
    validator = load_validator(ckpt_path) if ckpt_path else None
    if validator:
        print("Back-validation enabled: only keeping examples the model predicts correctly.")

    # ---- augmentation ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for rel in target_classes:
            pool = by_class.get(rel, [])
            if not pool:
                print(f"  {rel}: no examples in training data — skipping.")
                continue

            n_existing = len(pool)
            n_needed   = max(0, target_per_class - n_existing)
            if n_needed == 0:
                print(f"  {rel}: already at or above target — skipping.")
                continue

            print(f"\n[{rel}] Generating ~{n_needed} augmented examples …")
            generated = 0

            while generated < n_needed:
                inst = rng.choice(pool)
                paraphrases = call_openai(
                    client,
                    relation=inst["relation"],
                    head_text=inst["h"]["name"],
                    head_type=inst["head_type"],
                    tail_text=inst["t"]["name"],
                    tail_type=inst["tail_type"],
                    sentence=inst["text"],
                    n=paraphrases_per_call,
                    model=openai_model,
                )
                time.sleep(rate_limit_sleep)   # gentle rate limiting

                for para in paraphrases:
                    if generated >= n_needed:
                        break
                    aug = make_augmented_instance(inst, para)
                    if aug is None:
                        continue   # entity text not preserved → discard
                    if validator and not validate_instance(validator, aug):
                        continue   # back-validation failed → discard
                    out_f.write(json.dumps(aug, ensure_ascii=False) + "\n")
                    generated += 1

                print(f"  {rel}: {generated}/{n_needed}", end="\r")

            print(f"  {rel}: wrote {generated} augmented examples")
            total_written += generated

    print(f"\nDone. Total augmented instances written: {total_written}")
    print(f"Output: {output_path}")
    print(f"\nTo merge with training data:\n"
          f"  cat {data_path} {output_path} > data/eng_train_aug.txt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM data augmentation for rare BioNNE-R relation types"
    )
    parser.add_argument(
        "--data",   required=True,
        help="JSON-lines training file (output of prepare_data.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON-lines file for augmented examples",
    )
    parser.add_argument(
        "--target", type=int, default=150,
        help="Target number of examples per rare class (default: 150)",
    )
    parser.add_argument(
        "--classes", nargs="+", default=RARE_CLASSES,
        help=f"Relation types to augment (default: {RARE_CLASSES})",
    )
    parser.add_argument(
        "--paraphrases", type=int, default=3,
        help="Paraphrases per API call (default: 3)",
    )
    parser.add_argument(
        "--openai_model", default="gpt-4o-mini",
        help="OpenAI chat model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Trained checkpoint for back-validation (optional)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        help="Seconds to sleep between API calls (default: 0.5)",
    )
    args = parser.parse_args()

    augment(
        data_path=args.data,
        output_path=args.output,
        target_per_class=args.target,
        rare_classes=args.classes,
        paraphrases_per_call=args.paraphrases,
        openai_model=args.openai_model,
        ckpt_path=args.ckpt,
        seed=args.seed,
        rate_limit_sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
