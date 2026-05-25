"""
Dataset and tokenization utilities for BioNNE-R typed entity marker approach.

Reads the JSON-lines format produced by prepare_data.py and applies:
  - Typed entity markers:  <H:TYPE> head_text </H:TYPE>  <T:TYPE> tail_text </T:TYPE>
  - Nesting detection:     binary flag when one entity span contains the other
  - Class-weight computation for inverse-frequency weighted loss
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# All 8 entity types present in NEREL-BIO
ENTITY_TYPES = [
    "ANATOMY", "CHEM", "DEVICE", "DISO",
    "FINDING", "INJURY_POISONING", "LABPROC", "PHYS",
]

# 14 relation types + no_relation (must match prepare_data.py / rel2id.json)
RELATION_TYPES = [
    "ABBREVIATION", "ALTERNATIVE_NAME", "SUBCLASS_OF", "PART_OF",
    "TREATED_USING", "ORIGINS_FROM", "TO_DETECT_OR_STUDY", "AFFECTS",
    "HAS_CAUSE", "APPLIED_TO", "USED_IN", "ASSOCIATED_WITH",
    "PHYSIOLOGY_OF", "FINDING_OF", "no_relation",
]


# ---------------------------------------------------------------------------
# Special-token helpers
# ---------------------------------------------------------------------------

def get_typed_marker_tokens() -> list[str]:
    """Return all 32 special tokens needed for typed entity markers.

    Format: <H:TYPE>, </H:TYPE>, <T:TYPE>, </T:TYPE> for every entity type.
    These must be added to the tokenizer before training.
    """
    tokens: list[str] = []
    for et in ENTITY_TYPES:
        tokens += [f"<H:{et}>", f"</H:{et}>", f"<T:{et}>", f"</T:{et}>"]
    return tokens


# ---------------------------------------------------------------------------
# Marker insertion
# ---------------------------------------------------------------------------

def insert_typed_markers(
    text: str,
    h_start: int, h_end: int, head_type: str,
    t_start: int, t_end: int, tail_type: str,
) -> str:
    """Insert typed entity markers into *text* at the given character offsets.

    Handles all three span configurations (external, nested, cross-nested)
    by sorting insertions right-to-left so earlier offsets are not disturbed.

    At equal positions (e.g. adjacent entities where h_end == t_start) open
    markers are inserted before close markers so the result is:
        ... </H:TYPE><T:TYPE> ...   (close then open, reading left-to-right)
    """
    # (position, is_open, marker_text)
    insertions = [
        (h_start, True,  f"<H:{head_type}>"),
        (h_end,   False, f"</H:{head_type}>"),
        (t_start, True,  f"<T:{tail_type}>"),
        (t_end,   False, f"</T:{tail_type}>"),
    ]
    # Sort descending by position; at the same position process open markers
    # first — they land rightmost in the final string, i.e. closes appear left.
    insertions.sort(key=lambda x: (-x[0], 0 if x[1] else 1))

    result = text
    for pos, _, marker in insertions:
        result = result[:pos] + marker + result[pos:]
    return result


# ---------------------------------------------------------------------------
# Nesting detection
# ---------------------------------------------------------------------------

def is_nested(h_start: int, h_end: int, t_start: int, t_end: int) -> bool:
    """Return True if either entity span fully contains the other."""
    head_contains_tail = h_start <= t_start and t_end <= h_end
    tail_contains_head = t_start <= h_start and h_end <= t_end
    return head_contains_tail or tail_contains_head


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def pad_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length examples by padding to the batch maximum length.

    Use this as DataLoader(collate_fn=pad_collate_fn) when BioNNEDataset is
    created with pad_to_max_length=False.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)
    bs = len(batch)
    input_ids = torch.zeros(bs, max_len, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "h_pos":          torch.stack([item["h_pos"]          for item in batch]),
        "t_pos":          torch.stack([item["t_pos"]          for item in batch]),
        "valid_markers":  torch.stack([item.get("valid_markers", torch.tensor(True)) for item in batch]),
        "nesting_flag":   torch.stack([item["nesting_flag"]   for item in batch]),
        "label":          torch.stack([item["label"]          for item in batch]),
    }


class BioNNEDataset(Dataset):
    """PyTorch Dataset for BioNNE-R relation extraction.

    Each example becomes:
        input_ids       – tokenised marked text
        attention_mask  – 1 for real tokens, 0 for padding
        h_pos           – token index of the <H:TYPE> start marker
        t_pos           – token index of the <T:TYPE> start marker
        nesting_flag    – 1.0 if spans are nested, 0.0 otherwise
        label           – integer class id from rel2id

    When pad_to_max_length=False, input_ids/attention_mask are variable-length
    (not padded). Use pad_collate_fn as the DataLoader's collate_fn in that case.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        rel2id: dict[str, int],
        max_length: int = 256,
        pad_to_max_length: bool = True,
    ):
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.instances: list[dict] = []
        self._load(data_path)

        self._h_marker_ids = {
            et: tokenizer.convert_tokens_to_ids(f"<H:{et}>")
            for et in ENTITY_TYPES
        }
        self._t_marker_ids = {
            et: tokenizer.convert_tokens_to_ids(f"<T:{et}>")
            for et in ENTITY_TYPES
        }

        # Inference path: pre-tokenize everything in one batch call.
        # Individual tokenizer calls on Colab have ~18ms Python overhead each;
        # a single batch call amortizes that overhead → 10x speedup on large test sets.
        self._cache: list[dict] | None = None
        if not pad_to_max_length:
            self._precompute_cache()

    def _load(self, data_path: str) -> None:
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instances.append(json.loads(line))

    # Tokenize in chunks to avoid SIGSEGV in the Rust backend when passing
    # 800K+ strings at once (causes buffer overflow / OOM in the C++ allocator).
    _TOKENIZE_CHUNK = 8_000

    def _precompute_cache(self) -> None:
        """Batch-tokenize all instances in fixed-size chunks.

        Chunked Rust-backed calls keep per-item Python overhead low while
        avoiding the OOM / SIGSEGV that occurs when the full 800K-instance
        list is passed in one shot.
        """
        N = len(self.instances)
        CHUNK = self._TOKENIZE_CHUNK
        print(f"Pre-tokenizing {N:,} instances (chunk={CHUNK:,})…", flush=True)

        self._cache = [None] * N  # type: ignore[list-item]

        for chunk_start in range(0, N, CHUNK):
            chunk_end = min(chunk_start + CHUNK, N)
            chunk = self.instances[chunk_start:chunk_end]

            marked_texts: list[str] = []
            h_types: list[str] = []
            t_types: list[str] = []
            nesting_flags: list[float] = []
            label_ids: list[int] = []

            for inst in chunk:
                h_start, h_end = inst["h"]["pos"]
                t_start, t_end = inst["t"]["pos"]
                ht, tt = inst["head_type"], inst["tail_type"]
                marked_texts.append(
                    insert_typed_markers(inst["text"], h_start, h_end, ht, t_start, t_end, tt)
                )
                h_types.append(ht)
                t_types.append(tt)
                nesting_flags.append(float(is_nested(h_start, h_end, t_start, t_end)))
                label_ids.append(
                    self.rel2id.get(inst.get("relation", "no_relation"), self.rel2id["no_relation"])
                )

            enc = self.tokenizer(
                marked_texts,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

            for j, global_i in enumerate(range(chunk_start, chunk_end)):
                ids  = enc["input_ids"][j]
                mask = enc["attention_mask"][j]
                h_mid = self._h_marker_ids.get(h_types[j], -1)
                t_mid = self._t_marker_ids.get(t_types[j], -1)
                h_pos = ids.index(h_mid) if h_mid in ids else 0
                t_pos = ids.index(t_mid) if t_mid in ids else 0
                valid = (h_mid in ids) and (t_mid in ids)

                self._cache[global_i] = {
                    "input_ids":      torch.tensor(ids,              dtype=torch.long),
                    "attention_mask": torch.tensor(mask,             dtype=torch.long),
                    "h_pos":          torch.tensor(h_pos,            dtype=torch.long),
                    "t_pos":          torch.tensor(t_pos,            dtype=torch.long),
                    "valid_markers":  torch.tensor(valid,            dtype=torch.bool),
                    "nesting_flag":   torch.tensor(nesting_flags[j], dtype=torch.float),
                    "label":          torch.tensor(label_ids[j],     dtype=torch.long),
                }

            if chunk_start % (CHUNK * 10) == 0 and chunk_start > 0:
                print(f"  {chunk_start:,}/{N:,}…", flush=True)

        print(f"Pre-tokenization done.", flush=True)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            return self._cache[idx]

        inst = self.instances[idx]

        text = inst["text"]
        h_start, h_end = inst["h"]["pos"]
        t_start, t_end = inst["t"]["pos"]
        head_type = inst["head_type"]
        tail_type = inst["tail_type"]
        relation = inst.get("relation", "no_relation")

        marked_text = insert_typed_markers(
            text, h_start, h_end, head_type,
            t_start, t_end, tail_type,
        )

        nested = is_nested(h_start, h_end, t_start, t_end)

        encoding = self.tokenizer(
            marked_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        h_marker_id = self._h_marker_ids.get(head_type, -1)
        t_marker_id = self._t_marker_ids.get(tail_type, -1)
        ids_list = input_ids.tolist()
        h_pos = ids_list.index(h_marker_id) if h_marker_id in ids_list else 0
        t_pos = ids_list.index(t_marker_id) if t_marker_id in ids_list else 0

        label = self.rel2id.get(relation, self.rel2id["no_relation"])

        # Detection failsafe
        valid = (h_marker_id in ids_list) and (t_marker_id in ids_list)

        return {
            "input_ids":    input_ids,
            "attention_mask": attention_mask,
            "h_pos":        torch.tensor(h_pos,      dtype=torch.long),
            "t_pos":        torch.tensor(t_pos,      dtype=torch.long),
            "valid_markers": torch.tensor(valid,      dtype=torch.bool),
            "nesting_flag": torch.tensor(float(nested), dtype=torch.float),
            "label":        torch.tensor(label,      dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Class-weight helper (call once before training)
    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights: w_c = N / (K * n_c).

        Classes absent from training data get weight 1.0.
        Returns a tensor of shape [num_classes] suitable for
        nn.CrossEntropyLoss(weight=...).
        """
        counts: Counter = Counter()
        for inst in self.instances:
            rel = inst.get("relation", "no_relation")
            if rel in self.rel2id:
                counts[self.rel2id[rel]] += 1

        n_classes = len(self.rel2id)
        total = sum(counts.values())
        weights = []
        no_rel_id = self.rel2id.get("no_relation", -1)
        for i in range(n_classes):
            if i == no_rel_id:
                # Distribution mismatch: training has 1:3 ratio, test has ~1:100.
                # Standard weighting penalizes negatives (0.08) too much.
                # Clamping to 1.0 keeps negatives relevant during training.
                weights.append(1.0)
                continue
            n_c = counts.get(i, 0)
            weights.append(total / (n_classes * n_c) if n_c > 0 else 1.0)
        return torch.tensor(weights, dtype=torch.float)

    def relation_distribution(self) -> dict[str, int]:
        """Return {relation_name: count} for logging / inspection."""
        id2rel = {v: k for k, v in self.rel2id.items()}
        counts: Counter = Counter()
        for inst in self.instances:
            counts[inst.get("relation", "no_relation")] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
