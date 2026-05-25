# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioNNE-R is a **biomedical named entity relation extraction** shared task at BioASQ 2026. The goal is to classify relations between pre-annotated entity pairs in biomedical texts (14 relation types, 8 entity types) in English and Russian. Primary metric is **macro-averaged F1**.

Three subtasks:
- **Subtask 1**: English only
- **Subtask 2**: Russian only
- **Subtask 3**: Bilingual (combined training)

## Setup

```bash
pip install torch>=2.0.0 transformers nltk pandas scikit-learn
pip install git+https://github.com/thunlp/OpenNRE.git
cd baseline && python patch_opennre.py  # Fix UTF-8, AdamW import, multiprocessing
```

## Pipeline Commands

All commands run from `baseline/`:

```bash
# 1. Prepare training data (TSV + raw texts → OpenNRE JSON-lines)
python prepare_data.py ../data/en/train/eng-train-rel.tsv ../data/en/train/texts/ \
    -o data/eng_train.txt --entities ../data/en/train/eng-train-ent.tsv \
    --neg-ratio 3 --rel2id data/rel2id.json

# 2. Prepare dev data
python prepare_data.py ../data/en/dev/eng-dev-rel.tsv ../data/en/dev/texts/ \
    -o data/eng_dev.txt --entities ../data/en/dev/eng-dev-ent.tsv

# 3. Train
python baseline.py train --train data/eng_train.txt --dev data/eng_dev.txt \
    --rel2id data/rel2id.json --ckpt outputs/eng_model.pth.tar \
    [--model_name bert-base-multilingual-cased] [--epochs 10] [--lr 2e-5] [--batch_size 16]

# 4. Predict (generates CodaBench-compatible TSV, auto-drops no_relation rows)
python baseline.py predict --data data/eng_dev.txt --rel2id data/rel2id.json \
    --ckpt outputs/eng_model.pth.tar -o outputs/eng_pred.tsv

# 5. Evaluate (macro F1 is primary metric)
python score.py --pred outputs/eng_pred.tsv --gold ../data/en/dev/eng-dev-rel.tsv
```

For blind prediction (test set with entity annotations only, no gold relations):
```bash
python prepare_data.py eng-test-ent.tsv texts/ -o data/eng_test.txt
# Then run predict as above
```

## Data Formats

**Entity TSV** (`*-ent.tsv`): `document_id | entity_type | entity_text | entity_span`

**Relation TSV** (`*-rel.tsv`): `document_id | relation | head_text | head_span | head_type | tail_text | tail_span | tail_type`

Spans are character offsets `start-end` into the raw `.txt` files in `texts/`.

## Architecture

```
baseline/
├── prepare_data.py   # TSV + texts → OpenNRE JSON-lines; handles nested entities, neg sampling
├── baseline.py       # Train/predict using OpenNRE (BERT entity-encoder + SoftmaxNN)
├── patch_opennre.py  # Post-install fixes for OpenNRE compatibility
├── score.py          # Macro/micro F1 evaluation
└── baseline.ipynb    # Self-contained notebook version of full pipeline
data/
├── en/{train,dev}/   # English splits with TSVs and texts/
└── ru/{train,dev}/   # Russian splits (much larger: ~37K train entities)
```

**Baseline model**: `bert-base-multilingual-cased` with entity markers, OpenNRE SoftmaxNN head.

**Baseline performance**: English 0.6944, Russian 0.7166, Bilingual 0.7211 macro F1.

## Key Technical Details

- **Nested entities**: ~15% of relations involve nested entity spans. Standard entity masking fails; `prepare_data.py` uses interleaved entity markers to handle this.
- `prepare_data.py` auto-detects labeled vs. blind input (relation TSV vs. entity TSV).
- `rel2id.json` maps 15 classes (14 relations + `no_relation`); generated automatically on first `prepare_data.py` run with `--rel2id`.
- Sentence segmentation uses NLTK punkt tokenizer.
- `patch_opennre.py` must be re-run after any OpenNRE reinstall.

## Known Improvement Directions

Per the strategy guide in `hse_nlp/`:
1. Swap encoder to domain-specific models (PubMedBERT, BiomedBERT) — biggest gain for English
2. Typed entity markers with nesting-aware encoding
3. Class-weighted loss + tuning negative sampling ratio (controls `no_relation` prevalence)

Domain-specific BERT variants outperform `bert-base-multilingual-cased` by ~10-15 F1 points on biomedical RE (from BioNNE-L 2025 results). LLM reranking has not helped in prior iterations.

## Competition Dates

- Dev phase ends / evaluation starts: **April 30, 2026**
- Test predictions due: **May 14, 2026**
- Submission via CodaBench (same TSV format as gold `*-rel.tsv` files)
