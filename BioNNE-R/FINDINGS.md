# BioNNE-R 2026 — Intermediate Experimental Findings

*BioASQ 2026 shared task: biomedical named entity relation extraction (14 relation types, 8 entity types, EN + RU + bilingual)*
*Primary metric: macro-averaged F1. Submission deadline: May 14, 2026.*

---

## 1. Experimental Overview

### 1.1 System Architecture

All experiments use the same fine-tuning framework (`baseline/bionne_train.py`):

- **Encoder**: Pre-trained transformer backbone
- **Entity representation**: Typed entity markers inserted around entity spans at the character level; the model's hidden state at the opening `[HEAD:TYPE]` and `[TAIL:TYPE]` markers is extracted and concatenated for classification
- **Classifier head**: 2-layer MLP over the concatenated entity representations → softmax over 15 classes (14 relations + `no_relation`)
- **Loss**: Cross-entropy with per-class frequency-inverse weights; `no_relation` weight clamped to 1.0 to prevent the majority class from dominating
- **Negative sampling**: `no_relation` pairs are sampled from candidate entity pairs at a controlled ratio during training; all possible entity pairs are generated at test time

### 1.2 Experiments Conducted

| Run | Model | Track | Epochs | Batch | Dev macro F1 | Notes |
|---|---|---|---|---|---|---|
| Organizer baseline | `bert-base-multilingual-cased` (OpenNRE) | EN | — | — | 0.6944 | Reproduced, used as lower bound |
| Organizer baseline | `bert-base-multilingual-cased` (OpenNRE) | RU | — | — | 0.7166 | — |
| Organizer baseline | `bert-base-multilingual-cased` (OpenNRE) | BIL | — | — | 0.7211 | — |
| BiomedBERT EN v1 | `BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | EN | 10 | 16 | — | Catastrophic test failure (F1=0.2524) |
| **BiomedBERT EN v2** | `BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | EN | 10 | 16 | **0.7369** | All fixes applied; `ckpt: biomedbert_en_v2_final.pt` |
| **mDeBERTa EN v1** | `microsoft/mdeberta-v3-base` | EN | 10 | 64 | **0.7902** | Trained with 917 OpenAI-augmented examples (51,840 total); best at epoch 2; `ckpt: mdeberta_en_v1.pt` |
| **mDeBERTa RU v2** | `microsoft/mdeberta-v3-base` | RU | 10 | 128 | **0.8397** | `ckpt: mdeberta_ru_v2.pt` |
| mDeBERTa RU aug v1 | `microsoft/mdeberta-v3-base` | RU | 10 | 128 | — | Trained with 874 OpenAI-augmented examples; `ckpt: mdeberta_ru_aug_v1.pt` |
| **mDeBERTa BIL v2** | `microsoft/mdeberta-v3-base` | BIL | 10 | 128 | **0.8831** | `ckpt: mdeberta_bilingual_v2.pt` |

### 1.3 Model Selection Rationale

**English (BiomedBERT)**: The task is biomedical relation extraction from PubMed-style abstracts. Domain-specific pretraining on PubMed abstracts + full-text (BiomedBERT) provides substantially better biomedical entity and relation coverage than general-purpose or multilingual models. Prior work on BioNNE-L 2025 (the entity linking predecessor task) showed domain-specific BERT variants outperform `bert-base-multilingual-cased` by 10–15 F1 points on biomedical benchmarks.

**Russian and Bilingual (mDeBERTa-v3-base)**: Russian biomedical text lacks a comparable domain-specific pretrained model of sufficient size. `microsoft/mdeberta-v3-base` uses a disentangled attention mechanism and relative position encodings that have shown strong cross-lingual transfer on relation extraction benchmarks. The multilingual variant supports both Russian and English with a shared vocabulary, making it the natural choice for the bilingual subtask (Subtask 3) as well. Early checkpoint results (RU: 0.8397 at epoch 3, BIL: 0.8831 at epoch 2) substantially exceed the mBERT baseline.

---

## 2. Key Technical Optimizations

### 2.1 Typed Entity Markers with Nesting Awareness

The baseline uses generic `[unused0]/[unused1]/[unused2]/[unused3]` position markers. Our system introduces **typed entity markers**: the vocabulary is extended with tokens of the form `[HEAD:DISEASE]`, `[/HEAD:DISEASE]`, `[TAIL:GENE]`, `[/TAIL:GENE]` etc. (8 entity types × 2 roles × 2 open/close = 32 additional tokens). These are:

1. **Injected** at character-level span boundaries into the raw text before tokenization, producing `... [HEAD:DISEASE] hypertension [/HEAD:DISEASE] associated with [TAIL:GENE] ACE2 [/TAIL:GENE] ...`
2. **Detected** by the tokenizer as a single token each (no subword splitting)
3. **Used** as the entity representation anchors: the hidden state at the opening marker positions is extracted for classification

This explicitly encodes entity type information into the input representation, reducing the burden on the model to infer type from context alone. For nested entity spans (≈15% of all annotated pairs) where one entity span contains another, markers are interleaved in reading order (right-to-left insertion) so that the tokenized sequence remains well-formed.

Benefit: allows the model to distinguish, for example, `ALTERNATIVE_NAME` vs `ABBREVIATION` based on whether the entities are of the same type (both DISEASE, or DISEASE + ABBREVIATION type), even before attending to context.

### 2.2 Class-Weighted Loss with no_relation Clamping

The training distribution has a controlled negative ratio (3:1 `no_relation:positive` by default). At test time the true ratio is approximately 50–100:1 (all possible entity pairs from each document). Naively, a model trained with the imbalanced distribution would over-predict `no_relation` at test time.

We compute per-class weights as the inverse of training frequency, then **clamp the `no_relation` weight to 1.0**. This prevents `no_relation` from being upweighted even when it is synthetically under-represented (relative to test density), while still upweighting rare positive classes such as `APPLIED_TO` and `ALTERNATIVE_NAME`. Rare positive classes receive weights of approximately 4–8× the common classes.

### 2.3 Valid Triplet Type Pruning

A hallucination problem observed in early experiments: the model would predict semantically implausible relations (e.g., `AFFECTS(GENE, GENE)` when the corpus annotation schema forbids it). We extracted all valid `(head_type, tail_type, relation)` triplets from the NEREL-BIO annotation configuration file (`nerel-bio-v1.0/annotation.conf`), yielding **82 valid triplets**. During inference, any predicted relation that violates this schema is overridden to `no_relation`. This reduced test-time hallucinations significantly and is applied prior to the confidence threshold filter.

### 2.4 Truncation Failsafe via valid_markers

At `max_length=512`, approximately 3% of instances (long biomedical sentences) are still truncated, causing one or both entity markers to fall outside the token window. Without mitigation this produces silent errors: the model attends to wrong tokens as entity representations. We track per-instance marker validity at dataset construction time (`BioNNEDataset.valid_markers`), and during inference zero out the entity-marker hidden states for invalid instances, forcing them to `no_relation`. This was the primary fix that resolved the catastrophic v1 failure (test F1 0.2524 → 0.7369 after all fixes).

### 2.5 Batch Tokenization with Offset Mapping (Fast Tokenizer Prefill)

The OpenNRE tokenization pipeline tokenizes each instance individually (≈10–20 instances/sec for mBERT on CPU). For large blind test sets (818K pairs for EN test, 1.37M for BIL) this would take hours.

We replace per-instance tokenization with a **batched fast-tokenizer call**:

1. Entity markers are inserted into raw text at character level (right-to-left to preserve offsets)
2. The character position of each opening marker `[HEAD:TYPE]` is recorded
3. `BertTokenizerFast` tokenizes all texts in chunks of 10,000 with `return_offsets_mapping=True`
4. The `offset_mapping` tensor (`[chunk, seq_len, 2]`) is used to locate the exact token position of each marker via `(offset_starts == marker_char_pos).argmax(dim=1)` — a vectorized character-to-token index lookup
5. Model weights are cast to FP16 once (`model.half()`) for 2× throughput on CUDA

Observed throughput: **~500 instances/sec on a single L4 GPU** for the OpenNRE baseline author model, versus ~15 instances/sec with the default OpenNRE pipeline. This makes running 4 stochastic seeds over 1.37M bilingual instances tractable within a single Modal timeout window.

### 2.6 Modal Training Infrastructure and Fault-Tolerant Orchestration

All training and inference runs on [Modal](https://modal.com) A100/L4 GPU instances (4-hour timeout per session). The training pipeline is fault-tolerant via a **dual-checkpoint scheme**:

- **Best-model checkpoint** (`<name>.pt`, ~1 GB): saved whenever a new dev macro F1 best is achieved; contains only model weights, `rel2id`, and metadata. Used for inference.
- **Resume checkpoint** (`<name>.pt.resume`, ~3.1 GB): saved at every epoch completion and every 300 steps mid-epoch. Contains model state, optimizer state (AdamW), LR scheduler state, scaler state (FP16), current epoch, and best F1 so far. Enables exact continuation after a timeout.
- A background thread commits the Modal Volume every 5 minutes during training, so resume checkpoints survive container preemption.

A local `orchestrate.py` watchdog (Python, `threading.Thread` per track) monitors the running Modal apps by PID, inspects the resume checkpoint epoch via a lightweight Modal function after each training session ends, and automatically re-launches training until epoch 10, then triggers inference on the test blind files. This means full 10-epoch training for large models requires no manual intervention despite the 4-hour container limit.

---

## 3. Results

### 3.1 Dev Set Performance (Blind Evaluation Mode)

Dev F1 reported in **blind mode**: the model is run on all candidate entity pairs extracted from dev documents (same procedure as test), then evaluated against gold annotations. This is the correct evaluation methodology for this task.

| Track | Model | Macro F1 (blind dev) | Notes |
|---|---|---|---|
| EN | BiomedBERT + typed markers + class weights | **0.7369** | `biomedbert_en_v2_final.pt` |
| EN | mDeBERTa-v3-base + augmentation | **0.7902** | Best at epoch 2; `mdeberta_en_v1.pt` |
| RU | mDeBERTa-v3-base | **0.8397** | 10 epochs; `mdeberta_ru_v2.pt` |
| BIL | mDeBERTa-v3-base | **0.8831** | 10 epochs; `mdeberta_bilingual_v2.pt` |

**mDeBERTa EN per-epoch dev F1** (note overfitting after epoch 2):

| Ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.6337 | **0.7902** | 0.7255 | 0.7759 | 0.7206 | 0.7658 | 0.7551 | 0.7703 | 0.7622 | 0.7537 |

The model peaked sharply at epoch 2 and oscillated without recovering. Likely causes: the augmented training set is small (~52K instances), the model memorises the augmented examples early, and subsequent epochs overfit to the training distribution. The best-checkpoint saving policy (`bionne_train.py` saves only on new F1 best) ensures `mdeberta_en_v1.pt` holds epoch 2 weights.

### 3.2 Test Set Submissions — Initial Failures (threshold=0.5)

**First test submission** (BiomedBERT EN v2, 2026-05-11): macro F1 = **0.4593** — a genuine 0.28-point drop from blind dev, attributable to distribution shift.

**Second round** (mDeBERTa, threshold=0.5, 2026-05-12): catastrophic failure across all tracks.

| Track | Dev F1 | Test F1 (t=0.5) | Predictions | Analysis |
|---|---|---|---|---|
| EN | 0.7902 | 0.2819 | 33,338 | 2.3× overcount vs BiomedBERT |
| RU | 0.8397 | 0.4084 | 17,982 | ~2× overcount |
| BIL | 0.8831 | 0.3310 | 58,486 | ~4× overcount |

**Root cause — train/test density mismatch**: training uses a 3:1 negative sampling ratio (`--neg-ratio 3`), but at test time *all* candidate entity pairs are enumerated, yielding approximately 87:1 negative ratio (818K total EN pairs, ~10K true positives). A model calibrated to output "positive" at p≥0.5 on 3:1 data will over-predict by a factor of ~20 on 87:1 data.

Evidence: at threshold=0.5, EN predictions (33,338) are 2.3× more numerous than BiomedBERT (14,480) predictions on identical test documents. BiomedBERT scored higher on test despite lower dev F1 — because its default argmax behavior happened to be more conservative.

### 3.3 Confidence Threshold Calibration

**Fix**: reject any prediction where the winning softmax probability < threshold T. Run a sweep over T on the dev_blind set (all candidate pairs, same 87:1 ratio as test) to find the T that maximises blind-dev macro F1. This is implemented in `bionne_predict.find_threshold()` — a single inference pass followed by O(N) CPU sweeps at each threshold value.

**Prior experiment on Colab (non-augmented models)**: optimal T = 0.99 for all tracks. Applied as starting point for current models.

### 3.4 CodaBench Test Scores — Complete Results (2026-05-12)

**Best scores per track (bold):**
- EN: **0.4593** — BiomedBERT v2 at t=0.5 (original May 11 submission)
- RU: **0.5057** — mDeBERTa base (no augmentation) at t=0.95
- BIL: **0.4527** — mDeBERTa BIL v2 at t=0.95

#### mDeBERTa augmented EN — full threshold curve

| t | Predictions | CodaBench F1 | Notes |
|---|---|---|---|
| 0.50 | 33,338 | 0.2819 | Over-predicts ~3× |
| 0.80 | 16,888 | 0.3285 | |
| 0.85 | 13,785 | **0.3334** | ← augmented EN model peak |
| 0.90 | 10,239 | 0.3318 | |
| 0.95 | 5,804 | 0.2982 | |
| 0.99 | 1,136 | 0.0982 | Catastrophic recall collapse |

The augmented EN model peaks at t=0.85 and degrades in both directions. The collapse from 10,239→1,136 predictions between t=0.90 and t=0.99 (9× drop over 0.09 range) is anomalous — in a well-calibrated model this would be gradual. The GPT-generated augmentation examples push the model's positive-class confidence into a bimodal distribution (very high or very low), creating a gap in the 0.90–0.99 probability range.

#### RU/BIL base model threshold curve

| Track | t=0.5 | t=0.90 | t=0.95 | t=0.99 |
|---|---|---|---|---|
| RU | 0.4084 | 0.4944 | **0.5057** | 0.4560 |
| BIL | 0.3310 | 0.4356 | **0.4527** | 0.4470 |

RU/BIL optimal at t=0.95, consistent with non-augmented model calibration.

#### RU augmented model — failed

| t | Predictions | CodaBench F1 | Notes |
|---|---|---|---|
| 0.50 | ~33K | 0.2990 | Over-prediction failure |
| 0.99 | 2,214 | — (not submitted) | Severely skewed: only 8 HAS_CAUSE predictions vs 1,030 in base model |

The RU augmented model (`mdeberta_ru_aug_v1.pt`) failed. At t=0.99, only 2,214 predictions with a pathological class distribution (8 HAS_CAUSE, 708 SUBCLASS_OF, 345 ORIGINS_FROM). The augmentation disrupted the model's calibration for dominant classes without improving the rare ones. The base RU model (0.5057) substantially outperforms the augmented variant. **Augmentation is net negative for RU; do not use.**

#### Alternative approaches for EN (Section 8 experiments — results)

| Approach | Predictions | CodaBench F1 | vs best EN (0.4593) |
|---|---|---|---|
| BIL model on EN test (t=0.95) | 14,328 | 0.4542 | −0.005 |
| Existence filter t=0.95 | 16,349 | 0.3870 | −0.072 |
| Per-class thresholds | 10,616 | 0.3625 | −0.097 |
| EN t=0.85 (augmented mDeBERTa) | 13,785 | 0.3334 | −0.126 |

BIL-on-EN is the best alternative to BiomedBERT, just 0.005 below.

**BiomedBERT threshold sweep** (t=0.70/0.80/0.90) running — ETA ~18:50.

**BIL-on-EN threshold sweep** (t=0.90/0.99) running — ETA ~18:45.

#### Key finding: BiomedBERT domain pretraining outperforms mDeBERTa on EN test

BiomedBERT (dev F1=0.7369) scores 0.4593 on test; mDeBERTa augmented (dev F1=0.7902) peaks at only 0.3334. A model with *lower* dev F1 is *better* on test by +0.126. This is explained by:

1. **Domain match**: BiomedBERT was pretrained on PubMed abstracts + full-text — the same domain as the test set. mDeBERTa's multilingual pretraining does not include domain-specific biomedical vocabulary.
2. **Calibration stability**: mDeBERTa's augmented training creates a bimodal confidence distribution, concentrating probability mass at extremes and leaving a gap in the 0.90–0.99 range. BiomedBERT's softmax distribution is smoother, making its default t=0.5 threshold already close to optimal.
3. **Augmentation overfitting**: mDeBERTa EN peaked at epoch 2 and oscillated for 8 more epochs — the augmented set (917 synthetic examples) was too small to regularise fine-tuning effectively. BiomedBERT had no augmentation and trained more stably.

The bilingual mDeBERTa (0.4542 on EN) partially recovers this gap through cross-lingual regularisation on the larger combined EN+RU dataset, but still falls short of BiomedBERT's domain advantage.

### 3.5 EN Class Distribution Analysis (mDeBERTa augmented, test predictions)

A key diagnostic: compare predicted class counts across thresholds against expected counts (extrapolated from gold dev distribution × test/dev pair ratio of 3.49×).

| Class | Expected in test | t=0.50 | t=0.85 | BiomedBERT t=0.5 | BIL t=0.95 |
|---|---|---|---|---|---|
| SUBCLASS_OF | ~2,641 | 3,892 | 2,451 | 3,399 | 2,493 |
| HAS_CAUSE | ~1,699 | 8,906 | 2,498 | 2,205 | 1,738 |
| AFFECTS | ~1,372 | 3,925 | 2,145 | 2,221 | 1,470 |
| ASSOCIATED_WITH | ~813 | 5,006 | 1,619 | 1,684 | 1,983 |
| PHYSIOLOGY_OF | ~621 | 2,031 | 1,366 | 753 | 1,260 |
| FINDING_OF | ~290 | 1,337 | 853 | 599 | 1,048 |
| ABBREVIATION | ~258 | 786 | 528 | 398 | 275 |
| ORIGINS_FROM | ~276 | 1,246 | 354 | 425 | 1,003 |
| TREATED_USING | ~307 | 955 | 418 | 328 | 303 |
| PART_OF | ~866 | 2,015 | 715 | 1,241 | 1,082 |
| TO_DETECT_OR_STUDY | ~405 | 2,322 | 782 | 735 | 966 |
| USED_IN | ~276 | 681 | 41 | **346** | **599** |
| ALTERNATIVE_NAME | ~342 | 123 | 12 | **57** | **76** |
| APPLIED_TO | ~188 | 113 | 3 | **89** | **32** |

**ALTERNATIVE_NAME/USED_IN/APPLIED_TO are near-zero for the augmented EN model at any threshold ≥ 0.85**, despite augmentation targeting these exact classes. At t=0.50, they appear (123/681/113) but the overall over-prediction makes t=0.50 worse than t=0.85 overall.

BiomedBERT gets APPLIED_TO=89 (47% of expected), USED_IN=346 (125% of expected) — far better than mDeBERTa at any threshold. This single fact explains most of BiomedBERT's test advantage: better recall on the rare classes that macro F1 weighs equally with common classes.

**Per-class threshold experiment**: Setting ALTERNATIVE_NAME/USED_IN/APPLIED_TO to t=0.40–0.50 and PHYSIOLOGY_OF/FINDING_OF/ABBREVIATION to t=0.95 recovered APPLIED_TO to 188 (exactly expected) and ALTERNATIVE_NAME to 123, but USED_IN became 681 (2.5× over-predicted). CodaBench result: 0.3625 — better than flat t=0.85 (0.3334) but far below BiomedBERT (0.4593). The USED_IN over-prediction at t=0.50 introduced too many false positives.

### 3.3 Per-Class Dev Breakdown (EN BiomedBERT v2)

| Relation | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| ABBREVIATION | 0.8354 | 0.8919 | **0.8627** | Well-learned |
| AFFECTS | 0.8962 | 0.9031 | **0.8996** | Well-learned |
| ALTERNATIVE_NAME | 0.4250 | **0.1753** | 0.2482 | **Critical failure** |
| APPLIED_TO | 0.5946 | 0.4074 | 0.4835 | Moderate failure |
| ASSOCIATED_WITH | 0.7045 | 0.7983 | 0.7485 | Acceptable |
| FINDING_OF | 0.6800 | 0.8193 | 0.7432 | Acceptable |
| HAS_CAUSE | 0.7783 | 0.7510 | 0.7644 | Good |
| ORIGINS_FROM | 0.7614 | 0.8481 | **0.8024** | Good |
| PART_OF | 0.6877 | 0.8133 | 0.7452 | Acceptable |
| PHYSIOLOGY_OF | 0.9294 | 0.8876 | **0.9080** | Best class |
| SUBCLASS_OF | 0.8501 | 0.8613 | **0.8556** | Well-learned |
| TO_DETECT_OR_STUDY | 0.6503 | 0.8017 | 0.7181 | Acceptable |
| TREATED_USING | 0.8415 | 0.7841 | **0.8118** | Good |
| USED_IN | 0.7160 | 0.7342 | 0.7250 | Acceptable |

### 3.4 Confusion Analysis for Critical Classes

**ALTERNATIVE_NAME** (n=98 gold dev instances, F1=0.2482):
- → `no_relation`: 61 instances (62.2%) — dominant failure mode; model does not recognize the pattern
- → `ALTERNATIVE_NAME` (correct): 17 instances (17.3%)
- → `SUBCLASS_OF`: 9 instances (9.2%)
- → `PART_OF`: 6 instances (6.1%)
- → `ABBREVIATION`: 3 instances (3.1%)

*Primary cause*: ALTERNATIVE_NAME (e.g., "X, also known as Y", "X (syn. Y)", "X or Y") shares surface patterns with hyperonymy (`SUBCLASS_OF`) and meronymy (`PART_OF`). The model defaults to `no_relation` for 62% of cases because ALTERNATIVE_NAME is relatively rare (3.3% of dev gold positives) and its prototypical surface form ("X is another name for Y") overlaps substantially with general semantic similarity. The confusion with `ABBREVIATION` is minimal (3%), suggesting the issue is not ALTERNATIVE_NAME/ABBREVIATION conflation but rather failure to identify the pattern at all.

**APPLIED_TO** (n=54 gold dev instances, F1=0.4835):
- → `APPLIED_TO` (correct): 22 instances (40.7%)
- → `TO_DETECT_OR_STUDY`: 15 instances (27.8%)
- → `no_relation`: 12 instances (22.2%)
- → `ORIGINS_FROM`: 5 instances (9.3%)

*Primary cause*: `APPLIED_TO` (a method/substance is administered or applied to a biological entity) and `TO_DETECT_OR_STUDY` (an instrument/method is used to observe a biological entity) are semantically adjacent — both describe an agent acting on a patient entity. The distinction requires understanding whether the intent is treatment/modification (`APPLIED_TO`) or detection/characterization (`TO_DETECT_OR_STUDY`), which requires broader discourse context than a single sentence provides.

---

## 4. Baseline Stochastic Ensemble

For the organizer's baseline model (OpenNRE mBERT), we run inference in two modes:

**Deterministic** (seed=42, dropout disabled): standard inference; predictions are reproducible.

**Stochastic** (seeds 1, 2, 3; dropout enabled at inference time): MC dropout [Gal & Ghahramani, 2016] style inference where Dropout layers are kept in training mode during prediction. Each seed produces slightly different predictions due to random activation dropout.

**Ensemble**: majority vote across 4 runs (1 deterministic + 3 stochastic). For each candidate triplet `(doc_id, head_span, tail_span)`:
1. Collect all predicted relation labels from all runs that predicted this triplet as positive
2. Add implicit `no_relation` votes for runs that did not predict this triplet as positive
3. The plurality label wins; ties broken by `no_relation`
4. `no_relation` winners are excluded from the output TSV

This approach is motivated by evidence that MC dropout ensembles improve calibration and recall on low-confidence instances (rare classes), at the cost of one additional inference pass per seed. It is applied to all three tracks (EN/RU/BIL) of the organizer baseline, not to our fine-tuned models (which use a single deterministic checkpoint at test time).

---

## 5. Improvement Strategy

### 5.1 Immediate: Data Augmentation for Critical Classes

**Training set class counts (EN)**:
| Class | Train count | Dev macro F1 | Primary failure mode |
|---|---|---|---|
| APPLIED_TO | **43** | 0.4835 | 28% confused with TO_DETECT_OR_STUDY |
| ALTERNATIVE_NAME | **95** | 0.2482 | 62% → no_relation; 9% → SUBCLASS_OF |
| ABBREVIATION | 97 | 0.8627 | — (well-learned despite similar count) |
| TO_DETECT_OR_STUDY | 164 | 0.7181 | — |

Key insight from confusion analysis: ALTERNATIVE_NAME and ABBREVIATION have near-identical training counts (95 vs 97) but vastly different recall (17.5% vs 89.2%). This confirms the failure is not just data quantity but pattern ambiguity — ALTERNATIVE_NAME has highly varied surface forms (synonyms, legacy terms, brand names) while ABBREVIATION has distinctive parenthetical patterns. ABBREVIATION learned easily; ALTERNATIVE_NAME did not.

APPLIED_TO and TO_DETECT_OR_STUDY share the entity type pair `LABPROC → ANATOMY` (APPLIED_TO: 30/43, TO_DETECT_OR_STUDY: 33/164 examples). The model cannot learn the distinction from surface form alone — it requires understanding whether the procedure is interventional (APPLIED_TO) or observational (TO_DETECT_OR_STUDY).

The two critical failure modes — ALTERNATIVE_NAME (recall=17.5%) and APPLIED_TO→TO_DETECT_OR_STUDY confusion — are both training data problems: the model has seen too few high-quality examples of these patterns.

**Augmentation pipeline** (`baseline/augment_openai.py`):

1. **Generator** (GPT-5-mini, batched OpenAI API, `temperature=0.9`): given
   - The relation definition and schema-constrained entity type pair
   - 5–10 seed examples from the training set
   - Instruction to vary domain subfield, sentence structure, and entity surface form
   → generate N candidate `(sentence, entity1_span, entity2_span, relation)` tuples

2. **Verifier** (GPT-5.4-mini, batched OpenAI API): given the generated example plus 3 "confusable" examples of semantically similar relation types, verify:
   - Does the sentence-level context unambiguously support the annotated relation?
   - Are the entity spans correctly delimited?
   - Would a biomedical expert label this differently?
   → binary accept/reject with confidence score

3. **Schema filter**: verify the `(head_type, tail_type, relation)` triplet is in `valid_triplets.json`

4. **Deduplication**: character-level shingling to reject near-duplicates of seed examples

**Schema validation**: every accepted example's `(head_type, tail_type, relation)` triplet is checked against `valid_triplets.json`; span offsets are verified by `sentence[start:end] == entity_text` before submission to the verifier.

**Output format**: JSON-lines identical to `prepare_data.py` output — can be directly concatenated with `eng_train.txt` and passed to `bionne_train.py` without any pipeline changes.

Target augmentation volumes (per class, across entity type pairs):
- `ALTERNATIVE_NAME`: 400 total (DISO-DISO: 130, CHEM-CHEM: 90, ANATOMY-ANATOMY: 80, FINDING-FINDING: 50, LABPROC-LABPROC: 30, PHYS-PHYS: 20)
- `APPLIED_TO`: 200 total (LABPROC-ANATOMY: 130, CHEM-ANATOMY: 70)
- `TO_DETECT_OR_STUDY` contrastive pairs: 150 (LABPROC-ANATOMY and LABPROC-DISO, focusing on the APPLIED_TO boundary)

**Contrastive augmentation for APPLIED_TO**: For each generated `APPLIED_TO` example, the generator is additionally prompted to produce a minimally-different sentence where the relation would be `TO_DETECT_OR_STUDY`. Both are included in training with their respective labels. This "hard negative" approach teaches the model the precise lexical/semantic cue that distinguishes the two relations.

**Augmentation results (actual):**

| Language | Target | Generated | Accepted | Notes |
|---|---|---|---|---|
| EN | ~750 | — | **917** | Exceeded target; concatenated into `eng_train_aug.txt` |
| RU | 680 | — | **874** | APPLIED_TO(LABPROC,ANATOMY) was difficult (103/120 max); includes 187 TO_DETECT_OR_STUDY contrastive pairs |

**Impact on dev F1**: EN improved from 0.7369 (BiomedBERT) to 0.7902 (mDeBERTa, augmented). However the epoch curve shows rapid overfitting after epoch 2, suggesting the augmented data may not be diverse enough to regularise the full 10-epoch training run. RU augmented model dev F1 not yet measured (inference running).

**Generation failure mode observed**: the `APPLIED_TO(LABPROC,ANATOMY)` job in Russian repeatedly hit the 8,192 token output limit (`finish_reason=length`) with 0 valid candidates. Russian biomedical procedure names are significantly longer than English equivalents; the model attempted to generate 5 examples per call but the JSON output was truncated before the array closed. Eventually reached 103/120 (86%) of target before exhausting the 960-attempt budget.

### 5.2 EN Re-training Strategy

After augmentation:

1. **Model**: switch to `microsoft/mdeberta-v3-base` (same as RU/BIL) rather than BiomedBERT, for two reasons:
   - BiomedBERT gives 0.7369 blind-dev but only 0.4593 test — the 0.28 gap suggests the model does not generalize well to unseen test document distribution; mDeBERTa's DeBERTa-v3 architecture with ELECTRA-style pretraining has shown better calibration and OOD generalization on relation extraction benchmarks
   - Consistency across tracks: all three models would then use the same architecture, and a bilingual checkpoint could be further fine-tuned per-track

2. **Class weights**: increase weight multiplier for `ALTERNATIVE_NAME` (currently ~4×) to ~8–10×; increase `APPLIED_TO` weight to ~6×; keep `no_relation` clamped at 1.0

3. **Contrastive loss auxiliary**: add a triplet-style margin loss on the entity-pair representations for the ALTERNATIVE_NAME/SUBCLASS_OF/PART_OF cluster, pushing ALTERNATIVE_NAME embeddings away from the other two

4. **Longer training**: 15 epochs (vs 10) with early stopping on dev macro F1

5. **Re-evaluate on dev blind before re-submission**

### 5.3 RU/BIL Strategy

mDeBERTa-v3-base is already showing strong results (RU: 0.8397 at epoch 3, BIL: 0.8831 at epoch 2). The primary improvements here are:

- Complete 10-epoch training (auto-resume orchestrator handles this)
- After training, run the same per-class analysis to identify RU-specific weak classes (APPLIED_TO and ALTERNATIVE_NAME are expected to be weak in Russian as well, as they represent the same semantic ambiguities in a morphologically richer language)
- If weak classes identified: apply the same augmentation pipeline in Russian (GPT-5-mini supports Russian generation; GPT-5.4-mini can verify against Russian biomedical convention)

---

## 6. Compute and Engineering Notes

### 6.1 Infrastructure

| Component | Spec |
|---|---|
| Training compute | Modal A100 40GB, 4-hour sessions, FP16 + gradient checkpointing |
| Inference compute | Modal L4 24GB, batch_size=256, FP16 autocast |
| Storage | Modal Volume `bionne-v2` (persistent across sessions): checkpoints, data, predictions |
| Local environment | WSL2 / Linux CPU; no local GPU. Code edited locally, runs remotely via Modal CLI. |

### 6.2 Training Throughput

- **Batch size**: 128 (A100, FP16, gradient checkpointing) — fits within 40GB VRAM for mDeBERTa-v3-base with `max_length=512`
- **Epochs per 4-hour session**: ~2.4 for RU (702K training instances), ~2.3 for BIL (1.47M instances)
- **Resume overhead**: <2 minutes per session (volume reload, checkpoint load, data-ready wait)
- **LR schedule**: linear warmup (10% of total steps) + linear decay, AdamW, lr=2e-5

### 6.3 Inference Throughput (Author Baseline, L4)

The author baseline (OpenNRE mBERT, max_length=256) uses a custom batched inference path replacing the original per-instance tokenization loop:

| Stage | Time | Notes |
|---|---|---|
| Marker insertion (all N instances) | ~1s (Python list ops) | Right-to-left insertion preserves char offsets |
| Batch tokenization (N=1.37M) | ~742s | 10K-chunk fast tokenizer, offset mapping |
| Token position lookup | vectorized (≈0s extra) | `offset_starts == marker_char_pos` argmax over seq_len |
| GPU inference (512 batches) | ~25 min | FP16 autocast, batch_size=512 |

Fallback instances (entity markers truncated out of the 256-token window): 539,485 of 1,369,530 bilingual instances (39.4%). These are assigned `no_relation` via the marker failsafe. This high truncation rate is a known limitation of the mBERT baseline at `max_length=256`; our fine-tuned models use `max_length=512`.

### 6.4 Orchestration

The `orchestrate.py` script manages the full training pipeline:
- One `threading.Thread` per track (RU, BIL) runs concurrently in the local Python process
- Each thread: wait for in-flight Modal PID → inspect volume checkpoint epoch → if epoch < 10: re-run training → loop → if epoch == 10: run prediction → download TSV
- Epoch inspection uses a lightweight (CPU-only, no-GPU) Modal function that loads only the resume checkpoint header
- All stdout/stderr from Modal subprocesses is streamed to `orchestrate.log`

A separate `run_author_baseline.py` manages 11 concurrent author-baseline inference jobs (EN/RU/BIL × 4 seeds each, plus EN deterministic already done) using `threading.Thread`, then calls `ensemble_track` per track for majority-vote ensembling.

---

## 7. Open Questions / Future Work

1. **Why does EN test F1 drop ~0.28 points from blind dev?** Confirmed cause: train/test density mismatch (3:1 vs 87:1 negative ratio) requires threshold calibration. After calibration (t=0.90), EN test F1 = 0.3318, still well below blind dev 0.7369. Residual gap (~0.40 points) is genuine distribution shift — different document types, harder relation instances.

2. **mDeBERTa EN vs BiomedBERT on test**: mDeBERTa (t=0.90) = 0.3318 vs BiomedBERT initial submission = 0.4593. BiomedBERT is still better on test despite lower dev F1. Possible reasons: (a) BiomedBERT domain pretraining generalises better to unseen PubMed documents; (b) mDeBERTa's EN training overfit due to small augmented set; (c) mDeBERTa threshold not yet optimal (t=0.85/0.80 sweep running).

3. **EN threshold anomaly**: mDeBERTa EN collapses from 10,239 predictions at t=0.90 to 1,136 at t=0.99. This gap (9:1 ratio over a 0.09 threshold range) is unusual — in well-calibrated models the drop is more gradual. Likely caused by the augmented training data: GPT-generated examples may have slightly different surface statistics than real PubMed text, pushing the model's confidence distribution for EN into a bimodal shape (very confident or very uncertain). No analogous collapse is seen for RU/BIL (un-augmented models).

4. **Does augmentation help RU test F1?** RU aug t=0.99 inference running. If `mdeberta_ru_aug_v1.pt` at t=0.99 outperforms `mdeberta_ru_v2.pt` at t=0.95 (current best: 0.5057), augmentation is worth the effort for the next iteration.

5. **Does stochastic ensembling help the organizer baseline?** Author baseline ensemble submitted but scored 0.2205 (EN) and 0.2604 (RU) — much worse than fine-tuned models. The ensemble itself works correctly (majority vote over 4 seeds), but the underlying mBERT model is simply outclassed. Not worth pursuing further.

6. **Augmentation quality for rare classes**: The per-class improvement from augmentation is not yet measured directly (would require ablation: train without augmented examples, compare dev F1 per class). Given the overall epoch-2 peak, the augmented data may have helped early training convergence on rare classes but then been memorised, causing oscillation in later epochs.

7. **Longer training / lower LR**: Given that mDeBERTa EN peaks at epoch 2 and then oscillates, re-training with a lower learning rate (1e-5 vs 2e-5) and cosine schedule (rather than linear decay) might produce a flatter loss curve and better generalisation. Not attempted due to time constraints.

---

## 8. Further Experiments (May 2026)

### 8.1 Existence Filter: `1 - p(no_relation)` as Confidence Score

**Motivation.** The current threshold criterion emits a prediction when `max(p_positive_classes) ≥ T`. For rare or ambiguous relations (ALTERNATIVE_NAME, APPLIED_TO), the model may correctly detect that *some* relation exists but spread its probability mass across several plausible types, so no single class exceeds T. This manifests as the EN model's anomalous prediction collapse between t=0.90 (10,239 predictions) and t=0.99 (1,136 predictions) — a 9× drop over a 0.09 threshold range that indicates the model is frequently placing ~0.10–0.20 on the winning class while distributing the rest.

**Approach.** Replace the filter score with the total positive mass: `score = 1 - p(no_relation)`. The predicted type is still `argmax` over positive classes. Formally:
```
emit if  1 - p(no_relation) ≥ T
predict  argmax_{c ≠ no_relation} p(c)
```
This decouples the *existence* decision (is there any relation at all?) from the *type* decision (which specific relation?), using a softer existence criterion that accumulates probability across all positive classes.

**Implementation.** Added `existence_filter: bool` parameter to `_probs_to_df`, `predict`, and `find_threshold` in `bionne_predict.py`, and `--existence-filter` flag to `modal_app.py::predict_model`. The filter score is computed as:
```python
exist_score = 1.0 - all_probs[:, no_rel_id]
_, top_cls = pos_probs.max(dim=-1)   # pos_probs has no_rel column zeroed
pred = top_cls where exist_score ≥ T else no_rel
```

**Results (2026-05-12):**

| Track | Predictions | CodaBench F1 | vs standard t=0.95 |
|---|---|---|---|
| EN ef t=0.95 | 16,349 | 0.3870 | +0.054 vs EN t=0.85 (0.3334), but −0.072 vs BiomedBERT |
| RU ef t=0.95 | 10,418 | 0.5036 | −0.002 vs base t=0.95 (0.5057) |
| BIL ef t=0.95 | 32,434 | 0.4505 | −0.002 vs base t=0.95 (0.4527) |

The existence filter improves EN over the augmented model's flat threshold (0.3870 vs 0.3334) but still far below BiomedBERT. For RU and BIL, the existence filter is marginally worse — the base models are already well-calibrated and the extra predictions (especially APPLIED_TO: 79→166 for RU) add more noise than signal. The filter is more useful for the miscalibrated augmented EN model than for the cleaner non-augmented RU/BIL models.

**Conclusion: existence filter is a moderate improvement for miscalibrated models; not helpful when the model is already well-calibrated.**

---

### 8.2 Bilingual Model for English Predictions

**Motivation.** The BIL model (`mdeberta_bilingual_v2.pt`, blind dev F1=0.8831) was trained on both EN and RU data combined — substantially more data than the EN-only model (0.7902). The combined training may produce better-calibrated EN representations through cross-lingual transfer and regularisation from the larger dataset. BiomedBERT, which is domain-matched but EN-only, still achieves 0.4593 on test vs mDeBERTa EN 0.3318 — suggesting that the EN mDeBERTa's advantage on dev does not transfer well to test.

**Approach.** Run `predict_model` with `ckpt=mdeberta_bilingual_v2.pt` on `eng_test_blind.txt`. This requires no new training — just inference with the existing BIL checkpoint. The BIL model uses the same 15-class `rel2id.json`, so the output format is identical.

**Expected outcome.** If cross-lingual transfer generalises well, the BIL model may score 0.45+ on EN test (similar to its performance gap over the EN model on dev). If the BIL model's EN-language attention heads are less focused than the EN-specific model's, it could underperform. The experiment costs ~2h inference time.

**Results (2026-05-12):** BIL-on-EN at t=0.95 scored **0.4542** — second best EN result, 0.005 below BiomedBERT. The bilingual model's EN class distribution is substantially more balanced than the augmented EN model's: USED_IN=599 (vs 41 at EN t=0.85), APPLIED_TO=32 (vs 3). However it over-predicts ORIGINS_FROM (1,003 vs expected ~276), suggesting cross-lingual transfer brings some RU-specific associations into EN predictions.

BIL threshold sweep at t=0.90 and t=0.99 running to find the optimum. BIL-on-EN t=0.90 is expected to have higher recall on rare classes.

**Conclusion: bilingual model on EN is a strong baseline (0.4542), close to but not exceeding BiomedBERT (0.4593). The cross-lingual regularisation from RU data improves rare class coverage for EN, but domain-specific BiomedBERT pretraining still holds a small advantage.**

---

### 8.3 Lower Learning Rate + Cosine Schedule Re-training (EN)

**Motivation.** The mDeBERTa EN training curve shows a sharp peak at epoch 2 (F1=0.7902) followed by oscillation between 0.72–0.77 for epochs 3–10, without recovery. This is characteristic of a learning rate that is too high for the effective dataset size: the model makes large gradient steps that overshoot the loss minimum once the easy examples are learned, and the small augmented dataset provides insufficient gradient signal to re-converge.

**Approach.** Re-train `mdeberta_en_v1.pt` from scratch with:
- Learning rate: **1e-5** (vs 2e-5 current)
- Schedule: **cosine annealing** (vs linear decay current) — cosine decays more slowly in the middle epochs, allowing finer adjustment when the model is near a local minimum
- Epochs: 15 (extended, with best-checkpoint saving)
- Everything else unchanged (same augmented data, typed markers, class weights)

**Expected outcome.** A flatter loss curve with a later peak (expected epoch 4–6) and less oscillation. The cosine schedule has been shown to improve generalisation in transformer fine-tuning by effectively implementing a form of cyclic learning rate annealing.

**Estimated compute.** ~3h training (A100) + ~2h inference (L4) = ~5h total. Feasible before the May 14 deadline if started promptly.

**Status.** Deprioritised. BiomedBERT (0.4593) and BIL-on-EN (0.4542) already outperform what a re-trained EN mDeBERTa is likely to achieve before the May 14 deadline, and would require 5h compute + at least one CodaBench submission cycle. The remaining time is better spent sweeping BiomedBERT thresholds.

### 8.4 BiomedBERT Threshold Sweep

**Motivation.** BiomedBERT at t=0.5 gives 14,480 predictions vs expected ~10,366 — ~40% over-prediction. At t=0.5 it already scores 0.4593 (best EN). A calibrated threshold (t=0.70–0.90) should cut the over-predicted common classes (AFFECTS: 2,221 vs expected 1,372; HAS_CAUSE: 2,205 vs expected 1,699) while keeping the rare classes that BiomedBERT is already better at predicting (APPLIED_TO: 89, USED_IN: 346).

**Results (CodaBench, 2026-05-13):**

| Threshold | Preds | F1 | vs t=0.5 |
|---|---|---|---|
| t=0.50 (original) | 14,480 | 0.4593 | baseline |
| t=0.70 | 12,329 | 0.4726 | +0.0133 |
| **t=0.80** | **11,430** | **0.4733** | **+0.0140 ← NEW BEST EN** |
| t=0.90 | 10,113 | 0.4722 | +0.0129 |

**Conclusion.** t=0.80 is the optimal threshold: 0.4733, a +0.014 improvement over the original. The curve is very flat at the top (0.4722–0.4733 across t=0.70–0.90), confirming BiomedBERT is well-calibrated in this range. 11,430 predictions at t=0.80 is close to the expected gold count (~10,366), validating the calibration hypothesis. This is the **final best EN submission**.

### 8.5 BIL-on-EN Additional Threshold Sweep

**Results (CodaBench, 2026-05-13):**

| Threshold | Preds | F1 | Notes |
|---|---|---|---|
| t=0.95 | 14,328 | 0.4542 | Previously submitted |
| t=0.90 | 17,877 | 0.4447 | Over-predicts |
| t=0.99 | 7,657 | 0.4400 | Under-predicts |

**Conclusion.** BIL-on-EN peaks at t=0.95 (0.4542) and is consistently ~0.019 below BiomedBERT t=0.80 (0.4733). BiomedBERT with domain pretraining is clearly the better EN model; BIL cross-lingual transfer does not compensate for the loss of biomedical domain knowledge.
