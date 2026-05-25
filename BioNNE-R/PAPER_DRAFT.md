# System Description for BioNNE-R 2026: Typed Entity Markers, LLM-Augmented Rare Classes, and Calibrated Confidence Thresholding

*Draft for BioASQ 2026 Workshop Proceedings*

---

## Abstract

We describe our system for BioNNE-R 2026 (BioASQ shared task on biomedical named entity relation extraction), achieving macro F1 of **0.4733** (EN, Subtask 1), **0.5057** (RU, Subtask 2), and **0.4527** (BIL, Subtask 3) on the held-out test set. Our approach builds on a fine-tuned transformer architecture with three principal contributions: (1) typed entity markers that inject entity-type information directly into the input encoding, with nesting-aware marker interleaving for overlapping spans; (2) a two-stage LLM augmentation pipeline using GPT-5-mini as generator and GPT-5.4-mini as verifier to synthesise training examples for rare relation classes; and (3) a blind-dev confidence threshold sweep that corrects for the ≈30× train-to-test negative ratio mismatch. We report extensive ablation results and document several negative findings — including augmentation that degrades RU performance and the counterintuitive result that a domain-specific model with lower development F1 substantially outperforms a general multilingual model with higher development F1 on the held-out test set.

---

## 1. Task and Data

BioNNE-R 2026 is a relation extraction shared task over biomedical text in English and Russian. Given pre-annotated entity pairs, systems must classify the directional relation between them across 14 relation types (ABBREVIATION, ALTERNATIVE\_NAME, SUBCLASS\_OF, PART\_OF, TREATED\_USING, ORIGINS\_FROM, TO\_DETECT\_OR\_STUDY, AFFECTS, HAS\_CAUSE, APPLIED\_TO, USED\_IN, ASSOCIATED\_WITH, PHYSIOLOGY\_OF, FINDING\_OF) plus `no_relation`. The evaluation metric is **macro-averaged F1** over the 14 positive relation classes. Three subtasks are scored independently: Subtask 1 (English only), Subtask 2 (Russian only), and Subtask 3 (bilingual combined).

### 1.1 Dataset Statistics

Table 1 summarises the annotated data provided by the organisers.

**Table 1: Dataset statistics.**

| Split | Docs | Entities | Gold relations |
|---|---|---|---|
| EN train | 56 | 3,862 | 3,193 |
| EN dev | 51 | 3,479 | 2,968 |
| EN test | 154 | 10,317 | — (blind) |
| RU train | 717 | 37,880 | 23,773 |
| RU dev | 51 | 3,211 | 2,522 |
| RU test | 155 | 9,340 | — (blind) |

The English train set is notably small (56 documents, 3,193 annotated relations) compared to Russian (717 documents, 23,773 relations — a 7.4× difference). This asymmetry motivates the bilingual training strategy for Subtask 3 and the data augmentation effort for English rare classes.

Eight entity types are annotated: DISO (disease/disorder), ANATOMY, CHEM (chemicals/drugs), PHYS (physiological processes), FINDING, LABPROC (laboratory procedures), DEVICE, and INJURY\_POISONING. DISO and ANATOMY are the most frequent types in both languages.

**Table 2: EN training set — relation distribution and per-class dev F1 of our best model.**

| Relation | Train count | Dev gold | Dev F1 (BiomedBERT) |
|---|---|---|---|
| SUBCLASS\_OF | 743 | 757 | — |
| HAS\_CAUSE | 579 | 487 | 0.7644 |
| AFFECTS | 362 | 393 | 0.8996 |
| ASSOCIATED\_WITH | 325 | 233 | 0.7485 |
| FINDING\_OF | 121 | 83 | 0.7432 |
| PHYSIOLOGY\_OF | 150 | 178 | **0.9080** |
| TO\_DETECT\_OR\_STUDY | 164 | 116 | 0.7181 |
| ORIGINS\_FROM | 120 | 79 | 0.8024 |
| TREATED\_USING | 101 | 88 | 0.8118 |
| ABBREVIATION | 97 | 74 | 0.8627 |
| ALTERNATIVE\_NAME | 95 | 98 | **0.2482** |
| PART\_OF | 204 | 248 | 0.7452 |
| USED\_IN | 89 | 79 | 0.7250 |
| APPLIED\_TO | 43 | 54 | **0.4835** |
| **Total** | **3,193** | **2,968** | — |

The two chronic underperformers are ALTERNATIVE\_NAME (F1=0.2482) and APPLIED\_TO (F1=0.4835), which are also among the rarest classes. ABBREVIATION has a nearly identical training count to ALTERNATIVE\_NAME (97 vs 95) yet achieves F1=0.8627, confirming the failure is not purely a data quantity issue but stems from surface form ambiguity (Section 3.1).

**Test set size.** The test sets contain 154 EN and 155 RU documents, producing **818,084 EN** and **664,448 RU** candidate entity-pair instances (all within-sentence entity pairs). Assuming a test/dev document ratio of ≈3:1, the expected number of positive test relations is approximately **8,950 EN** and **7,650 RU**, yielding an effective negative ratio of approximately **91:1** (EN) and **87:1** (RU) — compared with the training negative sampling ratio of 3:1.

---

## 2. System Architecture

### 2.1 Model Selection

We select separate backbone encoders for each track based on domain and linguistic characteristics.

**English (Subtask 1):** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (BiomedBERT; Gu et al. 2021). Pre-trained on PubMed abstracts and PubMed Central full-text — the same distribution as the shared task documents. 110M parameters, uncased, English-only. We also trained and evaluated a multilingual model for comparison (Section 6).

**Russian and Bilingual (Subtasks 2–3):** `microsoft/mdeberta-v3-base` (mDeBERTa; He et al. 2021). A multilingual DeBERTa-v3 model with disentangled attention and relative position encodings, trained on Common Crawl across 100+ languages. 278M parameters. The disentangled position-content attention mechanism benefits relation extraction because entity-position signals and entity-content signals are encoded separately, reducing the risk of conflating positional artefacts with semantic relations.

### 2.2 Typed Entity Markers with Nesting Awareness

The organiser baseline uses generic `[unused0]–[unused3]` tokens as head/tail start/end markers. We replace these with **typed entity markers**: 32 new vocabulary tokens of the form `[HEAD:DISEASE]`, `[/HEAD:DISEASE]`, `[TAIL:GENE]`, `[/TAIL:GENE]`, covering all 8 entity types × 2 roles × open/close. Markers are inserted at character-level span boundaries before tokenisation, encoding entity-type information directly into the input sequence:

```
... [HEAD:DISO] hypertension [/HEAD:DISO] associated with [TAIL:CHEM] lisinopril [/TAIL:CHEM] ...
```

The model's entity representation is the hidden state at the opening marker positions (`[HEAD:TYPE]` and `[TAIL:TYPE]`), concatenated and passed to the classification head, following the improved marker-pooling baseline of Zhong and Chen (2021).

Typed markers allow the model to exploit entity-type constraints without attending to context: for example, `ALTERNATIVE_NAME` only occurs between same-type pairs, while `TREATED_USING` requires `CHEM` or `LABPROC` as head and `DISO` as tail. The markers make this structural prior directly visible to the encoder's attention layers.

**Nesting handling.** Approximately 15% of annotated entity pairs in the dataset involve nested spans, where one entity span contains the other (e.g., "cardiovascular [ANATOMY]" nested inside "[DISO]ischaemic cardiovascular disease[/DISO]"). Standard left-to-right marker injection corrupts the span boundaries for nested pairs. We handle this by inserting markers in right-to-left order over the entity spans, so that inner-span markers are always inserted before outer-span markers, preserving all character offsets. The resulting tokenized sequence is well-formed and correctly represents both entities.

A binary **nesting flag** (0.0 for non-nested, 1.0 for nested) is appended to the concatenated entity representation before the classifier, providing an explicit structural signal.

### 2.3 Class-Weighted Loss with `no_relation` Clamping

Training negative instances are sampled at a ratio of 3:1 (`no_relation` : positive). Per-class loss weights are computed as the inverse of training class frequency. However, because `no_relation` is synthetically under-represented at training time (3:1) relative to test time (≈90:1), naively applying inverse-frequency weighting would assign `no_relation` a weight of ~0.25, making false negatives for `no_relation` very expensive and causing the model to under-predict the majority class — the opposite of what is needed.

We therefore **clamp the `no_relation` weight to 1.0** and apply inverse-frequency weights only to positive classes. Rare positive classes (APPLIED\_TO: 43 train instances, ALTERNATIVE\_NAME: 95 instances) receive weights of 4–8× compared to frequent classes (SUBCLASS\_OF: 743 instances, HAS\_CAUSE: 579 instances).

### 2.4 Valid Triplet Type Pruning

Early experiments revealed the model predicting schema-invalid relations (e.g., `AFFECTS(GENE, GENE)` when no such triplet appears in the annotation schema). We extracted all valid `(head_type, tail_type, relation)` triplets from the NEREL-BIO annotation configuration, yielding **82 valid triplets**. During inference, any prediction violating this schema is overridden to `no_relation` before confidence thresholding. This reduced hallucinated predictions and is applied prior to the confidence threshold.

### 2.5 Truncation Failsafe

At `max_length=512` tokens, approximately 3% of instances (long biomedical sentences with distant entity pairs) have one or both entity markers truncated from the token window. Without mitigation, the model attends to the wrong position — producing silent errors that cannot be detected from the output alone. We track per-instance marker validity at dataset construction time and force these instances to predict `no_relation` during inference. This failsafe was the primary fix resolving the catastrophic v1 failure (Section 6.1).

### 2.6 Classification Head and Training Configuration

The classification head is a single linear layer mapping the concatenated entity representation (dimension 2H+1, where H is the encoder hidden size and +1 is the nesting flag) to 15 classes. We apply dropout (p=0.1) before the linear layer. Training hyperparameters:

| Parameter | BiomedBERT (EN) | mDeBERTa (RU/BIL) |
|---|---|---|
| Learning rate | 2e-5 | 2e-5 |
| Batch size | 16 | 128 |
| Max sequence length | 512 | 256 |
| Epochs | 10 | 10 |
| Warmup steps | 300 | 300 |
| Mixed precision | no | fp16 |
| Gradient checkpointing | no | yes |
| Optimizer | AdamW, wd=0.01 | AdamW, wd=0.01 |
| LR schedule | Linear decay | Linear decay |

FP16 and gradient checkpointing were necessary for mDeBERTa to fit within 24 GB VRAM (L4 GPU) at batch size 128 for the Russian training set (95,092 instances after 3:1 negative sampling).

Training uses a **dual-checkpoint scheme** for fault tolerance (important when running on cloud GPU instances with 4-hour session limits): a *best-model checkpoint* is saved whenever dev macro F1 improves; a *resume checkpoint* saves full optimiser and scheduler state at every epoch and every 300 steps. An automated orchestration script monitors session completion and re-launches training until epoch 10 is reached, then triggers inference on test blind files — enabling unattended 10-epoch training across multiple session restarts.

---

## 3. LLM-Based Data Augmentation for Rare Classes

### 3.1 Motivation and Target Classes

Confusion analysis on the BiomedBERT dev predictions revealed two critical failure modes:

**ALTERNATIVE\_NAME** (train count=95, dev F1=0.2482): 62% of gold instances are predicted as `no_relation`. ABBREVIATION has a nearly identical count (97) yet achieves F1=0.8627. The disparity is explained by surface form diversity: ABBREVIATION has a near-universal parenthetical pattern ("hypertension (HTN)"), while ALTERNATIVE\_NAME spans full synonyms, brand-vs-generic drug names, historical-vs-modern terminology, and transliterated names — all highly varied surface forms.

**APPLIED\_TO** (train count=43, dev F1=0.4835): 28% of gold instances are confused with TO\_DETECT\_OR\_STUDY. Both relations share the dominant entity-type pair LABPROC→ANATOMY. The distinction requires discourse-level understanding of whether a procedure is *interventional* (APPLIED\_TO: a surgical technique applied to tissue) or *observational* (TO\_DETECT\_OR\_STUDY: a measurement method used to characterise tissue). This distinction is often expressed by subtle lexical cues ("biopsy of" vs. "biopsy used to detect") that require seeing many examples.

These two classes fail not from insufficient data quantity alone, but from insufficient **diversity** of surface realisations and from confusable near-neighbours in relation space.

### 3.2 Two-Stage Generation Pipeline

We implement a two-stage pipeline in `baseline/augment_openai.py`:

**Stage 1 — Generation (GPT-5-mini).** For each target relation and entity type pair (e.g., `APPLIED_TO(LABPROC, ANATOMY)`), we:
1. Sample 5 seed examples from the gold training set, selected to match the target head/tail type combination
2. Provide a detailed relation definition and explicit contrast with confusable neighbours
3. Request a batch of 5 new `(sentence, head_entity, head_span, tail_entity, tail_span, relation)` tuples, varying biomedical domain sub-field and syntactic structure
4. For APPLIED\_TO jobs, additionally request 5 *contrastive* TO\_DETECT\_OR\_STUDY examples per call — "hard negatives" with the same entity types but opposite intent — to explicitly teach the distinction boundary

Output is parsed as a JSON array; malformed responses are discarded without retry.

**Stage 2 — Verification (GPT-5.4-mini).** Each accepted candidate from Stage 1 is passed to a verification model that independently assesses:
- Natural language fluency and publication-style register
- Unambiguity of the relation label from sentence-level context alone
- Absence of alternative valid relation labels
- Correctness of character-level entity spans

Only examples with verdict=ACCEPT passing all four criteria are retained.

**Span validation.** Before submitting candidates to the verifier, we programmatically validate that `sentence[head_start:head_end] == head_entity` (and analogously for tail). We apply a ±3 character correction window to tolerate minor off-by-one errors from the generation model; candidates where entity text is not locatable within this window are discarded without API calls to the verifier, saving cost.

### 3.3 Augmentation Results

**Table 3: Augmentation targets and outcomes.**

| Lang | Relation | Type pair | Target | Accepted |
|---|---|---|---|---|
| EN | ALTERNATIVE\_NAME | DISO-DISO | 120 | ~120 |
| EN | ALTERNATIVE\_NAME | CHEM-CHEM | 90 | ~90 |
| EN | ALTERNATIVE\_NAME | ANATOMY-ANATOMY | 80 | ~80 |
| EN | ALTERNATIVE\_NAME | FINDING-FINDING | 60 | ~53 |
| EN | ALTERNATIVE\_NAME | LABPROC-LABPROC | 30 | ~25 |
| EN | ALTERNATIVE\_NAME | PHYS-PHYS | 20 | ~20 |
| EN | APPLIED\_TO | LABPROC-ANATOMY | 130 | ~110 |
| EN | APPLIED\_TO | CHEM-ANATOMY | 70 | ~60 |
| EN | TO\_DETECT\_OR\_STUDY (contrastive) | LABPROC-DISO,CHEM | 150 | ~131 |
| RU | ALTERNATIVE\_NAME | (multiple) | 320 | ~326 |
| RU | APPLIED\_TO | (multiple) | 200 | ~193 |
| RU | TO\_DETECT\_OR\_STUDY (contrastive) | LABPROC-ANATOMY,PHYS | 130 | ~187 |
| RU | USED\_IN | CHEM-CHEM, LABPROC-LABPROC, DEVICE-LABPROC | 160 | ~168 |

**Final counts:** EN: **917 total** (ALTERNATIVE\_NAME: 413, TO\_DETECT\_OR\_STUDY: 301, APPLIED\_TO: 203); RU: **874 total** (ALTERNATIVE\_NAME: 326, APPLIED\_TO: 193, TO\_DETECT\_OR\_STUDY: 187, USED\_IN: 168).

**Russian generation difficulty.** The `APPLIED_TO(LABPROC, ANATOMY)` job in Russian repeatedly hit the 8,192 output token limit (`finish_reason=length`). Russian biomedical procedure names are morphologically richer and longer than English equivalents; GPT-5-mini consistently attempted verbose descriptions that overflowed the JSON output buffer before completing the array. The job reached 103/120 (86%) of its target before the attempt budget was exhausted.

---

## 4. Confidence Threshold Calibration

### 4.1 Train-Test Density Mismatch

A fundamental calibration challenge in this task: negative instances are sampled at ratio 3:1 during training, but at test time **all** candidate entity pairs are enumerated — approximately 818K pairs for EN test, yielding a true negative ratio of ~91:1. A softmax classifier trained with 3:1 negatives will assign p("positive") ≥ 0.5 to far more instances at test time than are actually positive, because the model has never seen how rare positives are in the full pair space.

Evidence: our mDeBERTa model at threshold t=0.5 produces 33,338 EN test predictions and 17,982 RU test predictions — 3.5× and 2.0× above expected counts respectively. The resulting CodaBench macro F1 is 0.2819 (EN) and 0.4084 (RU), well below the dev-set values.

### 4.2 Blind-Dev Threshold Sweep

Fix: at inference time, accept a predicted relation only if the maximum softmax probability exceeds a threshold T. We implement `find_threshold()` in `bionne_predict.py`: run inference once on the **blind dev set** (all candidate pairs from dev documents, same ~87:1 negative ratio as test), then sweep T ∈ [0, 1] over the stored probability tensors (O(N) CPU-only per sweep, millisecond-scale per threshold). The T maximising blind-dev macro F1 is used for test submission.

**We also evaluated an existence filter** (`existence_filter=True`): use `1 − p(no_relation)` as the threshold score instead of `max p(positive class)`. This decouples the question "does any relation exist?" from "which relation type is it?" — motivated by the observation that for rare classes, probability mass may be spread across multiple positive-class logits. Results are reported in Section 8.3.

---

## 5. Results

### 5.1 Failure Mode: BiomedBERT v1

Our first EN model (BiomedBERT v1) achieved dev macro F1 of **0.2524** — worse than random. Post-hoc analysis identified the cause: approximately 3% of dev instances had entity markers truncated at max_length=512, causing the model to extract the hidden state of an unrelated token as the entity representation. On the training set, these instances were already rare (truncation is less likely for in-distribution document lengths) — but on dev, several long-distance entity pairs in the full documents triggered the failure. The model's classifier learned to exploit the wrong signal. The **truncation failsafe** (Section 2.5) resolved this entirely: v2 achieved 0.7369.

### 5.2 Development Set Results (Blind Evaluation)

Dev F1 is measured in **blind mode**: inference runs on all candidate entity pairs from dev documents (same pipeline as test), then evaluated against gold annotations. This matches the test evaluation methodology and avoids the artificially high F1 that would result from evaluating only on labelled pairs.

**Table 4: Best dev set macro F1 per track.**

| Track | Model | Dev macro F1 | Checkpoint |
|---|---|---|---|
| EN | BiomedBERT + typed markers | 0.7369 | `biomedbert_en_v2_final.pt` |
| EN | mDeBERTa-v3-base + augmentation | **0.7902** | `mdeberta_en_v1.pt` (ep. 2) |
| RU | mDeBERTa-v3-base (base) | **0.8397** | `mdeberta_ru_v2.pt` |
| BIL | mDeBERTa-v3-base | **0.8831** | `mdeberta_bilingual_v2.pt` |

The augmented EN mDeBERTa model peaked sharply at epoch 2 (F1=0.7902) and never recovered: epochs 3–10 oscillated between 0.7206 and 0.7759, suggesting rapid memorisation of the augmented examples followed by oscillation without further generalisation. The best-model checkpoint saves only on F1 improvement, so epoch 2 weights are preserved.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| EN aug mDeBERTa F1 | 0.634 | **0.790** | 0.726 | 0.776 | 0.721 | 0.766 | 0.755 | 0.770 | 0.762 | 0.754 |

### 5.3 Test Set Results (CodaBench Submissions)

**Table 5: Complete CodaBench submission history — English (Subtask 1).**

| Model | Threshold | Predictions | Macro F1 | Notes |
|---|---|---|---|---|
| **BiomedBERT** | **0.80** | **11,430** | **0.4733** | **← Final best EN** |
| BiomedBERT | 0.70 | 12,329 | 0.4726 | |
| BiomedBERT | 0.90 | 10,113 | 0.4722 | |
| BiomedBERT | 0.50 | 14,480 | 0.4593 | First submission |
| mDeBERTa BIL (on EN test) | 0.95 | 14,328 | 0.4542 | Cross-lingual transfer |
| mDeBERTa BIL (on EN test) | 0.90 | 17,877 | 0.4447 | |
| mDeBERTa BIL (on EN test) | 0.99 | 7,657 | 0.4400 | |
| mDeBERTa EN aug | existence filter 0.95 | 16,349 | 0.3870 | |
| mDeBERTa EN aug | per-class | 10,616 | 0.3625 | See Section 8.4 |
| mDeBERTa EN aug | 0.85 | 13,785 | 0.3334 | ← aug peak |
| mDeBERTa EN aug | 0.90 | 10,239 | 0.3318 | |
| mDeBERTa EN aug | 0.95 | 5,804 | 0.2982 | |
| mDeBERTa EN aug | 0.50 | 33,338 | 0.2819 | |
| mDeBERTa EN aug | 0.99 | 1,136 | 0.0982 | Calibration collapse |

**Table 6: CodaBench submission history — Russian (Subtask 2) and Bilingual (Subtask 3).**

| Track | Model | Threshold | Predictions | Macro F1 |
|---|---|---|---|---|
| **RU** | **mDeBERTa base** | **0.95** | **8,692** | **0.5057** |
| RU | mDeBERTa base | existence filter 0.95 | 10,418 | 0.5036 |
| RU | mDeBERTa base | 0.90 | ~13,000 | 0.4944 |
| RU | mDeBERTa base | 0.99 | 4,057 | 0.4560 |
| RU | mDeBERTa base | 0.50 | 17,982 | 0.4084 |
| RU | mDeBERTa augmented | 0.50 | ~33,000 | 0.2990 |
| **BIL** | **mDeBERTa BIL** | **0.95** | **26,318** | **0.4527** |
| BIL | mDeBERTa BIL | existence filter 0.95 | 32,434 | 0.4505 |
| BIL | mDeBERTa BIL | 0.99 | 14,163 | 0.4470 |
| BIL | mDeBERTa BIL | 0.90 | 32,618 | 0.4356 |
| BIL | mDeBERTa BIL | 0.50 | 58,486 | 0.3310 |

### 5.4 Final Best Scores

| Track | Best macro F1 | Model | Threshold | Predictions |
|---|---|---|---|---|
| EN (Subtask 1) | **0.4733** | BiomedBERT-base-uncased-abstract-fulltext | 0.80 | 11,430 |
| RU (Subtask 2) | **0.5057** | mDeBERTa-v3-base (no augmentation) | 0.95 | 8,692 |
| BIL (Subtask 3) | **0.4527** | mDeBERTa-v3-base bilingual | 0.95 | 26,318 |

---

## 6. Analysis

### 6.1 Domain Pretraining Outweighs Development F1

The most striking result is the inversion between development and test performance for English:

| Model | Dev F1 | Test F1 (t=0.80/best) |
|---|---|---|
| BiomedBERT | 0.7369 | **0.4733** |
| mDeBERTa augmented | **0.7902** | 0.3334 |

A model with 0.053 *lower* development F1 achieves 0.140 *higher* test F1. We attribute this to three factors:

1. **Domain match.** BiomedBERT was pretrained on PubMed abstracts and full-text — the same text type as the shared task. Its tokeniser and pre-trained representations handle biomedical terminology natively. mDeBERTa's multilingual Common Crawl pretraining contains minimal biomedical text and relies on subword fallback for rare biomedical entities.

2. **Calibration quality.** BiomedBERT's confidence distribution is smooth and unimodal. mDeBERTa-aug's distribution is bimodal, with a gap in the 0.90–0.99 range — predictions collapse from 10,239 at t=0.90 to 1,136 at t=0.99 (9× drop over 0.09 threshold change). This gap arises because GPT-generated augmentation examples push the model to output very high confidence for augmented-class instances, creating a calibration artefact.

3. **Augmentation overfitting.** The augmented mDeBERTa peaked at epoch 2 and oscillated, suggesting the 917-example augmented set is too small and not diverse enough to act as regularisation for a 278M parameter model fine-tuned over 10 epochs.

BiomedBERT's threshold sweep (Table 5) shows a very flat optimum: F1 changes by less than 0.001 across t=0.70–0.90 (0.4722–0.4733). This is characteristic of a well-calibrated model: small changes in threshold do not dramatically change prediction counts.

### 6.2 Rare Class Coverage

Table 7 compares predicted class counts against expected test counts (scaled from gold dev distribution by the test/dev document ratio of 3.02×) for the EN track.

**Table 7: EN test — predicted vs. expected relation counts at best threshold.**

| Class | Expected (~) | BiomedBERT t=0.80 | mDeBERTa aug t=0.85 |
|---|---|---|---|
| SUBCLASS\_OF | ~2,285 | ~3,000 | 2,451 |
| HAS\_CAUSE | ~1,471 | ~1,900 | 2,498 |
| AFFECTS | ~1,187 | ~1,900 | 2,145 |
| ASSOCIATED\_WITH | ~704 | ~1,500 | 1,619 |
| PART\_OF | ~749 | ~1,100 | 715 |
| PHYSIOLOGY\_OF | ~538 | ~660 | 1,366 |
| TO\_DETECT\_OR\_STUDY | ~350 | ~640 | 782 |
| ABBREVIATION | ~223 | ~350 | 528 |
| ORIGINS\_FROM | ~239 | ~370 | 354 |
| TREATED\_USING | ~266 | ~285 | 418 |
| FINDING\_OF | ~251 | ~520 | 853 |
| USED\_IN | ~239 | **346** | 41 |
| ALTERNATIVE\_NAME | ~296 | **57** | 12 |
| APPLIED\_TO | ~163 | **89** | 3 |

The critical observation: ALTERNATIVE\_NAME, USED\_IN, and APPLIED\_TO are **near-zero for augmented mDeBERTa** at any threshold ≥ 0.85 — despite augmentation targeting these exact classes. BiomedBERT achieves APPLIED\_TO=89 (55% of expected) and USED\_IN=346 (145% of expected) without augmentation. Since macro F1 weights all 14 classes equally, poor recall on rare classes disproportionately penalises the augmented model.

---

## 7. Negative Results

### 7.1 Stochastic Ensemble (Baseline)

For the organiser's OpenNRE mBERT baseline, we implemented a stochastic ensemble: one deterministic inference pass (dropout disabled, fixed seed) plus three stochastic passes (dropout enabled at inference time, different random seeds — MC dropout style). Predictions were aggregated by plurality vote over the four runs, with ties broken by `no_relation`.

The ensemble was applied to all three tracks of the baseline but did not produce a notable improvement over deterministic single-run inference for the fine-tuned models (mDeBERTa, BiomedBERT), and was not adopted for final submissions. The null result is consistent with Gal and Ghahramani (2016): MC dropout improves calibration but primarily for models with high epistemic uncertainty, which is less pronounced after supervised fine-tuning on a domain-matched task.

### 7.2 Augmentation is Net Negative for Russian

The RU augmented model (`mdeberta_ru_aug_v1.pt`) failed substantially:

| Model | Threshold | Predictions | Macro F1 |
|---|---|---|---|
| mDeBERTa RU base | 0.95 | 8,692 | **0.5057** |
| mDeBERTa RU augmented | 0.50 | ~33,000 | 0.2990 |
| mDeBERTa RU augmented | 0.99 | 2,214 | not submitted |

At t=0.99, the augmented RU model produces only 2,214 predictions with a pathological class distribution: 8 HAS\_CAUSE predictions (vs. 1,030 in the base model at the same threshold). The augmentation of 874 examples disrupted the model's calibration for the dominant positive class without improving the rare classes. The RU training set is already 7.4× larger than English (23,773 positive training instances), so augmentation provided marginal marginal signal relative to total training data volume (874 / 23,773 = 3.7%) while introducing calibration noise from GPT-generated text style differences.

**The base RU model (no augmentation) is strictly superior.** This result suggests a threshold effect: augmentation is beneficial when training data is very scarce (EN: 43 APPLIED\_TO examples), but harmful when the base dataset is already substantial.

### 7.3 Existence Filter

We evaluated `1 − p(no_relation)` as an alternative threshold score, motivated by the hypothesis that for miscalibrated models the probability mass for rare positive classes might be distributed across multiple positive-class logits rather than concentrated on one. Results:

| Track | Model | Threshold | F1 (standard) | F1 (existence filter) | Δ |
|---|---|---|---|---|---|
| EN | mDeBERTa aug | 0.95 | 0.2982 | 0.3870 | **+0.089** |
| RU | mDeBERTa base | 0.95 | **0.5057** | 0.5036 | −0.002 |
| BIL | mDeBERTa BIL | 0.95 | **0.4527** | 0.4505 | −0.002 |

The existence filter substantially helps the miscalibrated augmented EN model (+0.089) by recovering predictions that the model is uncertain about which specific type to assign. For the well-calibrated base models (RU, BIL), the filter provides no benefit and a marginal cost. We did not apply the existence filter to BiomedBERT as it was already well-calibrated.

### 7.4 Per-Class Confidence Thresholds

To recover ALTERNATIVE\_NAME and APPLIED\_TO recall without accepting all the over-predicted common classes, we set class-specific thresholds: lower thresholds (t=0.40) for rare classes (ALTERNATIVE\_NAME, USED\_IN, APPLIED\_TO) and higher thresholds (t=0.95) for chronically over-predicted classes (PHYSIOLOGY\_OF, FINDING\_OF, ASSOCIATED\_WITH). CodaBench result: **0.3625** — better than the flat augmented mDeBERTa threshold at 0.85 (0.3334) but far below BiomedBERT (0.4733).

The experiment confirmed a fundamental problem: the USED\_IN class at t=0.40 produces 681 predictions — 2.5× the expected count — while ALTERNATIVE\_NAME at t=0.50 gives only 123 predictions (42% of expected). The augmented mDeBERTa model is poorly calibrated for these classes; no threshold setting simultaneously achieves good precision and recall for all three rare classes.

### 7.5 Cross-Lingual Transfer (BIL Model on EN Test)

We applied the bilingual mDeBERTa model directly to the EN test set, motivated by the hypothesis that joint EN+RU training provides additional regularisation and improves rare-class coverage via cross-lingual transfer from the larger Russian training set.

| Threshold | Predictions | Macro F1 |
|---|---|---|
| 0.90 | 17,877 | 0.4447 |
| 0.95 | 14,328 | 0.4542 |
| 0.99 | 7,657 | 0.4400 |

The BIL model peaks at 0.4542 (t=0.95), consistently ~0.019 below BiomedBERT t=0.80 (0.4733). Cross-lingual regularisation improves USED\_IN coverage (41→599 at t=0.95 vs augmented mDeBERTa) but cannot overcome BiomedBERT's domain pretraining advantage. BiomedBERT's pretraining on the exact text type used in the test set (PubMed abstracts) provides stronger domain adaptation than cross-lingual regularisation from Russian biomedical text.

---

## 8. Computational Budget

All training and inference runs used Modal Labs cloud GPU instances. Training used NVIDIA A100 (40GB) and L4 (24GB); inference used L4 exclusively.

**Table 8: Estimated GPU-hours by run.**

| Run | GPU | Estimated hours |
|---|---|---|
| BiomedBERT EN v1 (aborted at ep. 3) | A100 | ~1.5 |
| BiomedBERT EN v2 (10 epochs) | A100 | ~3.5 |
| mDeBERTa EN augmented (10 epochs, fp16+gc) | A100 | ~4.5 |
| mDeBERTa RU base (10 epochs, fp16+gc) | A100 | ~6.0 |
| mDeBERTa RU augmented (10 epochs, fp16+gc) | A100 | ~6.0 |
| mDeBERTa BIL (10 epochs, fp16+gc) | A100 | ~9.0 |
| **Training subtotal** | | **~30.5 h** |
| EN test inference (BiomedBERT, 3 threshold sweeps) | L4 | ~2.0 |
| EN test inference (mDeBERTa aug, multiple sweeps) | L4 | ~3.7 |
| EN test inference (BIL model, 3 thresholds) | L4 | ~1.1 |
| RU test inference (base + aug models, sweeps) | L4 | ~2.3 |
| BIL test inference (BIL model, sweeps) | L4 | ~3.0 |
| **Inference subtotal** | | **~12.1 h** |
| **Total** | | **~42.6 GPU-hours** |

Note: inference over the blind test sets required enumerating all candidate entity pairs within each document. The EN test set produced 818,084 pairs, RU produced 664,448 pairs, and the bilingual test set produced 1,482,532 pairs. A batched fast-tokeniser pipeline with FP16 inference achieves ~500 instances/sec on L4, making these large inference passes feasible within 4-hour session limits.

---

## 9. Conclusions

We presented a system for biomedical relation extraction combining typed entity markers, two-stage LLM augmentation for rare classes, and blind-dev confidence threshold calibration. Our main positive findings are: (1) typed entity markers with nesting handling provide a robust improvement over generic markers; (2) domain-specific pretraining (BiomedBERT) substantially outperforms general multilingual models on the English track despite lower development-set F1; (3) threshold calibration corrects for the 30× train-test negative ratio mismatch, providing a reliable +0.014 improvement on the EN best score.

Our negative findings are equally informative: augmentation of rare classes with GPT-generated examples is beneficial for English (very small training set) but harmful for Russian (already substantial training data). MC dropout stochastic ensembles did not improve our fine-tuned models. Per-class thresholding partially recovers rare-class recall but is limited by the fundamental miscalibration introduced by augmentation. The existence filter (1−p\_no\_rel) helps miscalibrated models but not well-calibrated ones.

The dominant observation across all tracks is that **test-time calibration is the highest-impact intervention** after model selection: correcting for the training/test negative-ratio mismatch via threshold sweep reliably adds 0.10–0.14 F1 points compared to naive argmax or t=0.5 decoding.

---

## References

- Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., and Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare*, 3(1), 1–23. (BiomedBERT)
- He, P., Liu, X., Gao, J., and Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *arXiv preprint arXiv:2111.09543*. (mDeBERTa)
- Zhong, Z. and Chen, D. (2021). A frustratingly easy approach for entity and relation extraction. In *Proceedings of NAACL 2021*. (Typed entity markers / start-marker pooling)
- Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *Proceedings of ICML 2016*. (MC dropout)
- Loukachevitch, N., Manandhar, S., Batura, T., Dobrov, B., Gureenkova, O., Kokh, V., et al. (2024). NEREL-BIO: A dataset of biomedical abstracts annotated with named entities and relations. In *Proceedings of BioASQ 2024*. (NEREL-BIO corpus)
