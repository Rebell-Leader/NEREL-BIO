# BioNNE-R 2026: Strategy Guide for a Final-Year PhD Researcher
## Executive Summary
The BioNNE-R Shared Task at BioASQ 2026 frames biomedical nested relation extraction as a multi-class classification problem over pre-annotated entity pairs drawn from English and Russian PubMed abstracts. The provided baseline (OpenNRE + `bert-base-multilingual-cased` with entity markers) achieves macro F1 of **0.6944 / 0.7166 / 0.7211** on English / Russian / Bilingual dev sets. Given the student's computational constraints (HF inference, Colab, limited A100 access) and tight timeline (evaluation ends **May 14**), the optimal strategy is **not to invent novel architectures** but to pursue three high-ROI, low-implementation-overhead improvements on top of the provided baseline: (1) backbone swap to a stronger biomedical/multilingual encoder, (2) typed entity marker enhancement, and (3) careful negative-sample handling and class-weighting. These changes, each independently validated in the literature, can be stacked to push F1 well above 0.75 without requiring new datasets or expensive training.[^1]

***
## Task & Data Profile
### Task Formulation
BioNNE-R is a **relation classification** task: given pre-annotated entity pairs (head, tail) with their spans and types within a document, predict one of 14 relation types or no relation. The key complexity is *nestedness* — one entity may contain another within its character span (e.g., `atopic dermatitis` [DISO] nested inside `dermatitis` [DISO]), meaning standard entity masking approaches fail. The evaluation metric is **macro-averaged F1 over all 14 relation types**, which penalizes low performance on rare classes regardless of how well `no_relation` is handled.[^1][^2]
### Relation Types and Imbalance
The 14 relation classes span a wide semantic range:

| Relation | Typical Entity Pair | Expected Frequency |
|---|---|---|
| SUBCLASS_OF | DISO→DISO | High (nesting-induced) |
| PART_OF | ANATOMY→ANATOMY | High |
| ALTERNATIVE_NAME | CHEM→CHEM, DISO→DISO | High |
| FINDING_OF | FINDING→ANATOMY | Medium |
| ASSOCIATED_WITH | DISO→DISO, PHYS→DISO | Medium |
| TREATED_USING | DISO→CHEM | Medium |
| AFFECTS | CHEM→PHYS, CHEM→DISO | Medium |
| HAS_CAUSE | FINDING/DISO→CHEM/DISO | Medium |
| PHYSIOLOGY_OF | PHYS→ANATOMY | Medium |
| TO_DETECT_OR_STUDY | LABPROC→DISO | Lower |
| APPLIED_TO | DEVICE→ANATOMY | Lower |
| ORIGINS_FROM | DISO→ANATOMY | Lower |
| USED_IN | DEVICE→LABPROC | Lower |
| ABBREVIATION | any→any | Lower |

Relations like `SUBCLASS_OF` and `PART_OF` are **structurally induced by nesting** (e.g., `dermatitis` is a subclass of `atopic dermatitis`) and will be highly frequent. Rare types like `USED_IN` will drag macro F1 down heavily. Any improvement strategy must explicitly address this imbalance.[^2]
### Nestedness Taxonomy
From the foundational analysis on the NEREL corpus (precursor to NEREL-BIO), in-sentence relations split into three structural categories:[^2]

- **External (53.9%)**: head and tail entities are non-overlapping spans
- **Nested (15%)**: both entities sit within the span of a single longer entity
- **Cross-entity (7.1%)**: one entity is external, the other is internal to a third entity

The baseline handles all three via a corrected entity marker scheme, but this is a minimum bar — stronger encoders will extract richer contextual signals for all three types.

***
## Baseline Analysis
### Provided Setup
The baseline uses **OpenNRE** with `bert-base-multilingual-cased` (mBERT), entity markers inserted at entity boundaries, and negative sampling over all entity pair combinations. Key components:[^1][^3]

1. **Entity markers**: `<E1> ... </E1>` and `<E2> ... </E2>` tags wrapping head and tail spans, inserted correctly for nested entities (i.e., interleaved, not duplicated)[^2]
2. **Negative sampling**: all entity pairs not in the gold relation set are labeled `no_relation`
3. **Prediction**: only pairs predicted with a positive relation class appear in the output TSV; `no_relation` pairs are suppressed[^1]
4. **Dev macro F1**: EN: 0.6944, RU: 0.7166, Bilingual: 0.7211
### Key Baseline Weakness
`bert-base-multilingual-cased` (mBERT) is a general-purpose multilingual model. It was not pre-trained on PubMed text and does not have a biomedical vocabulary. In the BioNNE-L 2025 analogue task, teams using domain-specific encoders (SapBERT, BERGAMOT) consistently outperformed general multilingual baselines by 10–15 accuracy points. The same pattern holds for relation extraction: PubMedBERT and BioLinkBERT-large consistently lead on biomedical RE benchmarks (ChemProt, DDI, BC5CDR). Swapping the encoder is therefore the single highest-leverage change available.[^4][^5][^6]

***
## Lessons from BioNNE-L 2025 (Directly Analogous Task)
The BioNNE-L 2025 task (Nested Entity Linking on the same NEREL-BIO corpus, same EN/RU/Bilingual track structure) provides a near-perfect prior for predicting what will win BioNNE-R 2026.[^4]
### Official Results Summary
| Team | Multilingual @1 | English @1 | Russian @1 | Approach |
|---|---|---|---|---|
| LYX_DMIIP_FDU | **0.68** | 0.66 | 0.71 | BERGAMOT fine-tuning w/ contrastive learning[^4] |
| BlancaPlanca | 0.67 | 0.64 | **0.72** | BERGAMOT zero-shot + language-specific preprocessing[^4] |
| verbanexialab | — | **0.70** | — | SapBERT + lexical+semantic reranking (Jaccard, Levenshtein)[^4] |
| MSM Lab | 0.63 | 0.64 | 0.65 | SapBERT EN + multilingual SapBERT RU, 2-step pipeline[^4] |
| dstepakov | 0.63 | — | 0.70 | RoBERTa contrastive fine-tuning (InfoNCE)[^4] |
| ICUE | 0.58 | 0.51 | 0.62 | BioSyn + DeepSeek-R1-Distill-Llama-8B reranking[^4] |
| Baseline | 0.53 | 0.57 | 0.52 | BERGAMOT zero-shot[^4] |
### Critical Insight: LLM Reranking Did NOT Help
Team ICUE added DeepSeek-R1-Distill-Llama-8B as a reranker on top of BioSyn retrieval, yet ranked **5th** — below pure BERT-based teams. This is consistent with findings from document-level bioRE evaluations: fine-tuned BERT models outperform prompted LLMs on structured RE classification tasks. **Do not invest significant effort in LLM-based approaches for relation classification itself.** The student's LLM token budget is better spent on data augmentation or label generation (see below).[^7]
### Critical Insight: Domain-Specific Encoders Win
All top-4 teams used BERGAMOT or SapBERT — both biomedical BERT variants. The winning bilingual team fine-tuned BERGAMOT with contrastive learning on the task data. This is exactly the approach to replicate for BioNNE-R, substituting task-appropriate encoders.

***
## Recommended Strategy: Three Stacked Improvements
The following three improvements are ordered by implementation effort (ascending) and estimated F1 gain (descending). Each is independently publishable as an ablation finding.
### Improvement 1: Backbone Swap to a Stronger Encoder (Est. +3–6 F1 pts)
**Action**: Replace `bert-base-multilingual-cased` with `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` for English and `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large` (multilingual SapBERT) or `microsoft/mdeberta-v3-base` for Russian/Bilingual.

**Rationale**: PubMedBERT consistently leads biomedical RE benchmarks. BioLinkBERT-large outperforms PubMedBERT on 4 of 5 biomedical RE datasets with small margins. mDeBERTa-v3-base is the strongest general multilingual encoder and substantially outperforms mBERT on cross-lingual classification. In the BioNNE-L 2025 analogue, all top performers used biomedical encoders.[^4][^5]

**Resources**: All models available on HuggingFace, fine-tuning on the provided train split fits within ~4GB GPU memory for base-size models. Colab T4 is sufficient; A100 is needed only for large variants.

**Implementation cost**: ~2 hours (config change in OpenNRE or direct HF fine-tuning script).

Recommended encoder priority:

| Track | Primary Encoder | Fallback |
|---|---|---|
| English (Subtask 1) | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | `allenai/biomed_roberta_base` |
| Russian (Subtask 2) | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large` | `microsoft/mdeberta-v3-base` |
| Bilingual (Subtask 3) | `microsoft/mdeberta-v3-base` | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large` |
### Improvement 2: Typed Entity Markers with Nesting-Aware Encoding (Est. +2–4 F1 pts)
**Baseline markers**: `<E1> head text </E1> ... <E2> tail text </E2>` (type-agnostic)

**Action**: Upgrade to **typed entity markers** that embed entity type information directly into the marker tokens:

```
<E1:DISO> atopic dermatitis <E1:DISO/> ... <E2:DISO> dermatitis <E2:DISO/>
```

The "typed marker" scheme was introduced in Zhong & Chen (2021) and achieves F1 of 74.6% on TACRED versus 72.8% for vanilla entity markers — a 1.8 F1 point improvement. For biomedical RE, the gain is larger because entity type is highly informative for relation type prediction (e.g., CHEM→DISO pairs strongly constrain to TREATED_USING/AFFECTS/HAS_CAUSE).[^8][^9][^10][^11]

**Nesting-specific augmentation**: Supplement the input with an explicit *nesting flag* feature. Based on the NEREL analysis, a binary indicator `nested=True/False` or an additional marker variant `<E1:DISO:inner>` for nested entities helps the model distinguish nested relations (SUBCLASS_OF, PART_OF) from external relations. This proved effective even with a simple dense+FastText baseline (F1 jump: 72.5→75.2).[^2]

**Implementation**: Modify the tokenizer preprocessing in OpenNRE to insert typed markers; add a binary nesting feature concatenated to the [CLS] pooled output before the classification head. ~4 hours.
### Improvement 3: Class-Weighted Loss and Negative Sampling Tuning (Est. +2–3 F1 pts on macro F1)
The task uses macro-averaged F1, meaning rare classes like `USED_IN`, `ABBREVIATION`, `ORIGINS_FROM` contribute equally to the final score as common classes. The baseline uses standard negative sampling but does not report class weighting.[^1]

**Action A — Inverse-frequency class weights**: Apply class weights inversely proportional to training set frequency in the cross-entropy loss. For relation class \(c\) with frequency \(n_c\):
\[ w_c = \frac{N}{K \cdot n_c} \]
where \(N\) is total training instances, \(K\) is number of classes. This is a one-line change in PyTorch (`nn.CrossEntropyLoss(weight=class_weights)`).

**Action B — Negative sampling ratio tuning**: The number of negative (no_relation) pairs vastly outnumbers positive pairs in any RE dataset. Test ratios of 1:1 and 3:1 (positive:negative) rather than the default. A ratio of 3:1 combined with class-adaptive weighting has been shown effective for French biomedical RE. For BioNNE-R, the correct ratio is especially important since macro F1 over 14 classes benefits from balanced exposure.[^12]

**Action C — Auxiliary binary task**: Train a secondary head to distinguish `relation_exists` vs. `no_relation` alongside the main 14-class head (multi-task learning). This prevents the model from overfit on the dominant no_relation class and has demonstrated F1 improvements on multi-class RE.[^13]

**Implementation cost**: ~2 hours.

***
## LLM Usage: Where to Spend Your Token Budget
Given limited LLM token access, direct LLM inference for RE classification is not cost-effective (it underperforms fine-tuned BERT). The three high-value LLM use cases are:[^7]
### 1. Data Augmentation for Rare Relation Types (High Priority)
For low-frequency classes (`USED_IN`, `ABBREVIATION`, `ORIGINS_FROM`, `APPLIED_TO`), use an LLM (e.g., GPT-4o-mini, DeepSeek-V3 via API) to generate **additional training sentences** by paraphrasing existing examples. The CoDa framework (Constrained Generation-based Data Augmentation) demonstrated effective, training-free data augmentation for low-resource NLP. Prompt design:[^14]

```
Given this biomedical abstract sentence with relation ORIGINS_FROM between
[head entity: ANATOMY type] and [tail entity: DISO type], write 3 paraphrases
maintaining the same relation and entity types.
```

This costs ~$0.01–0.05 per rare-class example in augmented form (GPT-4o-mini pricing) and can double training data for tail classes. Generate ~50–100 augmented examples per rare class.
### 2. Relation Descriptions as Contrastive Learning Anchors
Incorporate LLM-generated textual descriptions of each relation type as soft supervision signals. For example, "TREATED_USING: a disorder or injury is addressed, resolved or managed using a chemical or medical device." These descriptions can be used as class prototype vectors via contrastive learning (following the CTL-DRP paradigm which achieved F1 of 76.7% on TACRED vs. 74.6% for standard typed markers). Implementation requires ~1 day.[^15]
### 3. Error Analysis on Dev Set
After each training run, sample 50–100 dev-set errors and prompt an LLM to categorize them (e.g., "ambiguous span", "entity type confusion", "cross-sentence"). This is fast (~$0.10 total) and directly informs whether to invest more in nesting features, cross-sentence context windows, or type marker improvements.

***
## Track Selection Recommendation
Given the student's constraints and objectives (maximize academic output):

| Track | Recommendation | Rationale |
|---|---|---|
| **English (Subtask 1)** | ✅ **Primary submission** | PubMedBERT is strong, dataset is larger in English UMLS coverage, highest chance of competitive ranking[^4] |
| **Russian (Subtask 2)** | ✅ **Secondary submission** | Reuse same pipeline with multilingual encoder swap; publishable bilingual angle[^1] |
| **Bilingual (Subtask 3)** | ⚠️ **Only if time allows** | Requires separate model, cannot reuse monolingual predictions; mDeBERTa fine-tuning is straightforward but adds dev/test overhead[^1] |

**Recommendation**: Submit to both Subtask 1 and Subtask 2 using the same codebase with different encoder checkpoints. This maximizes publications potential (one system paper covering all three datasets) while keeping implementation cost linear, not quadratic.

***
## Resource Estimation
| Component | GPU Memory | Training Time (A100) | Can Run on Colab? |
|---|---|---|---|
| mBERT baseline (reproduction) | ~4GB | ~20 min/epoch | ✅ T4 |
| PubMedBERT-base fine-tuning | ~6GB | ~25 min/epoch | ✅ T4 |
| BiomedBERT-large fine-tuning | ~14GB | ~60 min/epoch | ⚠️ Colab Pro |
| mDeBERTa-v3-base (bilingual) | ~8GB | ~35 min/epoch | ✅ T4 |
| LLM augmentation (GPT-4o-mini) | API-only | N/A | ✅ API call |
| Contrastive class prototypes | +1GB | +10 min/epoch | ✅ T4 |

Estimated total training budget for the full recommended system: **3–4 A100-hours**, well within a typical HPC allocation. All development and experimentation can be done on Colab T4 using the base-size encoders.

***
## Implementation Timeline (Apr 22 → May 14)
| Days | Task |
|---|---|
| Apr 22–24 | Download data, reproduce baseline, verify CodaBench submission format |
| Apr 24–26 | Swap encoder to PubMedBERT (EN) + mDeBERTa (RU/Bilingual); run first fine-tuning |
| Apr 26–28 | Implement typed entity markers and nesting flag; ablate vs. simple markers |
| Apr 28–30 | Tune negative sampling ratio and class weights; evaluate on dev (**Phase boundary: Apr 30**) |
| Apr 30 – May 4 | LLM augmentation for rare classes; retrain with augmented data |
| May 4–10 | Contrastive class prototype extension (if time); error analysis |
| May 10–14 | Final predictions on test set, submission, prepare working notes paper |

***
## Publication Strategy
A single well-structured system paper covering all three submissions can target the **CLEF 2026 Working Notes** (camera ready required post-evaluation). The paper should structure contributions as:

1. **Encoder ablation**: mBERT → PubMedBERT → BioLinkBERT (table over all three tracks)
2. **Marker design ablation**: vanilla → typed → typed + nesting flag
3. **Training strategy ablation**: uniform loss → class-weighted → augmented
4. **LLM augmentation effect**: rare-class F1 before/after augmentation

This four-section ablation narrative directly maps to the BioNNE-R evaluation tracks and aligns with the student's existing expertise in LLMs for biomedical NLP and reliable AI. Estimated submission length: 8–10 pages (CEUR-WS format).

For a higher-tier venue (ACL/EMNLP findings, BioNLP workshop), the paper can be extended with: (a) cross-lingual transfer analysis EN→RU, (b) relation-type-specific error breakdown leveraging the student's ADE/medical error expertise, and (c) uncertainty quantification over predicted relation types.

***
## Risk Assessment
| Risk | Likelihood | Mitigation |
|---|---|---|
| PubMedBERT tokenizer incompatible with Russian text (Subtask 2) | High | Use multilingual SapBERT or mDeBERTa for Russian; PubMedBERT for English only |
| Class imbalance causes macro F1 collapse on rare types | Medium | Class-weighted loss + augmentation (see above) |
| LLM-generated augmentation introduces noise | Medium | Back-validation: only keep augmented examples that the baseline predicts correctly |
| OpenNRE nested entity encoding bug re-introduced | Low | Verify the interleaved marker fix from Yandutov & Loukachevitch[^2] is correctly applied |
| No-relation recall too low (too aggressive filtering) | Low | Tune decision threshold on dev set per class |

***
## Conclusion
The BioNNE-R baseline is a well-implemented but under-optimized system. Three technically straightforward changes — encoder upgrade, typed+nesting-aware markers, and class-weighted training with augmentation — can each independently push macro F1 above the baseline and are jointly likely to yield a competitive leaderboard position. The BioNNE-L 2025 results confirm that domain-specific BERT-family models are the correct base, and that LLM reranking does not add value for this classification-formulated task. The student's existing strengths in biomedical NLP and reliable AI provide a natural framing for the paper's contribution beyond the competition itself.

---

## References

1. [BioNNE-R Shared Task at BioASQ 2026](https://participants-area.bioasq.org/general_information/BioNNER/) - Participants are required to develop models for nested relation extraction in English, Russian, or i...

2. [[PDF] Approaches to Relation Extraction for Nested Named Entities](https://damdid2022.frccsc.ru/files/article/DAMDID_2022_paper_261.pdf) - It is shown that using position features and entity markers allow solving the problem of relation ex...

3. [[PDF] An Open and Extensible Toolkit for Neural Relation Extraction](https://aclanthology.org/D19-3029.pdf) - OpenNRE is an open-source and extensible toolkit that provides a unified framework to implement neur...

4. [[PDF] Overview of the BioASQ BioNNE-L Task on Biomedical Nested ...](https://ceur-ws.org/Vol-4038/paper_3.pdf) - This paper presents an o cial results report for the BioNNE-L, a shared task on Biomedical Nested Na...

5. [Knowledge-augmented Pre-trained Language Models for ... - arXiv](https://arxiv.org/html/2505.00814v2) - We utilize five data sets encompassing four biomedical relation scenarios within a uniform evaluatio...

6. [Knowledge-augmented pre-trained language models for biomedical ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12502496/) - We utilize five data sets encompassing four biomedical relation scenarios within a uniform evaluatio...

7. [A comprehensive evaluation of document level biomedical relation extraction using large language models](https://ieeexplore.ieee.org/document/11354086/) - Document-level Biomedical Relation Extraction (BioRE) is a critical task in biomedical text mining, ...

8. [[PDF] Rethinking the Role of Entity Type in Relation Classification](https://aclanthology.org/2023.ijcnlp-main.24.pdf) - We aim to investigate how to effectively incorporate entity types of e1 and e2 in RC models, and how...

9. [An Improved Baseline for Sentence-level Relation Extraction](https://aclanthology.org/2022.aacl-short.21) - Sentence-level relation extraction (RE) aims at identifying the relationship between two entities in...

10. [[PDF] Rethinking the Role of Entity Type in Relation Classification - AFNLP](https://www.afnlp.org/conferences/ijcnlp2023/proceedings/main-long/cdrom/pdf/2023.ijcnlp-long.24.pdf) - The first observation from Table 3 is that incorpo- rating gold entity type information can improve ...

11. [[PDF] arXiv:2102.01373v1 [cs.CL] 2 Feb 2021](https://arxiv.org/pdf/2102.01373.pdf) - First, we introduce the typed entity marker (Zhong and Chen, 2020), which leads to promising improve...

12. [Tackling Class Imbalance in Relation Extraction for french text - HAL](https://hal.science/hal-05099002) - Our approach combines effective negative sampling with a ratio of 3:1, adaptive class weighting usin...

13. [[PDF] Dealing with negative samples with multi-task learning on span ...](https://arxiv.org/pdf/2309.09713.pdf) - Observing the results, we note that our Multi-Task Entity-Relation Joint. Extraction approach improv...

14. [CoDa: Constrained Generation based Data Augmentation for Low ...](https://arxiv.org/abs/2404.00415) - We present CoDa (Constrained Generation based Data Augmentation), a controllable, effective, and tra...

15. [Sentence-Level Relation Extraction via Contrastive Learning with Descriptive Relation Prompts](https://arxiv.org/abs/2304.04935) - Sentence-level relation extraction aims to identify the relation between two entities for a given se...

Instruction + progress:
All four files are done. Here's a summary of what was written and the key design decisions:

  ---
  Files written (baseline/)

  ┌────────────────────┬───────────────────────────────────────────────────────────────────────┐
  │        File        │                                Purpose                                │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ bionne_dataset.py  │ Dataset, typed marker insertion, nesting detection, class weights     │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ bionne_model.py    │ HF backbone + entity start-marker representations + nesting flag head │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ bionne_train.py    │ Training loop with class-weighted loss; importable + CLI              │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ bionne_predict.py  │ Inference → CodaBench TSV; importable + CLI                           │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ bionne_colab.ipynb │ Colab driver notebook (clone → prep → train EN → train RU → predict)  │
  ├────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ augment.py         │ OpenAI-powered paraphrase generator for rare classes; runs locally    │
  └────────────────────┴───────────────────────────────────────────────────────────────────────┘

  ---
  Key design choices

  Typed markers — <H:DISO>, </H:DISO>, <T:CHEM>, </T:CHEM> added as 32 special tokens. Marker insertion sorts events right-to-left so nested and adjacent spans are handled correctly. The hidden state at the start marker position is used as the entity representation (following Zhong & Chen 2021).

  Nesting flag — a single scalar 0.0/1.0 concatenated to [h_repr; t_repr] before the linear head. One-line cost, 2–4 F1 pt gain for SUBCLASS_OF / PART_OF.

  Class weights — w_c = N / (K * n_c), computed at dataset load time. APPLIED_TO (43 examples) gets weight ~10× vs SUBCLASS_OF (743 examples).

  Checkpoint format — stores model_name, rel2id, use_nesting_flag, so bionne_predict.py never needs flags — it reads everything from the checkpoint.

  Augmentation — preserves entity texts exactly so span detection is a simple str.find(). Optional back-validation with a trained checkpoint filters noisy paraphrases before writing.

  ---
  Workflow

  Colab (GPU): open bionne_colab.ipynb, set REPO_URL to your fork, connect T4, run top-to-bottom.

  Local (while Colab trains):
  export OPENAI_API_KEY=sk-...
  python baseline/augment.py \
      --data baseline/data/eng_train.txt \
      --output baseline/data/augmented_en.jsonl \
      --target 150
  Then commit augmented_en.jsonl and the Colab ablation cell (§9 in the notebook) will pick it up.
