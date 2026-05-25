# BioNNE-R Relation Extraction Recovery Plan

## 1. Problem Diagnosis (Recovered May 10, 2026)
We identified a severe Macro F1 collapse on the Subtask 1 (English) test set. The model was predicting ~40,000 relations when only ~9,000 were expected.

### Root Causes
1. **Critical Truncation Bug:** 51% of test instances > 256 tokens. Markers were being cut off, causing the model to represent entities using the [CLS] token and hallucinate high-confidence garbage.
2. **Distribution Mismatch:** Training ratio (1:3) vs. Test density (~1:100). Inverse-frequency weighting (0.08 for no_relation) heavily biased the model toward false positives.
3. **Invalid Triplet Hallucination:** 25% of predictions were (HeadType, TailType, Relation) combinations that never occurred in training.

---

## 2. Approved Strategy
The following decisions were confirmed by the user:
* **Truncation Handling:** Increase MAX_LEN to 512. If markers are still missing, forcefully predict no_relation.
* **Calibration Strategy:** Use Per-Class Thresholds optimized on the Blind Dev set.
* **Type Pruning:** Implement strict pre-filtering to drop triplets never seen in training.
* **Training:** Increase NEG_RATIO to 10-15 and clamp no_relation weight to 1.0.

---

## 3. Implementation Plan

### Phase 1: Surgical Edits (The "Failsafe")
* **bionne_dataset.py**: Update BioNNEDataset to detect missing markers and return a valid_markers mask.
* **bionne_predict.py**: Modify _probs_to_df to respect the valid_markers mask and forcefully override invalid markers to no_relation.

### Phase 2: Type Pruning Logic
* **prepare_data.py** (or new script): Extract all valid (head_type, tail_type, relation) triplets from train TSVs into data/valid_triplets.json.
* **bionne_predict.py**: Update to load valid_triplets.json and prune impossible predictions.

### Phase 3: Calibration & Retraining
* **bionne_colab.ipynb**: Update training params (MAX_LEN=512, NEG_RATIO=15).
* **bionne_predict.py**: Implement per-class threshold optimization sweep on eng_dev_blind.txt.

---

## 4. Current Task Status
* **Status:** Resuming Execution Phase.
* **Next Step:** Verify existing code in baseline/bionne_dataset.py for any partial implementations of the truncation failsafe.
