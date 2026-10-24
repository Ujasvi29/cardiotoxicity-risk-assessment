# Module 14: Class Imbalance Handling for Multimodal CTRCD Classification

## Audit findings and evidence boundaries

The existing Module 12 classifier returns two logits in order `(0, 1)`. Its source defines `CTRCD_CLASS_LABELS = (0, 1)`, and the dataset exploration notebook states `0` is CTRCD-negative and `1` is CTRCD-positive. The training labels and raw CSVs are not available here to independently check their contents. Module 13 used unweighted `torch.nn.CrossEntropyLoss` for training and validation, and it accepts already-paired embeddings through `PairedEmbeddingDataset`; it has no loader for the project CSVs or image features.

The notebook discusses row alignment between functional and combined CSVs, but neither the dataset artifacts nor a provenance-backed patient mapping are available in this checkout. The repository configuration and split-generation code state patient-level independence is unverified and manifests use `patient_id: null`. Module 13's identifier overlap checks are structural and depend on caller-provided identifiers; they cannot establish mapping provenance or patient identity. Consequently, no real-data training or held-out cohort evaluation was attempted.

## Strategy and formula

Weighted cross-entropy is the only balancing strategy added. It preserves paired records and changes the loss contribution without synthesizing, duplicating, or rearranging multimodal examples. For class `c`, with `N` training labels and training count `n_c`, the raw inverse-frequency weight is:

```text
w_c = N / (K * n_c)
```

where `K=2`; weights are ordered to match the classifier logits `(0, 1)`. Their average across training examples, `sum_c(n_c * w_c) / N`, is 1. Weights are computed once from the training partition only, converted to the model loss device and dtype, and reused for minibatches. Both classes must occur in training. Empty labels, invalid class IDs, unsupported tensor shapes/types, and single-class training labels fail explicitly. Default `class_weighting=False` preserves the existing unweighted training behavior. Validation and evaluation losses remain unweighted so they keep their prior interpretation.

No resampling, SMOTE, automatic threshold selection, or probability calibration was introduced. Class weighting can change probability calibration; calibration must be assessed independently before clinical interpretation. The classifier currently predicts with argmax, and no threshold-selection interface exists. A future threshold decision must use validation data only and must not be described as clinically validated on that basis.

## Integration and metrics

Use the existing `MultimodalTrainer` with the opt-in setting:

```python
from src.training.multimodal_trainer import MultimodalTrainer, TrainingConfig

trainer = MultimodalTrainer(TrainingConfig(class_weighting=True))
result = trainer.fit(model, train_data, validation_data)
```

The trainer derives weights from `train_data` labels; it does not accept validation or test labels in the class-weight function. The validation-only hyperparameter search also supports `TuningConfig(class_weighting=True)`. Test data are not a search argument. The separate `evaluate` method checks structural record/group disjointness and reports metrics, but callers must keep test results out of all model, checkpoint, hyperparameter, and threshold decisions.

Metrics use class order `(0, 1)` and explicitly identify class 1 as CTRCD-positive. Reports include class counts/proportions, confusion matrix, accuracy, positive-class precision/recall/F1, specificity, NPV, balanced accuracy, ROC-AUC, average precision, log loss, and the two-class Brier score when valid probabilities are supplied. ROC-AUC and balanced accuracy are `null` when either observed class is absent; positive-class recall and average precision are `null` when no positives are present; specificity is `null` with no negatives. Precision, NPV, and F1 are `null` when their denominators are zero. Other defined metrics remain reportable for one-class evaluation. Probability arrays are checked for shape, finiteness, range, and row normalization.

Train/validation/test record and patient-group overlaps are checked by the existing Module 13 API. These checks only establish structural disjointness of supplied opaque keys. They do not verify that caller-provided IDs or image-clinical mappings reflect real patients. No balancing is applied before splitting; weights are calculated after the caller has supplied the training partition.

## Verification evidence

Module 14's tests use synthetic in-memory tensors only. They verify the formula, normalized values for balanced/moderately/severely imbalanced counts, deterministic results, invalid inputs, one-class failure, loss backpropagation, trainer integration, training-only calculation, metric edge cases, and CUDA behavior only if CUDA is available.

Evidence categories:

1. **Structural validation:** implemented for shapes, class IDs, probabilities, and existing supplied record/group identifiers; real mapping provenance is not verified.
2. **Synthetic unit-test evidence:** reported by the commands below; not cohort evidence.
3. **Real-data training evidence:** none; dataset and paired-feature artifacts are unavailable.
4. **Held-out evaluation evidence:** none.
5. **Clinical validation evidence:** none.

CUDA was unavailable; the conditional CUDA test skipped. `null` metric values represent quantities undefined for the observed labels or predictions, not measured zeros. Class weighting is an optimization technique, not evidence of improved performance or clinical utility.

## Verification results

Executed with `.venv\\Scripts\\python.exe`:

- `-m pytest -q tests\\test_class_imbalance.py` — **14 passed, 1 skipped** (CUDA unavailable).
- Imbalance plus Module 12/13 regression tests — **93 passed, 1 skipped** (CUDA unavailable).
- `-m pytest -q` — **145 passed, 54 skipped, 0 failed**. One skip is the CUDA-only Module 14 test; the remaining skips are existing integrations requiring unavailable data artifacts.
- `-m py_compile` on the new imbalance module and touched training/test modules — passed.
- `-m compileall -q src tests` — passed.
- `git diff --check` — passed (Git printed line-ending normalization warnings for existing tracked source/test files). An explicit trailing-whitespace scan including untracked Module 14 files found no issues.

No real-data training or held-out evaluation was performed.
