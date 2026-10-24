# Module 15: Model Validation and Overfitting Analysis

## Purpose and audit

Module 15 analyzes existing Module 13 fit histories and provides a final, held-out checkpoint evaluation wrapper. It does not train, tune, select, or alter a model. The existing trainer records epoch-level `train_loss`, `validation_loss`, `validation_roc_auc`, and `validation_accuracy`, plus the best validation epoch and selection metric. Module 13 already safely loads checkpoints with `weights_only=True`, checks model metadata and strict state-dict compatibility, performs record/patient-group overlap checks, and calculates core metrics. Module 15 reuses those APIs rather than implementing another model or trainer.

Module 12 fixes classifier logit order as `(0, 1)`; the dataset exploration notebook defines label `1` as CTRCD-positive. The notebook references clinical, functional, combined CSVs and TDI images, but these dataset artifacts are absent in the current checkout. Config and split-generation code explicitly state patient-level independence is unverified; generated manifests use null patient IDs. Supplied opaque ID checks can detect structural overlap only and cannot establish patient identity, split provenance, or image-clinical mapping correctness. No real-data evaluation was performed.

## History analysis

`analyze_training_result` accepts the result mapping from `MultimodalTrainer.fit`; `analyze_training_history` accepts the recognized history series and optional training metadata. Missing, empty, non-finite, non-numeric, or inconsistent series are reported as insufficient, invalid, or inconsistent evidence. No epoch values or metrics are filled in. Trends report start/end/change/range and direction changes. Best epoch and the validation selection metric are reported from supplied metadata only.

A possible overfitting pattern requires all three of these observed trends over the configured recent window (default three epochs): training loss falls, validation loss rises, and a recorded validation metric (ROC-AUC preferred, otherwise accuracy) does not improve. This is a heuristic warning, not a diagnosis. Loss gap is reported only when metadata explicitly records `class_weighting=False`; Module 14 uses weighted training loss and unweighted validation loss, which are not directly comparable. Missing class-weighting metadata also leaves the loss gap unavailable and the loss-divergence conclusion inconclusive. Flat or jointly worsening losses are described as a possible optimization stall or underfitting ambiguity, not a definitive diagnosis. Analysis has no test-set input and has no training side effects.

## Held-out checkpoint evaluation

`evaluate_selected_checkpoint(checkpoint_path, test_data, train_data=..., validation_data=...)` delegates safe loading to `load_checkpoint` and evaluation/split checks to `MultimodalTrainer.evaluate`. The optional `expected_model` checks architecture compatibility (class labels, fusion dimensions, classifier widths, and dropout). Evaluation reports SHA-256 of checkpoint bytes, not the filesystem path. It returns class counts/proportions, confusion matrix, class-wise precision/recall/F1, accuracy, sensitivity, specificity, NPV, ROC-AUC, average precision, balanced accuracy, log loss, and Brier score where defined. Class 1 is explicitly labeled CTRCD-positive; probabilities from the existing evaluator are used for ranking and Brier metrics. Undefined values remain `null`.

The wrapper checks train/validation/test record and patient-group separation through Module 13. It reports those as structural checks and always marks patient identity, pairing provenance, and patient-level independence unverified because those facts are not encoded in the current dataset interface. Keep test results out of checkpoint, hyperparameter, threshold, or model selection; the API cannot enforce downstream experiment workflow. The checkpoint wrapper does not return or modify the loaded model. No threshold selection or calibration is performed, and Brier score alone is not a calibration assessment.

## Files added

- `src/evaluation/validation_analysis.py`
- `tests/test_validation_analysis.py`
- `docs/module_15_report.md`
- `src/evaluation/__init__.py` exports the Module 15 public helpers.

Modules 9–12, Module 14, dataset/notebook artifacts, and dependencies were not changed. No dataset or patient information is written to reports.

## Evidence and limitations

1. **Verified implementation behavior:** synthetic tests cover history parsing/trends, weighted-loss comparability, safe checkpoint loading through the existing API, expected-architecture checks, single-class held-out metrics, structural split checks, and parameter/mode stability of evaluation.
2. **Synthetic unit-test evidence:** tests use small generated tensors and labels only; their metrics are not research results.
3. **Real-data findings:** none. No source data, generated feature artifacts, or supported paired-data loader were found in the checkout.
4. **Unverified:** patient-level split independence, opaque-ID provenance, and image-clinical correspondence. CUDA availability was not required by these tests; Module 15 tests are CPU-only.
5. **Clinical validation:** none. Retrospective classification metrics and heuristic history diagnostics do not establish clinical usefulness or generalization.

## Verification results

The pre-change full test suite passed with **145 passed, 54 skipped**. After implementation, executed using `.venv\\Scripts\\python.exe`:

- `-m pytest -q tests\\test_validation_analysis.py` — **29 passed**.
- Module 15 plus Modules 12–14 regression tests — **122 passed, 1 skipped** (the conditional CUDA test).
- `-m pytest -q` — **174 passed, 54 skipped, 0 failed**. The existing integration skips require unavailable data artifacts; the one additional skip is the CUDA-only Module 14 test.
- `-m py_compile` for Module 15 and its touched integration modules/tests — passed.
- `-m compileall -q src tests` — passed.
- `git diff --check` — passed; Git printed a line-ending normalization warning for `src/evaluation/__init__.py`.

No real-data training, tuning, held-out evaluation, or clinical validation was performed.
