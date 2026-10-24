# Module 13: Multimodal CTRCD Training Infrastructure

## Scope and data readiness

Module 13 consumes caller-supplied, already paired image and clinical embeddings. It does not load data, infer image-to-record mappings, or verify the provenance of an external mapping. The checkout has no supported paired-feature loader or verified split/mapping provenance, so this report makes no real-cohort readiness or evaluation claim.

`PairedEmbeddingDataset` checks tensor ranks, row counts, positive feature dimensions, finite floating-point embeddings, binary `torch.long` labels, non-empty opaque IDs, exact image/clinical/label record-ID order, and within-partition record-ID uniqueness. Train, validation, and test partitions are checked for record and patient-group overlap. These are structural checks only: identical IDs and disjoint groups do not prove that the IDs correctly represent real patients or that modality records were paired correctly. Provenance must be verified by the data owner before constructing these datasets.

## Training and selection behavior

`MultimodalTrainer.fit` trains the Module 12 classifier from paired embeddings with cross-entropy and AdamW. Validation must contain both classes, and validation ROC-AUC controls early stopping and best-epoch restoration. Metrics use class order `(0, 1)` and a confusion matrix with explicit labels. Specificity is represented as `null` when its negative-class denominator is zero; ROC-AUC and average precision are undefined for a one-class partition and such partitions are rejected before metric calculation.

`MultimodalHyperparameterSearch.search` accepts only training and validation datasets. It chooses the best trial using validation ROC-AUC. Expected per-trial rejections can be signaled with `TrialFailure` and are recorded; unexpected exceptions are recorded in `trials.json` when an output directory is supplied and then re-raised. Failed trials are never eligible for selection. Test isolation is an experiment-workflow requirement: do not inspect test metrics, predictions, or outcomes while tuning, selecting a checkpoint, choosing a final model, or making any other model-selection decision. After all choices are locked, evaluate once on an untouched test partition. The API cannot enforce how callers use results obtained from a separate test evaluation.

The tuner does not select or return a final clinical model. Its selected trial is a validation-selected candidate. A caller must make a final training/model-selection decision using training and validation data alone before test evaluation. No training or model-selection operation is implemented here for the purpose of clinical deployment.

## Reproducibility and checkpoints

Training uses seeded data-loader generators and isolates PyTorch CPU and selected CUDA RNG streams with `torch.random.fork_rng`; CUDA behavior depends on an available supported CUDA device and has not been exercised on this CPU-only verification run. Model initialization is supplied by the existing model constructors. JSON reports and tensor checkpoints are written through temporary files and atomically replaced. Checkpoint loading requests `weights_only=True`, validates recorded class order and model configuration, and strictly loads the state dictionary. This is not a substitute for verifying the checkpoint's source or intended experiment.

No CLI, data-loader interface, dataset file, pairing, label, or experiment result is introduced by this module.

## Evidence

The Module 13 unit tests use synthetic tensors and synthetic opaque record/group IDs. They provide software-level evidence for validation, training mechanics, split structural checks, checkpoint round trips, metric conventions, and trial handling only. They are not cohort evidence. No real-data training, tuning, held-out evaluation, mapping-provenance verification, or clinical validation was performed.

## Verification

Verification used `.venv\\Scripts\\python.exe` where pytest was required:

- `python -m py_compile src/training/multimodal_trainer.py` — passed.
- `python -m pytest -q tests/test_multimodal_trainer.py` — **21 passed**.
- `python -m pytest -q` — **131 passed, 53 skipped, 0 failed**. Existing skipped integrations require data artifacts absent from this checkout; Module 13 tests did not skip.
- `python -m compileall -q src tests` — passed.
- `git diff --check` — passed. New, untracked files were additionally checked for trailing whitespace because Git's regular diff check does not include untracked files.
