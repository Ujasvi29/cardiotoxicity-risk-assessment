# Module 12: Multimodal CTRCD Classification Component

## Scope and verified target schema

`MultimodalCTRCDClassifier` adds a compact classification head to the existing Module 11 attention-gated fusion component. It accepts paired image and clinical embeddings, reuses Module 11 unchanged, and returns a two-logit representation for CTRCD classes 0 and 1. It does not include training, a loss/optimizer loop, dataset loader, metrics, checkpointing, or clinical validation.

The repository schema (`src/data/preprocessing.py`, `configs/data_config.json`) names `CTRCD` as a binary target with domain `{0,1}` and describes it as CTRCD yes/no. The baseline schema excludes both `CTRCD` and `time`; the latter is follow-up duration and is not a predictor. This supports the two-class output order `(0, 1)`, but the raw data are absent, so the cohort's label coding and prediction-time validity were not independently checked. No prediction horizon is specified in the repository.

## Model interfaces

The default classifier composes:

1. Module 9 image embeddings: 1,280 features by default (Module 9 can be configured to project to another size).
2. Module 10 clinical embeddings: 32 features by default, after the existing training-fitted clinical preprocessing interface.
3. The existing Module 11 `AttentionGatedFusion`, whose default fused output has 128 features.
4. A classification head `Linear(128, 64)`, ReLU, dropout 0.1, then `Linear(64, 2)`.

The hidden dimensions, dropout, and initialization seed are configurable. The two output classes are fixed to the verified binary schema. The model takes embeddings rather than raw images or clinical tables; the caller runs the existing encoders and must supply verified matching record order. It does not infer or create pairings. The classifier can be trained through gradients along with the trainable fusion module and input embeddings; this report does not claim an integrated end-to-end raw-modality training pipeline. Module 9's default backbone is frozen unless configured otherwise.

`forward(image_embeddings, clinical_embeddings)` returns logits `[batch, 2]` in class order `(0, 1)` for training. `predict_proba` switches to evaluation/inference mode, returns softmax probabilities in the same class order, and restores the previous train/eval mode. `predict` returns the argmax class in `{0,1}`; no validation-selected threshold is introduced. `return_diagnostics=True` returns logits plus Module 11's attention weights and gate values. Those values are model diagnostics, not causal explanations or definitive clinical importance.

## Data pairing, training, and evaluation status

The raw dataset, image files, generated split artifacts, and verified image-to-clinical mapping are absent in this checkout. Existing split manifests document that patient identifiers are anonymized and patient-level independence is not verified. Consequently, there is no safe way to construct paired multimodal training examples or evaluate on a held-out test set. This module creates no dataset, pairing, labels, or experimental results. Training, early stopping, class weighting, checkpointing, metrics, and performance claims remain unimplemented until provenance-backed pairing and appropriate split information are available. Synthetic embeddings are used only in unit tests.

## Changed files

- `src/models/multimodal_classifier.py`: binary logits head, inference probabilities/labels, fusion diagnostics, validated configuration.
- `tests/test_multimodal_classifier.py`: deterministic offline tests using synthetic embeddings.
- `docs/module_12_report.md`: this report.

No existing encoders, fusion logic, preprocessing, data files, or dependencies were changed.

## Verification results

Executed using `.venv\\Scripts\\python.exe`:

- `-m pytest -q tests/test_multimodal_classifier.py` — **22 passed**.
- `-m pytest -q` — **110 passed, 53 skipped, 0 failed**. The skips are existing integrations requiring original CSVs, image data, or generated processed/split/mapping artifacts absent from this checkout. No Module 12 test was skipped.
- `-m py_compile src/models/multimodal_classifier.py tests/test_multimodal_classifier.py` — passed.
- `-m compileall -q src tests` — passed.
- `git diff --check` — passed.

CUDA was unavailable in this environment. No real-data pairing, training, or held-out evaluation was performed.
