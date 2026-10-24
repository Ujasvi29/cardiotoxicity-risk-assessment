# Module 9: EfficientNet-B0 Image Feature Extraction

## Objective and integration

Module 9 exports image vectors independently of clinical records, targets, classification, and multimodal fusion. It reuses `src.data.image_preprocessing.preprocess_single_image` for RGB conversion, the existing TDI spectral ROI crop, 224×224 bilinear resize, and `[0, 1]` pixel scaling. The module then applies ImageNet channel mean/std normalization. This preserves the established TDI crop while adding the normalization expected by pretrained ImageNet weights; no second crop/resize pipeline was introduced.

The existing image preprocessor previously used a machine-specific root path, created output directories on import, and omitted the `Optional` import used by its annotations. The minimal repair makes the root repository-relative, delays directory creation until the audit writes outputs, and imports the annotation. Existing ROI behaviour is unchanged.

## Model and configuration

- Architecture: Torchvision EfficientNet-B0; original classifier replaced with `nn.Identity`, so the forward output is the pooled pre-classifier vector.
- Dimension: 1,280, confirmed against Torchvision's `classifier[1].in_features` at construction. Optional `projection_dim` adds a linear projection; raw 1,280 features are the default.
- Weights: default is random initialization and requires no download. `pretrained=True` requests `EfficientNet_B0_Weights.DEFAULT`; expected network/cache errors are raised unless `allow_untrained_fallback=True`. A fallback emits a warning and records `random_fallback_after_weight_error`; all outputs carry the actual status.
- Device: explicit `cpu` by default or `cuda` when available. Requesting unavailable CUDA raises a clear error.
- Trainability: `trainable_blocks=0` freezes the entire backbone by default. Values 1–9 unfreeze that many final sequential entries of `backbone.features`; optional projection parameters remain trainable. `forward` preserves gradients for future fine-tuning. `extract` temporarily enters eval mode and inference mode and restores the caller's previous training flag.
- Input: float tensor `[batch, 3, height, width]`, nonempty and finite. Batch size one is supported.
- Export: `python -m src.data.image_features` processes supported image files in filename order and writes `image_features.npy` and `image_features_manifest.json`; existing output files are never overwritten. The manifest records source paths and filenames, feature shape/dimension, preprocessing settings, and weight status. It carries no clinical labels or patient mapping.

## Existing components and data audit

The repository contains Modules 4–8 preprocessing, split and dataset preparation code, and tests for those components; there is no prior image model factory. `configs/data_config.json` documents the cohort counts, but raw `dataset/`, generated `data/` mapping reports, and image files are absent in this checkout. The mapping test specifies functional-to-image links as unverified and explicitly rejects inference from filename order. No actual images/counts or mapping artifacts could be audited here. No mapping, labels, or supervised image training were created.

## Files created or modified

- `src/models/efficientnet_encoder.py`: configurable encoder, weight status, device checks, input/output validation, freezing/fine-tuning, inference helper.
- `src/data/image_features.py`: existing-pipeline adapter, ImageNet normalization, batch extraction, identity manifest, no-overwrite export, CLI.
- `src/data/image_preprocessing.py`: portable root, import-safe output directory creation, missing typing import.
- `tests/test_efficientnet_encoder.py`: offline model, shape, trainability, preprocessing and export tests.
- `requirements.txt`: Torch and Torchvision runtime dependencies.
- `docs/module_9_report.md`: this report.

## Run instructions

From the repository root, install the requirements in the project's selected Python environment. For a CPU-only Windows environment, replace the Torch packages from the documented PyTorch CPU wheel index:

```powershell
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m src.data.image_features dataset/IMAGES outputs/module_9_features --device cpu --batch-size 16
```

The command above uses random weights. To request ImageNet weights, add `--pretrained`; if they are unavailable, extraction stops. Add `--allow-untrained-fallback` only when random-weight fallback is acceptable and clearly recorded. Use a fresh output directory. The CLI sorts filenames for deterministic ordering, but ordering does not define clinical correspondence.

## Verification and unresolved blockers

The selected interpreter was Python 3.10.11 (setup documentation targets Python 3.13/14). Dependencies were installed into the gitignored `.venv`: Torch 2.14.1+cpu, Torchvision 0.29.1+cpu, and pytest 9.1.1. `.venv\\Scripts\\python.exe -m pytest tests/test_efficientnet_encoder.py -q` completed with **17 passed**. No pretrained weights were downloaded. CUDA was unavailable.

The full suite completed with **30 passed, 53 skipped, 0 failed**. The skips are explicitly marked integration tests requiring unavailable raw CSVs, TDI images, or generated preprocessing/split/mapping artifacts. The filtered data-independent run reported **30 passed, 53 deselected**. These results do not verify the original dataset or its generated reports.

## Acceptance status

| Criterion | Status | Evidence / limitation |
|---|---|---|
| Repository structure inspected before changes | PASS | Existing source, tests, config, setup docs, dataset availability, and initial Git state inspected. |
| Modules 1–8 preserved | PASS | Data logic was retained; machine-specific paths and import-time directory creation were corrected. |
| EfficientNet-B0 integrated and classifier bypassed | PASS | Torchvision B0 with identity classifier. |
| Feature vector and actual dimension | PASS | Runtime tests verified the actual output shape is 1,280. |
| Explicit pretrained status and offline construction | PASS | Random default tested offline; explicit requested weights and opt-in fallback. |
| CPU and conditional CUDA | PASS | CPU inference tested; CUDA is guarded by runtime availability checks. |
| Frozen and configurable tuning | PASS | Frozen default; configurable final feature blocks tested. |
| Existing preprocessing reused | PASS | Existing RGB/ROI/resize/scaling function reused; normalization applied once after it. |
| Input/output validation | PASS | Implemented and exercised by the new tests. |
| Module 9 tests added and executed | PASS | 17 tests passed on CPU without pretrained downloads. |
| Regression tests executed | PASS | 30 runnable tests passed; 53 data-dependent integration tests were explicitly skipped. |
| Feature export and manifest | PASS | NPY plus JSON manifest and overwrite refusal tested. |
| No unsupported image mapping or supervised training | PASS | No mapping was constructed or supervised image model trained. |
| Documentation updated | PASS | This report contains execution and limitation details. |
| Git/final diff audited | PASS | Final branch/status/diff checked; `.venv` is ignored. |

## Module 10 prerequisites

The encoder can be consumed downstream for unlabeled image-only analysis. Supervised image training or multimodal fusion is not ready: obtain and validate a provenance-backed image-to-record mapping, then run the skipped integration suite against the original dataset and generated artifacts. The current source does not establish clinical validity or predictive performance.
