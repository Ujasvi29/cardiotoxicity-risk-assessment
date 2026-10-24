# Module 10: MLP-Based Clinical Feature Extraction

## Role and research boundary

`ClinicalMLPEncoder` maps preprocessed baseline clinical vectors to a fixed-size representation. It has no classification head and does not implement multimodal fusion. An MLP embedding is a learned representation; its dimensions are not inherently interpretable clinical variables.

The project specification and `configs/data_config.json` define CTRCD as the target and `time` as follow-up duration. The existing preprocessing schema defines 9 continuous and 16 binary baseline features (25 total). These are the only supported encoder input columns. The preprocessor rejects target, follow-up, identifier-like, and other unsupported fields. The raw CSVs and generated split artifacts were absent during this implementation, so no real clinical records were processed.

## Architecture and defaults

Defaults are implementation choices because the project specification does not state MLP dimensions: input dimension 25, hidden dimensions `(128, 64)`, ReLU activations, dropout 0.1, and output embedding dimension 32. Hidden dimensions, output dimension, activation (`relu`, `gelu`, `tanh`), dropout, device, and optional initialization seed are configurable. CPU is the default; CUDA is conditional on availability. A supplied seed initializes the model reproducibly without consuming or resetting the caller's global random stream.

The encoder accepts finite floating tensors `[batch, 25]`, including batch size one. Its `forward` method preserves gradients. `extract` temporarily uses evaluation and inference mode, then restores the prior train/eval state.

## Clinical preprocessing and schema

`ClinicalFeaturePreprocessor` reuses the project's `build_clinical_preprocessor`: median imputation and robust scaling for continuous columns, and most-frequent imputation for binary columns. It validates the exact supported schema, numeric dtypes, and binary domains, then returns columns in the stable `FEATURE_COLS` order regardless of input order. Fit it on the training partition only, then call `transform` on validation, test, and inference partitions. The component cannot determine a caller's partition provenance; the caller must split before fitting. Its schema metadata records columns, scaling, imputers, and this fit-scope requirement.

The existing baseline schema is relied upon to identify the 25 permitted inputs. No raw schema audit was possible because the original dataset is missing. If data owners establish that any listed feature is post-baseline or outcome-derived, the schema must be revised before use. No ambiguous clinical meaning was inferred.

## Training, extraction, and export

No training objective is specified in the available project documentation. No objective was invented, and no model training or embedding extraction from real or synthetic clinical records was performed. The reusable encoder supports future gradient-based training. No embedding persistence pipeline was added because the repository specification does not require clinical embedding export at this stage. IDs and labels are therefore not part of any embedding export or matrix.

## Files added

- `src/models/clinical_encoder.py`: validated MLP encoder and fitted clinical preprocessing wrapper.
- `tests/test_clinical_encoder.py`: offline tests with explicitly synthetic fixtures.
- `docs/module_10_report.md`: this report.

No raw data or unrelated documentation was changed. README changes were unnecessary.

## Verification and limitations

Executed using `.venv\\Scripts\\python.exe`:

- `-m pytest -q tests/test_clinical_encoder.py` — **22 passed**.
- Relevant regression modules (Module 9, path handling, preprocessing, splitting, dataset preparation, image preprocessing, and mapping) — **52 passed, 53 skipped**. The skips are pre-existing integration checks requiring missing raw CSVs, image files, or generated processed/split/mapping artifacts.
- `-m pytest -q` — **52 passed, 53 skipped, 0 failed**.
- `-m compileall -q src tests` — passed.
- `git --no-pager diff --check` — passed.

The full-suite skips are documented in the pre-push audit: the original clinical dataset, images, and generated integration artifacts are not present in this checkout. The new Module 10 tests are not skipped and use only clearly labelled synthetic fixtures. Unit tests do not establish performance or suitability on the real cohort. CUDA tests run conditionally and CUDA was unavailable. The existing project notes that patient-level independence is not verified because clinical identifiers are anonymized; this module does not alter or resolve that limitation.

Prerequisites for real-data use are to obtain the original dataset, inspect its columns and baseline timing with the study owner, create partitions using the project's split protocol, and fit a separate preprocessor on training features only. A documented representation-learning or supervised objective and training protocol are needed before claiming a learned clinical representation from the cohort.
