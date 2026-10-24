# Pre-push audit

## Findings before changes

The checkout was on `main`. It already contained the uncommitted Module 9 changes; those were retained. The original dataset, generated `data/processed`, `data/splits`, and mapping outputs were absent from this checkout. A search under the workspace did not find the clinical CSV.

The prior full-suite result of **53 failed, 19 passed** reproduced. The 53 failures came from the legacy data-dependent tests: dataset preparation (16), image preprocessing (5), mapping (12), preprocessing (7), and splitting (13). The failures were caused by references to the old `C:\\dev\\cardiotoxicity risk assesment` location and by unavailable raw CSVs, images, and generated integration artifacts. The raw-file checks used assertions such as `exists() and size == expected` with the message “Clinical CSV modified!”; the CSV did not exist at the checked path, so this result did not establish that a file had been changed. No expected file sizes or raw data were altered.

Some old tests passed vacuously because they iterated over empty artifact/image globs. Those checks are now explicitly marked as data-dependent and skip until their artifacts exist. Leakage and split behavior that can be tested without the clinical dataset now use deterministic synthetic fixtures. No data-dependent assertion was weakened.

No dependency error caused the 53 failures. This run used Python 3.10.11 in `.venv`; setup documentation targets Python 3.13/14. PyTorch and pytest were installed in the ignored virtual environment for the Module 9 tests.

## Changes

- Added `src/utils/paths.py`: repository-root discovery, `CARDIOTOXICITY_DATA_ROOT` support (absolute paths and paths relative to the repository), and clear missing-input errors.
- Replaced machine-specific roots in exploration, quality, preprocessing, splitting, image preprocessing, and the corresponding tests.
- Deferred output-directory creation until the relevant pipeline or report-writing function runs. Importing modules no longer creates files or directories.
- Marked integrations requiring raw CSVs, images, processed outputs, split artifacts, or mapping reports. The test prerequisite fixture reports missing paths and how to configure the external data root.
- Kept data-independent leakage, CPU preprocessing, split determinism, ROI bounds, path resolution, and Module 9 tests runnable without the original dataset.
- Audited Module 9 and added coverage for explicit pretrained fallback, non-network initialization errors, normalized inputs, grayscale/RGBA inputs, deterministic identity ordering, and overwrite refusal.
- Updated setup/README guidance and ignored `.pytest_cache`.

## Validation results

Executed with `.venv\\Scripts\\python.exe`:

- `-m pytest tests/test_efficientnet_encoder.py -q` — **17 passed**.
- `-m pytest -q -m "not requires_raw_data and not requires_image_data and not requires_processed_data and not requires_split_data and not requires_mapping_data"` — **30 passed, 53 deselected**.
- `-m pytest -q` — **30 passed, 53 skipped, 0 failed**. Skips require the unavailable raw CSVs, images, or preprocessing/split/mapping outputs. Pytest reports each marker and missing path.
- `-m compileall -q src tests` — passed.
- `-m src.data.image_features --help` — passed.
- `git diff --check` — passed; Git emitted only line-ending conversion notices.
- Ran `tests/test_paths.py` from a temporary working directory — **6 passed**, confirming path resolution does not depend on caller cwd.

Pretrained weights were not downloaded. CUDA was unavailable and was not tested. No raw datasets, model weights, or feature exports were generated.

## Remaining prerequisites

Set `CARDIOTOXICITY_DATA_ROOT` to the directory containing the original CSV files and optional `IMAGES/` directory. Run the data-generation steps from `setup_instructions.md`, then run the full suite again to execute the skipped integration tests. A verified image-to-record mapping is still absent; image-only features must not be joined to clinical records or treated as labeled CTRCD examples.

The repository is left uncommitted for review. No push was performed.
