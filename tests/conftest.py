"""Shared explicit prerequisites for data-dependent integration tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import BINARY_COLS, CONTINUOUS_COLS, FEATURE_COLS, TARGET_COL, TIME_COL
from src.utils.paths import DATA_ROOT_ENV, PROJECT_ROOT, RAW_DATA_FILES, get_data_root


def _prerequisites(marker_name: str) -> list[Path]:
    data_root = get_data_root()
    if marker_name == "requires_raw_data":
        return [data_root / name for name in RAW_DATA_FILES]
    if marker_name == "requires_image_data":
        image_dir = data_root / "IMAGES"
        return [image_dir] if image_dir.is_dir() and any(image_dir.glob("*.png")) else [image_dir / "<at least one PNG image>"]
    if marker_name == "requires_processed_data":
        return [PROJECT_ROOT / "data/processed" / name for name in (
            "feature_schema.json", "clinical_features_imputed_full.csv",
            "clinical_features_scaled_full.csv", "clinical_targets_full.csv",
            "multimodal_clinical_imputed.csv", "multimodal_clinical_scaled.csv",
            "multimodal_targets.csv", "functional_waveforms_raw_cm_s.csv",
            "functional_waveforms_scaled.csv",
        )]
    if marker_name == "requires_split_data":
        paths = [
            PROJECT_ROOT / "data/splits/cohort_a_clinical_splits.json",
            PROJECT_ROOT / "data/splits/cohort_b_multimodal_splits.json",
            PROJECT_ROOT / "data/processed/feature_schema.json",
        ]
        paths.extend(PROJECT_ROOT / "data/processed/splits" / f"cohort_{cohort}_{split}_{kind}.csv"
                     for cohort in ("a", "b") for split in ("train", "val", "test")
                     for kind in (("features", "targets") if cohort == "a" else
                                  ("features", "targets", "waveforms_raw", "waveforms_scaled")))
        return paths
    if marker_name == "requires_mapping_data":
        return [
            *(data_root / name for name in RAW_DATA_FILES),
            PROJECT_ROOT / "data/processed/mapping/record_linkage_audit.csv",
            PROJECT_ROOT / "data/processed/mapping/modality_availability_summary.csv",
            PROJECT_ROOT / "data/processed/mapping/unresolved_mappings.csv",
        ]
    return []


@pytest.fixture(autouse=True)
def skip_integration_test_when_prerequisite_is_missing(request):
    for marker_name in (
        "requires_raw_data", "requires_image_data", "requires_processed_data",
        "requires_split_data", "requires_mapping_data",
    ):
        if request.node.get_closest_marker(marker_name):
            missing = [path for path in _prerequisites(marker_name) if not path.exists()]
            if missing:
                formatted = "\n".join(f"  - {path}" for path in missing[:5])
                extra = f"\n  - and {len(missing) - 5} more" if len(missing) > 5 else ""
                pytest.skip(
                    f"{marker_name} prerequisite unavailable:\n{formatted}{extra}\n"
                    f"Set {DATA_ROOT_ENV} to the original dataset directory and generate the documented outputs first."
                )


@pytest.fixture
def synthetic_clinical_frame():
    """Small deterministic clinical-shaped frame for data-independent unit tests."""
    rng = np.random.default_rng(321)
    count = 80
    frame = pd.DataFrame({column: rng.normal(size=count) for column in CONTINUOUS_COLS})
    frame.loc[0, CONTINUOUS_COLS[0]] = np.nan
    for index, column in enumerate(BINARY_COLS):
        values = np.resize(np.asarray([0.0, 1.0, np.nan]), count)
        frame[column] = np.roll(values, index)
    frame[TARGET_COL] = np.resize([0, 1], count)
    frame[TIME_COL] = rng.integers(1, 1000, size=count)
    return frame[FEATURE_COLS + [TARGET_COL, TIME_COL]]
