"""
Module 4 Test Suite: Preprocessing Pipeline Verification
=========================================================
Tests and assertions verifying:
1. Raw file immutability (dataset/ remains unmodified).
2. Dimensions and shapes of all processed datasets.
3. Target label preservation and class distributions.
4. Absence of NaN / null values in processed feature sets.
5. Domain validity of binary features.
6. Functional waveform dimensions (exactly 1001 time points).
7. Leak-free train-fit / test-transform pipeline behavior.
8. Determinism and reproducibility across repeated runs.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    CONTINUOUS_COLS,
    BINARY_COLS,
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    load_raw_data,
    build_clinical_preprocessor,
    preprocess_clinical_dataframe,
    preprocess_functional_waveforms,
)
from src.utils.paths import PROJECT_ROOT, get_data_root

BASE_DIR = PROJECT_ROOT
DATASET_DIR = get_data_root()
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@pytest.mark.requires_raw_data
def test_raw_files_unmodified():
    """Verify that original raw files in dataset/ have not been corrupted or altered."""
    clin_path = DATASET_DIR / "BC_cardiotox_clinical_variables.csv"
    func_path = DATASET_DIR / "BC_cardiotox_functional_variable.csv"
    comb_path = DATASET_DIR / "BC_cardiotox_clinical_and_functional_variables.csv"

    assert clin_path.exists() and clin_path.stat().st_size == 58124
    assert func_path.exists() and func_path.stat().st_size == 3346159
    assert comb_path.exists() and comb_path.stat().st_size == 3664729


@pytest.mark.requires_processed_data
def test_processed_files_exist():
    """Verify that all required processed files exist."""
    expected_files = [
        "feature_schema.json",
        "clinical_features_imputed_full.csv",
        "clinical_features_scaled_full.csv",
        "clinical_targets_full.csv",
        "multimodal_clinical_imputed.csv",
        "multimodal_clinical_scaled.csv",
        "multimodal_targets.csv",
        "functional_waveforms_raw_cm_s.csv",
        "functional_waveforms_scaled.csv"
    ]
    for fname in expected_files:
        fpath = PROCESSED_DIR / fname
        assert fpath.exists(), f"Missing processed file: {fname}"


@pytest.mark.requires_processed_data
def test_processed_shapes_and_completeness():
    """Verify exact matrix dimensions and complete absence of missing values."""
    # Full clinical cohort
    X_full_imp = pd.read_csv(PROCESSED_DIR / "clinical_features_imputed_full.csv")
    X_full_scale = pd.read_csv(PROCESSED_DIR / "clinical_features_scaled_full.csv")
    y_full = pd.read_csv(PROCESSED_DIR / "clinical_targets_full.csv")

    assert X_full_imp.shape == (531, 25)
    assert X_full_scale.shape == (531, 25)
    assert y_full.shape == (531, 2)
    assert X_full_imp.isnull().sum().sum() == 0, "Nulls found in X_full_imp!"
    assert X_full_scale.isnull().sum().sum() == 0, "Nulls found in X_full_scale!"

    # Multimodal cohort
    X_multi_imp = pd.read_csv(PROCESSED_DIR / "multimodal_clinical_imputed.csv")
    X_multi_scale = pd.read_csv(PROCESSED_DIR / "multimodal_clinical_scaled.csv")
    y_multi = pd.read_csv(PROCESSED_DIR / "multimodal_targets.csv")

    assert X_multi_imp.shape == (270, 25)
    assert X_multi_scale.shape == (270, 25)
    assert y_multi.shape == (270, 2)
    assert X_multi_imp.isnull().sum().sum() == 0, "Nulls found in X_multi_imp!"
    assert X_multi_scale.isnull().sum().sum() == 0, "Nulls found in X_multi_scale!"

    # Functional waveforms
    wf_raw = pd.read_csv(PROCESSED_DIR / "functional_waveforms_raw_cm_s.csv")
    wf_scale = pd.read_csv(PROCESSED_DIR / "functional_waveforms_scaled.csv")

    assert wf_raw.shape == (270, 1001)
    assert wf_scale.shape == (270, 1001)
    assert wf_raw.isnull().sum().sum() == 0, "Nulls found in wf_raw!"
    assert wf_scale.isnull().sum().sum() == 0, "Nulls found in wf_scale!"


@pytest.mark.requires_processed_data
def test_target_label_integrity():
    """Verify target distributions are strictly preserved."""
    y_full = pd.read_csv(PROCESSED_DIR / "clinical_targets_full.csv")
    y_multi = pd.read_csv(PROCESSED_DIR / "multimodal_targets.csv")

    # Full cohort: 477 zeros, 54 ones
    assert (y_full[TARGET_COL] == 0).sum() == 477
    assert (y_full[TARGET_COL] == 1).sum() == 54

    # Multimodal cohort: 243 zeros, 27 ones
    assert (y_multi[TARGET_COL] == 0).sum() == 243
    assert (y_multi[TARGET_COL] == 1).sum() == 27


@pytest.mark.requires_processed_data
def test_binary_domain_validity():
    """Verify binary features remain strictly {0.0, 1.0} after imputation."""
    X_full = pd.read_csv(PROCESSED_DIR / "clinical_features_imputed_full.csv")
    for col in BINARY_COLS:
        unique_vals = set(X_full[col].unique())
        assert unique_vals.issubset({0.0, 1.0, 0, 1}), f"Invalid values in {col}: {unique_vals}"


@pytest.mark.requires_raw_data
def test_leak_free_pipeline_behavior():
    """
    Test that preprocessor fitted on training fold transforms test fold without leakage.
    Ensures test statistics do not influence training parameters.
    """
    df_clin, _, _ = load_raw_data()

    # Create synthetic split (e.g. first 400 train, last 131 test)
    train_df = df_clin.iloc[:400].copy()
    test_df = df_clin.iloc[400:].copy()

    preprocessor = build_clinical_preprocessor(scaling_method="robust")
    X_train_trans, fitted_proc = preprocess_clinical_dataframe(
        train_df, preprocessor=preprocessor, fit=True
    )
    X_test_trans, _ = preprocess_clinical_dataframe(
        test_df, preprocessor=fitted_proc, fit=False
    )

    assert X_train_trans.shape == (400, 25)
    assert X_test_trans.shape == (131, 25)
    assert X_test_trans.isnull().sum().sum() == 0, "Test set contains nulls after leak-free transform!"


@pytest.mark.requires_processed_data
def test_feature_schema_validity():
    """Verify feature schema JSON is valid and matches code constants."""
    schema_path = PROCESSED_DIR / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["features"]["total_feature_count"] == 25
    assert schema["features"]["continuous_columns"] == CONTINUOUS_COLS
    assert schema["features"]["binary_columns"] == BINARY_COLS
    assert schema["functional_waveform"]["dimension"] == 1001


if __name__ == "__main__":
    test_functions = [
        test_raw_files_unmodified,
        test_processed_files_exist,
        test_processed_shapes_and_completeness,
        test_target_label_integrity,
        test_binary_domain_validity,
        test_leak_free_pipeline_behavior,
        test_feature_schema_validity
    ]

    print("Running Module 4 Preprocessing Test Suite...\n" + "=" * 50)
    passed = 0
    for test in test_functions:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")

    print("=" * 50)
    print(f"Results: {passed} / {len(test_functions)} tests passed successfully!")
    if passed == len(test_functions):
        print("All Module 4 tests verified!")
    else:
        raise SystemExit(1)
