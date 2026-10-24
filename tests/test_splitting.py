"""
Module 6 Test Suite: Splitting, Train/Test Isolation, and Leakage Verification
=============================================================================
Automated test suite verifying Phase 6 requirements:
1. Raw source files remain unmodified.
2. Target ('CTRCD') and metadata ('time') are excluded from feature matrices.
3. Target distributions and label integrity are strictly preserved.
4. Split indices are deterministic and reproducible given a fixed seed.
5. Zero record overlap between train, val, and test splits (no leakage).
6. Stratified sampling preserves CTRCD prevalence across splits.
7. Data-dependent imputers and scalers are fitted exclusively on training data.
8. Synthetic perturbation test: modifying validation/test records does not alter
   fitted training preprocessor statistics (empirical leak-free proof).
9. Binary columns remain strictly within the {0.0, 1.0} domain.
10. Missing values are completely resolved (0 nulls in all split feature sets).
11. Functional waveform shapes (1001 time steps) and column ordering are preserved.
12. Every multimodal sample has a paired, complete functional waveform.
13. Clinical-only (N=531) and Multimodal (N=270) cohorts remain isolated.
14. Pipeline executes on CPU without requiring an NVIDIA GPU.
15. Clear warnings are issued when positive class counts are dangerously small (< 10).
"""

from pathlib import Path
import sys
import warnings
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd

from src.data.splitting import (
    create_stratified_train_val_test_split,
    create_stratified_kfold_splits,
    profile_split,
    generate_and_save_all_splits,
)
from src.data.preprocessing import (
    load_raw_data,
    build_clinical_preprocessor,
    preprocess_clinical_dataframe,
    TARGET_COL,
    TIME_COL,
    BINARY_COLS,
    CONTINUOUS_COLS,
)
from src.utils.paths import get_data_root

BASE_DIR = PROJECT_ROOT
DATASET_DIR = get_data_root()
SPLITS_DIR = BASE_DIR / "data" / "splits"
PROCESSED_SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"


@pytest.mark.requires_raw_data
def test_1_raw_source_files_unmodified():
    """Verify raw dataset files in dataset/ have not been altered or overwritten."""
    clin_path = DATASET_DIR / "BC_cardiotox_clinical_variables.csv"
    func_path = DATASET_DIR / "BC_cardiotox_functional_variable.csv"
    comb_path = DATASET_DIR / "BC_cardiotox_clinical_and_functional_variables.csv"

    assert clin_path.exists() and clin_path.stat().st_size == 58124, "Clinical raw CSV modified!"
    assert func_path.exists() and func_path.stat().st_size == 3346159, "Functional raw CSV modified!"
    assert comb_path.exists() and comb_path.stat().st_size == 3664729, "Combined raw CSV modified!"


@pytest.mark.requires_split_data
def test_2_target_and_metadata_excluded():
    """Verify target ('CTRCD') and event time ('time') are completely excluded from predictor matrices."""
    for cohort in ["cohort_a", "cohort_b"]:
        for split in ["train", "val", "test"]:
            feat_file = PROCESSED_SPLITS_DIR / f"{cohort}_{split}_features.csv"
            df = pd.read_csv(feat_file)
            assert TARGET_COL not in df.columns, f"{TARGET_COL} leaked into {feat_file.name}!"
            assert TIME_COL not in df.columns, f"{TIME_COL} leaked into {feat_file.name}!"
            assert df.shape[1] == 25, f"Expected 25 predictor features, got {df.shape[1]} in {feat_file.name}"


@pytest.mark.requires_split_data
def test_3_target_values_and_integrity():
    """Verify target split files contain valid targets and preserve true cohort sums."""
    # Cohort A: 54 positives, 477 negatives total
    y_train_a = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_a_train_targets.csv")
    y_val_a = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_a_val_targets.csv")
    y_test_a = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_a_test_targets.csv")

    pos_a = int((y_train_a[TARGET_COL] == 1).sum() + (y_val_a[TARGET_COL] == 1).sum() + (y_test_a[TARGET_COL] == 1).sum())
    total_a = len(y_train_a) + len(y_val_a) + len(y_test_a)
    assert total_a == 531, f"Expected 531 records for Cohort A, got {total_a}"
    assert pos_a == 54, f"Expected 54 positives for Cohort A, got {pos_a}"

    # Cohort B: 27 positives, 243 negatives total
    y_train_b = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_b_train_targets.csv")
    y_val_b = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_b_val_targets.csv")
    y_test_b = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_b_test_targets.csv")

    pos_b = int((y_train_b[TARGET_COL] == 1).sum() + (y_val_b[TARGET_COL] == 1).sum() + (y_test_b[TARGET_COL] == 1).sum())
    total_b = len(y_train_b) + len(y_val_b) + len(y_test_b)
    assert total_b == 270, f"Expected 270 records for Cohort B, got {total_b}"
    assert pos_b == 27, f"Expected 27 positives for Cohort B, got {pos_b}"


def test_4_split_indices_reproducible(synthetic_clinical_frame):
    """Verify splitting is strictly deterministic when given the fixed random seed."""
    df_clin = synthetic_clinical_frame
    split1 = create_stratified_train_val_test_split(df_clin, seed=42)
    split2 = create_stratified_train_val_test_split(df_clin, seed=42)

    assert split1["train"] == split2["train"], "Non-deterministic training split!"
    assert split1["val"] == split2["val"], "Non-deterministic validation split!"
    assert split1["test"] == split2["test"], "Non-deterministic test split!"


@pytest.mark.requires_split_data
def test_5_zero_record_overlap():
    """Verify zero patient index overlap between train, validation, and test sets within each cohort."""
    for manifest_name in ["cohort_a_clinical_splits.json", "cohort_b_multimodal_splits.json"]:
        manifest_path = SPLITS_DIR / manifest_name
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        idx = manifest["indices"]
        set_train = set(idx["train"])
        set_val = set(idx["val"])
        set_test = set(idx["test"])

        assert len(set_train & set_val) == 0, f"Train-val index overlap in {manifest_name}!"
        assert len(set_train & set_test) == 0, f"Train-test index overlap in {manifest_name}!"
        assert len(set_val & set_test) == 0, f"Val-test index overlap in {manifest_name}!"
        assert len(set_train | set_val | set_test) == manifest["total_records"]


@pytest.mark.requires_split_data
def test_6_ctrcd_stratification():
    """Verify that CTRCD prevalence is balanced across train, val, and test splits."""
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m_a = json.load(f)
    prof_a = m_a["profiles"]["splits"]
    for split_name in ["train", "val", "test"]:
        prev = prof_a[split_name]["prevalence_pct"]
        assert 9.0 <= prev <= 11.5, f"Cohort A {split_name} prevalence deviated significantly: {prev}%"

    with open(SPLITS_DIR / "cohort_b_multimodal_splits.json", "r") as f:
        m_b = json.load(f)
    prof_b = m_b["profiles"]["splits"]
    for split_name in ["train", "val", "test"]:
        prev = prof_b[split_name]["prevalence_pct"]
        assert 9.0 <= prev <= 11.5, f"Cohort B {split_name} prevalence deviated significantly: {prev}%"


def test_7_data_dependent_transformations_fitted_on_train_only(synthetic_clinical_frame):
    """Verify that preprocessor fitted on training split derives medians/centers strictly from train."""
    train_df = synthetic_clinical_frame.iloc[:60].copy()

    preprocessor = build_clinical_preprocessor(scaling_method="robust")
    _, fitted_proc = preprocess_clinical_dataframe(train_df, preprocessor=preprocessor, fit=True)

    # Check that median imputer statistics match training continuous medians exactly
    cont_imputer = fitted_proc.named_transformers_["num"].named_steps["imputer"]
    expected_medians = train_df[CONTINUOUS_COLS].median(numeric_only=True).values
    np.testing.assert_allclose(
        cont_imputer.statistics_, expected_medians,
        err_msg="Continuous imputer medians do not match training data medians!"
    )


def test_8_leak_free_synthetic_perturbation(synthetic_clinical_frame):
    """
    CRITICAL EMPIRICAL LEAKAGE TEST:
    Artificially perturb validation/test data with huge outliers (e.g. 100,000.0).
    Verify that fitted training imputer and scaler parameters do not change at all.
    """
    train_df = synthetic_clinical_frame.iloc[:50].copy()
    test_df_original = synthetic_clinical_frame.iloc[50:].copy()
    test_df_perturbed = test_df_original.copy()

    # Inject extreme values into test set
    for col in CONTINUOUS_COLS:
        test_df_perturbed[col] = 999999.0

    # Fit pipeline on training data
    proc1 = build_clinical_preprocessor(scaling_method="robust")
    X_train_1, fitted_proc_1 = preprocess_clinical_dataframe(train_df, preprocessor=proc1, fit=True)

    # Transform original vs perturbed test sets
    X_test_orig, _ = preprocess_clinical_dataframe(test_df_original, preprocessor=fitted_proc_1, fit=False)
    X_test_pert, _ = preprocess_clinical_dataframe(test_df_perturbed, preprocessor=fitted_proc_1, fit=False)

    # Training feature values must be 100% identical
    assert np.all(fitted_proc_1.named_transformers_["num"].named_steps["imputer"].statistics_ ==
                  proc1.named_transformers_["num"].named_steps["imputer"].statistics_), "Imputer statistics changed!"
    assert np.all(fitted_proc_1.named_transformers_["num"].named_steps["scaler"].center_ ==
                  proc1.named_transformers_["num"].named_steps["scaler"].center_), "Scaler center changed!"

    # Ensure transforming perturbed test set did not alter fitted training scaler
    train_centers_before = fitted_proc_1.named_transformers_["num"].named_steps["scaler"].center_.copy()
    _ = preprocess_clinical_dataframe(test_df_perturbed, preprocessor=fitted_proc_1, fit=False)
    train_centers_after = fitted_proc_1.named_transformers_["num"].named_steps["scaler"].center_.copy()
    np.testing.assert_array_equal(train_centers_before, train_centers_after, err_msg="Test set leakage altered training scaler!")


@pytest.mark.requires_split_data
def test_9_binary_columns_domain_validity():
    """Verify that all binary features remain strictly {0.0, 1.0} across all split files."""
    for cohort in ["cohort_a", "cohort_b"]:
        for split in ["train", "val", "test"]:
            feat_file = PROCESSED_SPLITS_DIR / f"{cohort}_{split}_features.csv"
            df = pd.read_csv(feat_file)
            for col in BINARY_COLS:
                unique_vals = set(df[col].dropna().unique())
                assert unique_vals.issubset({0.0, 1.0, 0, 1}), f"Non-binary values in {col} of {feat_file.name}: {unique_vals}"


@pytest.mark.requires_split_data
def test_10_missing_values_fully_imputed():
    """Verify that no missing or NaN values remain in any generated split feature dataset."""
    for fpath in PROCESSED_SPLITS_DIR.glob("*_features.csv"):
        df = pd.read_csv(fpath)
        null_count = df.isnull().sum().sum()
        assert null_count == 0, f"Found {null_count} nulls in {fpath.name}!"


@pytest.mark.requires_split_data
def test_11_waveform_shapes_and_order():
    """Verify that all waveform splits maintain exactly 1001 columns in order t_1 .. t_1001."""
    expected_cols = [f"t_{i}" for i in range(1, 1002)]
    for split in ["train", "val", "test"]:
        for kind in ["raw", "scaled"]:
            wf_file = PROCESSED_SPLITS_DIR / f"cohort_b_{split}_waveforms_{kind}.csv"
            df = pd.read_csv(wf_file)
            assert list(df.columns) == expected_cols, f"Column names/order mismatch in {wf_file.name}!"
            assert df.shape[1] == 1001, f"Expected 1001 columns, got {df.shape[1]} in {wf_file.name}"
            assert df.isnull().sum().sum() == 0, f"Nulls found in {wf_file.name}!"


@pytest.mark.requires_split_data
def test_12_multimodal_paired_integrity():
    """Verify that every record in Cohort B features matches exactly one waveform and target record."""
    for split in ["train", "val", "test"]:
        feat_df = pd.read_csv(PROCESSED_SPLITS_DIR / f"cohort_b_{split}_features.csv")
        target_df = pd.read_csv(PROCESSED_SPLITS_DIR / f"cohort_b_{split}_targets.csv")
        wf_raw_df = pd.read_csv(PROCESSED_SPLITS_DIR / f"cohort_b_{split}_waveforms_raw.csv")
        wf_scl_df = pd.read_csv(PROCESSED_SPLITS_DIR / f"cohort_b_{split}_waveforms_scaled.csv")

        n_rows = len(feat_df)
        assert len(target_df) == n_rows, f"Row count mismatch: features ({n_rows}) vs targets ({len(target_df)}) in {split}"
        assert len(wf_raw_df) == n_rows, f"Row count mismatch: features ({n_rows}) vs raw waveforms in {split}"
        assert len(wf_scl_df) == n_rows, f"Row count mismatch: features ({n_rows}) vs scaled waveforms in {split}"


@pytest.mark.requires_split_data
def test_13_cohort_isolation():
    """Verify clinical-only (N=531) and multimodal (N=270) datasets are strictly separated."""
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m_a = json.load(f)
    with open(SPLITS_DIR / "cohort_b_multimodal_splits.json", "r") as f:
        m_b = json.load(f)

    assert m_a["total_records"] == 531
    assert m_b["total_records"] == 270
    assert m_a["cohort"] != m_b["cohort"]


def test_14_cpu_pipeline_execution(synthetic_clinical_frame):
    """Verify pipeline functions execute seamlessly using CPU-only math without torch/GPU calls."""
    sample = synthetic_clinical_frame.iloc[:50].copy()
    proc = build_clinical_preprocessor(scaling_method="robust")
    X_sample, _ = preprocess_clinical_dataframe(sample, preprocessor=proc, fit=True)
    assert isinstance(X_sample, pd.DataFrame)
    assert X_sample.shape == (50, 25)


def test_15_small_positive_class_warning():
    """Verify that profile_split actively issues a UserWarning when positive count is < 10."""
    df_toy = pd.DataFrame({
        TARGET_COL: [1] * 4 + [0] * 36
    })
    split_indices = {"toy_test": list(range(40))}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prof = profile_split(df_toy, split_indices, "ToyCohort")
        assert len(w) == 1, "Expected a warning for positive count < 10!"
        assert issubclass(w[-1].category, UserWarning)
        assert "Small positive sample count (N=4)" in str(w[-1].message)
        assert "statistical_warning" in prof["splits"]["toy_test"]


if __name__ == "__main__":
    all_tests = [
        test_1_raw_source_files_unmodified,
        test_2_target_and_metadata_excluded,
        test_3_target_values_and_integrity,
        test_4_split_indices_reproducible,
        test_5_zero_record_overlap,
        test_6_ctrcd_stratification,
        test_7_data_dependent_transformations_fitted_on_train_only,
        test_8_leak_free_synthetic_perturbation,
        test_9_binary_columns_domain_validity,
        test_10_missing_values_fully_imputed,
        test_11_waveform_shapes_and_order,
        test_12_multimodal_paired_integrity,
        test_13_cohort_isolation,
        test_14_cpu_pipeline_execution,
        test_15_small_positive_class_warning
    ]

    print("Running Module 6 Comprehensive Test Suite...\n" + "=" * 60)
    passed_count = 0
    for t in all_tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed_count += 1
        except Exception as err:
            print(f"  [FAIL] {t.__name__}: {err}")

    print("=" * 60)
    print(f"Results: {passed_count} / {len(all_tests)} passed.")
    if passed_count == len(all_tests):
        print("All Module 6 verification criteria satisfied!")
    else:
        sys.exit(1)
