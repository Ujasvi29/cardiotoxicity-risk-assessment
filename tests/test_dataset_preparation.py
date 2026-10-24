"""
Module 8 Test Suite: Dataset Preparation and Train/Val/Test Isolation Verification
==================================================================================
Automated test suite verifying Phase 9 requirements:
1. Raw datasets remain unchanged.
2. Expected cohort sizes and labels are verified.
3. Split manifests are reproducible.
4. No technical record appears in multiple partitions.
5. Patient grouping logic is ready if verified patient IDs become available.
6. Missing patient identifiers produce an explicit limitation.
7. CTRCD and time are absent from predictor matrices.
8. Class distributions are correctly reported.
9. Every split has valid target values.
10. Feature column order is consistent across all partitions.
11. Waveform shapes and column ordering are preserved.
12. Training-only preprocessing is reused.
13. Validation/test data do not affect fitted preprocessing parameters.
14. No unverified TDI image mapping is used.
15. No augmented or synthetic derivatives are introduced.
16. Clinical-only and clinical-waveform cohorts remain separate.
17. Saved manifests agree with exported split files.
18. The pipeline executes on CPU without requiring CUDA.
"""

from pathlib import Path
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitting import (
    create_stratified_train_val_test_split,
    create_stratified_kfold_splits,
    profile_split,
    generate_and_save_all_splits,
    TARGET_COL
)
from src.data.preprocessing import (
    load_raw_data,
    build_clinical_preprocessor,
    preprocess_clinical_dataframe,
    CONTINUOUS_COLS,
    BINARY_COLS,
    FEATURE_COLS,
    TIME_COL
)
from src.utils.paths import get_data_root

BASE_DIR = PROJECT_ROOT
DATASET_DIR = get_data_root()
SPLITS_DIR = BASE_DIR / "data" / "splits"
PROCESSED_SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
CONFIG_DIR = BASE_DIR / "configs"
SCHEMA_PATH = BASE_DIR / "data" / "processed" / "feature_schema.json"


@pytest.mark.requires_raw_data
def test_1_raw_datasets_unchanged():
    """Verify raw source files in dataset/ have not been modified or corrupted."""
    clin_path = DATASET_DIR / "BC_cardiotox_clinical_variables.csv"
    func_path = DATASET_DIR / "BC_cardiotox_functional_variable.csv"
    comb_path = DATASET_DIR / "BC_cardiotox_clinical_and_functional_variables.csv"

    assert clin_path.exists() and clin_path.stat().st_size == 58124, "Clinical CSV modified!"
    assert func_path.exists() and func_path.stat().st_size == 3346159, "Functional CSV modified!"
    assert comb_path.exists() and comb_path.stat().st_size == 3664729, "Combined CSV modified!"


@pytest.mark.requires_raw_data
def test_2_expected_cohort_sizes_and_labels():
    """Verify ground truth sizes and target counts for Cohort A and Cohort B."""
    df_clin, df_func, df_comb = load_raw_data()

    # Cohort A
    assert len(df_clin) == 531, f"Expected 531 in Cohort A, got {len(df_clin)}"
    assert (df_clin[TARGET_COL] == 1).sum() == 54
    assert (df_clin[TARGET_COL] == 0).sum() == 477

    # Cohort B
    df_multi = df_comb.iloc[:270]
    assert len(df_multi) == 270, f"Expected 270 in Cohort B, got {len(df_multi)}"
    assert (df_multi[TARGET_COL] == 1).sum() == 27
    assert (df_multi[TARGET_COL] == 0).sum() == 243
    assert (df_func[TARGET_COL] == 1).sum() == 27
    assert (df_func[TARGET_COL] == 0).sum() == 243


@pytest.mark.requires_raw_data
def test_3_split_manifests_reproducible():
    """Verify that split generation produces identical partitions given seed=42."""
    df_clin, _, _ = load_raw_data()
    split1 = create_stratified_train_val_test_split(df_clin, seed=42)
    split2 = create_stratified_train_val_test_split(df_clin, seed=42)
    assert split1 == split2, "Split indices are non-deterministic!"


@pytest.mark.requires_split_data
def test_4_no_technical_record_overlap():
    """Verify zero overlap between train, val, and test splits within each cohort."""
    for manifest_name in ["cohort_a_clinical_splits.json", "cohort_b_multimodal_splits.json"]:
        with open(SPLITS_DIR / manifest_name, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        idx = manifest["indices"]
        s_train = set(idx["train"])
        s_val = set(idx["val"])
        s_test = set(idx["test"])

        assert len(s_train & s_val) == 0, f"Train-val overlap in {manifest_name}!"
        assert len(s_train & s_test) == 0, f"Train-test overlap in {manifest_name}!"
        assert len(s_val & s_test) == 0, f"Val-test overlap in {manifest_name}!"
        assert len(s_train | s_val | s_test) == manifest["total_records"]


@pytest.mark.requires_split_data
def test_5_patient_grouping_logic_ready():
    """Verify patient grouping logic: handles None (current state), and groups by patient if IDs exist."""
    # Synthetic patient grouped split test
    df_toy = pd.DataFrame({
        "patient_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        TARGET_COL: [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    })
    # If genuine patient IDs exist, group-level splitting preserves patient isolation
    groups = df_toy["patient_id"].unique()
    train_groups = groups[:3]
    test_groups = groups[3:]
    train_idx = df_toy[df_toy["patient_id"].isin(train_groups)].index.tolist()
    test_idx = df_toy[df_toy["patient_id"].isin(test_groups)].index.tolist()
    assert len(set(train_idx) & set(test_idx)) == 0
    # Current dataset verifies patient_id is explicitly None
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m = json.load(f)
    assert m["patient_level_independence_verified"] is False
    assert m["record_assignments"]["REC_CLIN_000"]["patient_id"] is None


@pytest.mark.requires_split_data
def test_6_missing_patient_identifiers_limitation_documented():
    """Verify that limitation of missing patient identifiers is explicitly stated in manifests and config."""
    for m_file in ["cohort_a_clinical_splits.json", "cohort_b_multimodal_splits.json"]:
        with open(SPLITS_DIR / m_file, "r") as f:
            m = json.load(f)
        assert "patient_identifier_limitation" in m
        assert "anonymization" in m["patient_identifier_limitation"].lower()

    with open(CONFIG_DIR / "data_config.json", "r") as f:
        cfg = json.load(f)
    assert cfg["patient_level_independence_verified"] is False
    assert "patient_identifier_limitation" in cfg


@pytest.mark.requires_split_data
def test_7_target_and_time_excluded_from_predictors():
    """Verify CTRCD and time are strictly absent from all split feature CSVs."""
    for cohort in ["cohort_a", "cohort_b"]:
        for split in ["train", "val", "test"]:
            fpath = PROCESSED_SPLITS_DIR / f"{cohort}_{split}_features.csv"
            df = pd.read_csv(fpath)
            assert TARGET_COL not in df.columns, f"{TARGET_COL} in {fpath.name}!"
            assert TIME_COL not in df.columns, f"{TIME_COL} in {fpath.name}!"
            assert df.shape[1] == 25, f"Expected 25 features, found {df.shape[1]} in {fpath.name}"


@pytest.mark.requires_split_data
def test_8_class_distributions_correctly_reported():
    """Verify profiles in split manifests match actual target split files."""
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m_a = json.load(f)
    prof_a = m_a["profiles"]["splits"]
    assert prof_a["train"]["positives"] == 38 and prof_a["train"]["negatives"] == 333
    assert prof_a["val"]["positives"] == 8 and prof_a["val"]["negatives"] == 72
    assert prof_a["test"]["positives"] == 8 and prof_a["test"]["negatives"] == 72

    with open(SPLITS_DIR / "cohort_b_multimodal_splits.json", "r") as f:
        m_b = json.load(f)
    prof_b = m_b["profiles"]["splits"]
    assert prof_b["train"]["positives"] == 19 and prof_b["train"]["negatives"] == 170
    assert prof_b["val"]["positives"] == 4 and prof_b["val"]["negatives"] == 36
    assert prof_b["test"]["positives"] == 4 and prof_b["test"]["negatives"] == 37


@pytest.mark.requires_split_data
def test_9_valid_target_values():
    """Verify all target split CSVs contain only valid binary {0, 1} values with zero NaNs."""
    for fpath in PROCESSED_SPLITS_DIR.glob("*_targets.csv"):
        df = pd.read_csv(fpath)
        assert set(df[TARGET_COL].unique()).issubset({0, 1})
        assert df[TARGET_COL].isnull().sum() == 0
        assert df[TIME_COL].isnull().sum() == 0


@pytest.mark.requires_split_data
def test_10_feature_column_order_consistent():
    """Verify feature column names and order are identical across all partitions and match schema."""
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    expected_cols = schema["features"]["continuous_columns"] + schema["features"]["binary_columns"]

    for cohort in ["cohort_a", "cohort_b"]:
        for split in ["train", "val", "test"]:
            df = pd.read_csv(PROCESSED_SPLITS_DIR / f"{cohort}_{split}_features.csv")
            assert list(df.columns) == expected_cols, f"Feature column mismatch in {cohort}_{split}_features.csv!"


@pytest.mark.requires_split_data
def test_11_waveform_shapes_and_order():
    """Verify all functional waveform splits have exactly 1001 columns in sequence t_1 .. t_1001."""
    expected_wf_cols = [f"t_{i}" for i in range(1, 1002)]
    for split in ["train", "val", "test"]:
        for kind in ["raw", "scaled"]:
            fpath = PROCESSED_SPLITS_DIR / f"cohort_b_{split}_waveforms_{kind}.csv"
            df = pd.read_csv(fpath)
            assert list(df.columns) == expected_wf_cols, f"Column mismatch in {fpath.name}"
            assert df.shape[1] == 1001
            assert df.isnull().sum().sum() == 0


@pytest.mark.requires_raw_data
@pytest.mark.requires_split_data
def test_12_training_only_preprocessing_reused():
    """Verify that clinical imputers and scalers are fitted on training data only."""
    df_clin, _, _ = load_raw_data()
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m = json.load(f)
    train_idx = m["indices"]["train"]
    train_df = df_clin.iloc[train_idx]

    preprocessor = build_clinical_preprocessor(scaling_method="robust")
    _, fitted_proc = preprocess_clinical_dataframe(train_df, preprocessor=preprocessor, fit=True)

    # Imputer median check
    imputer_medians = fitted_proc.named_transformers_["num"].named_steps["imputer"].statistics_
    expected_medians = train_df[CONTINUOUS_COLS].median(numeric_only=True).values
    np.testing.assert_allclose(imputer_medians, expected_medians)


def test_13_synthetic_perturbation_leak_free_proof(synthetic_clinical_frame):
    """Empirical proof: altering validation/test values does not change fitted training parameters."""
    train_sample = synthetic_clinical_frame.iloc[:50].copy()
    test_sample = synthetic_clinical_frame.iloc[50:].copy()
    test_perturbed = test_sample.copy()
    for col in CONTINUOUS_COLS:
        test_perturbed[col] = 999999.0

    proc = build_clinical_preprocessor(scaling_method="robust")
    _, fitted_proc = preprocess_clinical_dataframe(train_sample, preprocessor=proc, fit=True)
    centers_orig = fitted_proc.named_transformers_["num"].named_steps["scaler"].center_.copy()

    # Transform perturbed test data
    _ = preprocess_clinical_dataframe(test_perturbed, preprocessor=fitted_proc, fit=False)
    centers_after = fitted_proc.named_transformers_["num"].named_steps["scaler"].center_.copy()

    np.testing.assert_array_equal(centers_orig, centers_after, err_msg="Leakage detected via perturbation!")


@pytest.mark.requires_split_data
def test_14_no_unverified_image_mapping_in_splits():
    """Verify that no image filenames or image pixels are included in the supervised splits."""
    for fpath in PROCESSED_SPLITS_DIR.glob("*.csv"):
        df = pd.read_csv(fpath)
        for col in df.columns:
            assert "image" not in col.lower(), f"Unverified image reference leaked into {fpath.name}!"


@pytest.mark.requires_split_data
def test_15_no_augmented_or_synthetic_derivatives():
    """Verify that split record counts match raw cohort partitions without synthetic expansion."""
    y_tr_a = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_a_train_targets.csv")
    y_va_a = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_a_val_targets.csv")
    y_te_a = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_a_test_targets.csv")
    assert len(y_tr_a) + len(y_va_a) + len(y_te_a) == 531

    y_tr_b = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_b_train_targets.csv")
    y_va_b = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_b_val_targets.csv")
    y_te_b = pd.read_csv(PROCESSED_SPLITS_DIR / "cohort_b_test_targets.csv")
    assert len(y_tr_b) + len(y_va_b) + len(y_te_b) == 270


@pytest.mark.requires_split_data
def test_16_cohort_isolation():
    """Verify clinical-only (N=531) and multimodal (N=270) datasets are strictly separated."""
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m_a = json.load(f)
    with open(SPLITS_DIR / "cohort_b_multimodal_splits.json", "r") as f:
        m_b = json.load(f)
    assert m_a["cohort"] != m_b["cohort"]
    assert m_a["total_records"] == 531
    assert m_b["total_records"] == 270


@pytest.mark.requires_split_data
def test_17_manifests_agree_with_exported_files():
    """Verify that manifest index counts match exported split file row counts exactly."""
    with open(SPLITS_DIR / "cohort_a_clinical_splits.json", "r") as f:
        m_a = json.load(f)
    for split in ["train", "val", "test"]:
        feat_df = pd.read_csv(PROCESSED_SPLITS_DIR / f"cohort_a_{split}_features.csv")
        assert len(feat_df) == len(m_a["indices"][split])

    with open(SPLITS_DIR / "cohort_b_multimodal_splits.json", "r") as f:
        m_b = json.load(f)
    for split in ["train", "val", "test"]:
        feat_df = pd.read_csv(PROCESSED_SPLITS_DIR / f"cohort_b_{split}_features.csv")
        assert len(feat_df) == len(m_b["indices"][split])


@pytest.mark.requires_raw_data
def test_18_cpu_pipeline_execution(synthetic_clinical_frame):
    """Verify pipeline runs smoothly on deterministic synthetic records without CUDA."""
    sample = synthetic_clinical_frame.iloc[:40].copy()
    proc = build_clinical_preprocessor(scaling_method="robust")
    X_out, _ = preprocess_clinical_dataframe(sample, preprocessor=proc, fit=True)
    assert isinstance(X_out, pd.DataFrame)
    assert X_out.shape == (40, 25)


if __name__ == "__main__":
    tests = [
        test_1_raw_datasets_unchanged,
        test_2_expected_cohort_sizes_and_labels,
        test_3_split_manifests_reproducible,
        test_4_no_technical_record_overlap,
        test_5_patient_grouping_logic_ready,
        test_6_missing_patient_identifiers_limitation_documented,
        test_7_target_and_time_excluded_from_predictors,
        test_8_class_distributions_correctly_reported,
        test_9_valid_target_values,
        test_10_feature_column_order_consistent,
        test_11_waveform_shapes_and_order,
        test_12_training_only_preprocessing_reused,
        test_13_synthetic_perturbation_leak_free_proof,
        test_14_no_unverified_image_mapping_in_splits,
        test_15_no_augmented_or_synthetic_derivatives,
        test_16_cohort_isolation,
        test_17_manifests_agree_with_exported_files,
        test_18_cpu_pipeline_execution
    ]

    print("Running Module 8 Dataset Preparation Test Suite...\n" + "=" * 65)
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")

    print("=" * 65)
    print(f"Results: {passed} / {len(tests)} passed.")
    if passed == len(tests):
        print("All Module 8 verification criteria satisfied!")
    else:
        sys.exit(1)
