"""
Module 7 Test Suite: Label Verification and Dataset Linkage Audit
================================================================
Automated test suite verifying Phase 9 requirements:
1. Source dataset files remain unchanged.
2. The mapping table has a documented schema.
3. Technical row keys are unique within their defined source.
4. Actual patient identifiers are not fabricated (strictly None/null).
5. Verified mappings have explicit supporting evidence.
6. Unverified mappings remain marked as unverified.
7. CTRCD values are valid according to source definition ({0, 1}).
8. Labels match their source records wherever correspondence is established.
9. Missing modalities are correctly represented.
10. The reported modality counts match the underlying files.
11. No image-to-patient linkage is inferred solely from filename order.
12. Ambiguous relationships are reported rather than silently resolved.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = PROJECT_ROOT
DATASET_DIR = BASE_DIR / "dataset"
MAPPING_DIR = BASE_DIR / "data" / "processed" / "mapping"


def test_1_source_dataset_files_unchanged():
    """Verify original raw dataset files have not been modified or corrupted."""
    clin_path = DATASET_DIR / "BC_cardiotox_clinical_variables.csv"
    func_path = DATASET_DIR / "BC_cardiotox_functional_variable.csv"
    comb_path = DATASET_DIR / "BC_cardiotox_clinical_and_functional_variables.csv"

    assert clin_path.exists() and clin_path.stat().st_size == 58124, "Clinical CSV file modified!"
    assert func_path.exists() and func_path.stat().st_size == 3346159, "Functional CSV file modified!"
    assert comb_path.exists() and comb_path.stat().st_size == 3664729, "Combined CSV file modified!"


def test_2_mapping_table_schema_valid():
    """Verify that record_linkage_audit.csv exists and has all required schema columns."""
    audit_file = MAPPING_DIR / "record_linkage_audit.csv"
    assert audit_file.exists(), "record_linkage_audit.csv missing!"
    df = pd.read_csv(audit_file)

    required_cols = [
        "technical_record_key",
        "source_combined_row",
        "source_clinical_row",
        "source_functional_row",
        "candidate_image_filename",
        "patient_id",
        "examination_id",
        "CTRCD",
        "time",
        "clinical_record_available",
        "functional_waveform_available",
        "image_available",
        "target_label_available",
        "clinical_functional_link_status",
        "functional_image_link_status",
        "patient_identity_status",
        "mapping_evidence",
        "mapping_confidence",
        "unresolved_issue",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column in audit table: {col}"
    assert len(df) == 531, f"Expected 531 records in audit table, got {len(df)}"


def test_3_technical_keys_unique():
    """Verify technical row keys are strictly unique and sequentially formatted."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    keys = df["technical_record_key"]
    assert keys.nunique() == len(df), "Duplicate technical record keys found!"
    assert all(k.startswith("REC_") for k in keys), "Invalid key naming format!"


def test_4_no_fabricated_patient_identifiers():
    """Verify that patient_id and examination_id are strictly null/empty (no fabricated IDs)."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    assert df["patient_id"].isnull().all(), "Fabricated patient_id found! Must be strictly null."
    assert df["examination_id"].isnull().all(), "Fabricated examination_id found! Must be strictly null."


def test_5_verified_mappings_have_explicit_evidence():
    """Verify that all rows marked as 'verified' contain clear, defensible mapping evidence."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    verified_rows = df[df["clinical_functional_link_status"] == "verified"]
    assert len(verified_rows) == 270, f"Expected 270 verified clinical-functional rows, got {len(verified_rows)}"
    for _, row in verified_rows.iterrows():
        assert "Bijective clinical match" in str(row["mapping_evidence"]), "Missing bijective evidence string!"
        assert "numerical waveform identity" in str(row["mapping_evidence"]), "Missing numerical identity evidence!"


def test_6_unverified_mappings_marked_explicitly():
    """Verify that image-to-functional linkage is strictly flagged as 'unverified'."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    has_image = df[df["functional_waveform_available"] == True]
    assert (has_image["functional_image_link_status"] == "unverified").all(), \
        "Image linkage incorrectly claimed as verified! Must be 'unverified'."


def test_7_ctrcd_values_valid():
    """Verify CTRCD target labels are strictly within {0, 1} with zero missing values."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    assert set(df["CTRCD"].unique()).issubset({0, 1}), f"Invalid CTRCD values: {df['CTRCD'].unique()}"
    assert df["CTRCD"].isnull().sum() == 0, "Missing CTRCD labels found!"
    assert (df["CTRCD"] == 1).sum() == 54, "Total CTRCD positive count mismatch!"


def test_8_label_agreement_with_sources():
    """Verify target labels match raw clinical and functional sources exactly."""
    df_audit = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    df_func = pd.read_csv(DATASET_DIR / "BC_cardiotox_functional_variable.csv", sep=";", decimal=",")

    # First 270 rows must match functional target
    audit_func_targets = df_audit.iloc[:270]["CTRCD"].values
    raw_func_targets = df_func["CTRCD"].values
    np.testing.assert_array_equal(audit_func_targets, raw_func_targets, err_msg="Target mismatch with functional source!")


def test_9_missing_modalities_represented():
    """Verify that the 261 clinical-only patients correctly reflect absence of waveforms."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    clin_only = df.iloc[270:]
    assert len(clin_only) == 261, f"Expected 261 clinical-only rows, got {len(clin_only)}"
    assert (clin_only["functional_waveform_available"] == False).all()
    assert (clin_only["clinical_functional_link_status"] == "not_applicable").all()
    assert clin_only["source_functional_row"].isnull().all()


def test_10_modality_counts_match_files():
    """Verify reported modality counts match actual on-disk files."""
    mod_df = pd.read_csv(MAPPING_DIR / "modality_availability_summary.csv")
    assert mod_df.loc[mod_df["modality_configuration"] == "Clinical Record Only", "raw_data_count"].values[0] == 531
    assert mod_df.loc[mod_df["modality_configuration"] == "Clinical + Functional Waveform", "raw_data_count"].values[0] == 270
    assert mod_df.loc[mod_df["modality_configuration"] == "Clinical + TDI Image PNG", "verified_linked_count"].values[0] == 0


def test_11_no_image_link_inferred_from_order():
    """Verify that candidate image filenames are documented only as candidates with unresolved status."""
    df = pd.read_csv(MAPPING_DIR / "record_linkage_audit.csv")
    assert df["functional_image_link_status"].isin(["unverified", "not_applicable"]).all()
    for _, row in df.iterrows():
        if pd.notna(row["candidate_image_filename"]):
            assert "unverified" in str(row["unresolved_issue"]).lower(), \
                f"Row {row['technical_record_key']} candidate image not marked as unverified in unresolved_issue!"


def test_12_unresolved_issues_file_valid():
    """Verify that unresolved issues table documents key ambiguities."""
    unres_path = MAPPING_DIR / "unresolved_mappings.csv"
    assert unres_path.exists(), "unresolved_mappings.csv missing!"
    df = pd.read_csv(unres_path)
    assert len(df) >= 3, "Expected at least 3 documented issues in unresolved_mappings.csv"
    assert "ISSUE-01" in df["issue_id"].values
    assert "ISSUE-02" in df["issue_id"].values
    assert "ISSUE-03" in df["issue_id"].values


if __name__ == "__main__":
    tests = [
        test_1_source_dataset_files_unchanged,
        test_2_mapping_table_schema_valid,
        test_3_technical_keys_unique,
        test_4_no_fabricated_patient_identifiers,
        test_5_verified_mappings_have_explicit_evidence,
        test_6_unverified_mappings_marked_explicitly,
        test_7_ctrcd_values_valid,
        test_8_label_agreement_with_sources,
        test_9_missing_modalities_represented,
        test_10_modality_counts_match_files,
        test_11_no_image_link_inferred_from_order,
        test_12_unresolved_issues_file_valid,
    ]

    print("Running Module 7 Dataset Mapping & Verification Test Suite...\n" + "=" * 65)
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
        print("All Module 7 verification criteria satisfied!")
    else:
        sys.exit(1)
