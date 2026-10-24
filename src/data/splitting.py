"""
Module 6: Reproducible Stratified Splitting
============================================
Implements leak-free, reproducible train/validation/test partitions
and Stratified K-Fold cross-validation splits for:
1. Cohort A (Clinical-only, N=531, 54 CTRCD-positive cases)
2. Cohort B (Multimodal, N=270, 27 CTRCD-positive cases)

Ensures:
- Strict stratification on CTRCD target label.
- Deterministic partitioning via fixed random seed (default: 42).
- Zero record overlap between train, val, and test splits.
- Explicit warnings on small positive-class sample sizes.
- Saving human-readable and machine-readable split manifests.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

BASE_DIR = Path(r"c:\dev\cardiotoxicity risk assesment")
DATASET_DIR = BASE_DIR / "dataset"
SPLITS_DIR = BASE_DIR / "data" / "splits"
CONFIG_DIR = BASE_DIR / "configs"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RANDOM_SEED = 42
TARGET_COL = "CTRCD"


def create_stratified_train_val_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_RANDOM_SEED
) -> Dict[str, List[int]]:
    """
    Split a dataframe into train, validation, and test indices using two-stage
    stratified sampling to guarantee exact target distribution preservation.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    y = df[target_col].values
    indices = np.arange(len(df))

    # Stage 1: Separate Train (70%) from Temporary (30% = Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx_loc, temp_idx_loc = next(sss1.split(indices, y))

    train_indices = indices[train_idx_loc].tolist()
    temp_indices = indices[temp_idx_loc]
    temp_y = y[temp_idx_loc]

    # Stage 2: Split Temporary (30%) equally into Val (15%) and Test (15%)
    val_share_of_temp = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_share_of_temp), random_state=seed)
    val_idx_loc, test_idx_loc = next(sss2.split(temp_indices, temp_y))

    val_indices = temp_indices[val_idx_loc].tolist()
    test_indices = temp_indices[test_idx_loc].tolist()

    # Validation checks
    assert len(set(train_indices) & set(val_indices)) == 0, "Overlap between train and val!"
    assert len(set(train_indices) & set(test_indices)) == 0, "Overlap between train and test!"
    assert len(set(val_indices) & set(test_indices)) == 0, "Overlap between val and test!"
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(df), "Index count mismatch!"

    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices)
    }


def create_stratified_kfold_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_splits: int = 5,
    seed: int = DEFAULT_RANDOM_SEED
) -> List[Dict[str, List[int]]]:
    """Generate 5-fold stratified cross-validation splits."""
    y = df[target_col].values
    indices = np.arange(len(df))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(indices, y), 1):
        folds.append({
            "fold": fold_num,
            "train": sorted(indices[train_idx].tolist()),
            "val": sorted(indices[val_idx].tolist())
        })

    return folds


def profile_split(df: pd.DataFrame, split_indices: Dict[str, List[int]], cohort_name: str) -> Dict:
    """Analyze class distribution and issue warnings for small positive counts."""
    profile = {"cohort": cohort_name, "total_records": len(df), "splits": {}}

    for split_name, idx_list in split_indices.items():
        subset_y = df.iloc[idx_list][TARGET_COL]
        n_pos = int((subset_y == 1).sum())
        n_neg = int((subset_y == 0).sum())
        n_tot = len(subset_y)
        pct_pos = round(n_pos / n_tot * 100, 2) if n_tot > 0 else 0

        profile["splits"][split_name] = {
            "count": n_tot,
            "positives": n_pos,
            "negatives": n_neg,
            "prevalence_pct": pct_pos
        }

        # Issue scientific warning if positive count is precariously low
        if n_pos < 10:
            msg = (
                f"[{cohort_name} - {split_name}] Warning: Small positive sample count (N={n_pos}). "
                f"Single-instance misclassifications will swing recall by {100.0/n_pos:.1f}%. "
                f"Repeated stratified cross-validation is strongly recommended for development."
            )
            warnings.warn(msg, UserWarning)
            profile["splits"][split_name]["statistical_warning"] = msg

    return profile


def generate_and_save_all_splits(seed: int = DEFAULT_RANDOM_SEED) -> Dict:
    """
    Generate, profile, and save all split manifests for:
    1. Cohort A (Clinical-only, N=531)
    2. Cohort B (Multimodal, N=270)
    """
    print(f"Loading raw datasets for split generation (seed={seed})...")
    clin_path = DATASET_DIR / "BC_cardiotox_clinical_variables.csv"
    func_path = DATASET_DIR / "BC_cardiotox_functional_variable.csv"
    comb_path = DATASET_DIR / "BC_cardiotox_clinical_and_functional_variables.csv"

    df_clin = pd.read_csv(clin_path, sep=';', decimal=',')
    df_comb = pd.read_csv(comb_path, sep=';', decimal=',')
    df_func = pd.read_csv(func_path, sep=';', decimal=',')

    # Cohort A: 531 clinical patients
    print("Generating Cohort A (Clinical-only, N=531) splits...")
    splits_clinical = create_stratified_train_val_test_split(df_clin, seed=seed)
    kfolds_clinical = create_stratified_kfold_splits(df_clin, seed=seed)
    profile_clinical = profile_split(df_clin, splits_clinical, "Cohort A (Clinical)")

    # Cohort B: 270 multimodal patients (rows 0..269 of combined dataset)
    print("Generating Cohort B (Multimodal, N=270) splits...")
    df_multi = df_comb.iloc[:270].copy()
    assert (df_multi[TARGET_COL].values == df_func[TARGET_COL].values).all(), "Multimodal target misalignment!"

    splits_multimodal = create_stratified_train_val_test_split(df_multi, seed=seed)
    kfolds_multimodal = create_stratified_kfold_splits(df_multi, seed=seed)
    profile_multimodal = profile_split(df_multi, splits_multimodal, "Cohort B (Multimodal)")

    # Save manifests
    # Cohort A record-level assignments
    rec_assignments_a = {}
    for part, idxs in splits_clinical.items():
        for i in idxs:
            rec_assignments_a[f"REC_CLIN_{i:03d}"] = {
                "source_row": int(i),
                "assigned_partition": part,
                "CTRCD": int(df_clin[TARGET_COL].iloc[i]),
                "patient_id": None
            }

    manifest_clinical = {
        "cohort": "clinical_only",
        "description": "531 breast cancer patients with 25 baseline clinical features",
        "total_records": 531,
        "seed": seed,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_generation_method": "StratifiedShuffleSplit (two-stage)",
        "cross_validation_method": "StratifiedKFold (5-fold)",
        "patient_identifier_status": "anonymized_technical_reference",
        "patient_level_independence_verified": False,
        "patient_identifier_limitation": "Patient identifiers were removed during clinical ethical anonymization. Partitions guarantee technical row isolation but cannot verify absence of repeat examinations.",
        "profiles": profile_clinical,
        "indices": splits_clinical,
        "kfold_indices": kfolds_clinical,
        "record_assignments": rec_assignments_a
    }
    clin_manifest_path = SPLITS_DIR / "cohort_a_clinical_splits.json"
    with open(clin_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_clinical, f, indent=2)

    # Cohort B record-level assignments
    rec_assignments_b = {}
    for part, idxs in splits_multimodal.items():
        for i in idxs:
            rec_assignments_b[f"REC_MULTI_{i:03d}"] = {
                "source_row": int(i),
                "assigned_partition": part,
                "CTRCD": int(df_multi[TARGET_COL].iloc[i]),
                "patient_id": None
            }

    manifest_multimodal = {
        "cohort": "multimodal",
        "description": "270 breast cancer patients with paired baseline clinical and 1001-point TDI functional waveforms",
        "total_records": 270,
        "seed": seed,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_generation_method": "StratifiedShuffleSplit (two-stage)",
        "cross_validation_method": "StratifiedKFold (5-fold)",
        "patient_identifier_status": "anonymized_technical_reference",
        "patient_level_independence_verified": False,
        "patient_identifier_limitation": "Patient identifiers were removed during clinical ethical anonymization. Partitions guarantee technical row isolation but cannot verify absence of repeat examinations.",
        "profiles": profile_multimodal,
        "indices": splits_multimodal,
        "kfold_indices": kfolds_multimodal,
        "record_assignments": rec_assignments_b
    }
    multi_manifest_path = SPLITS_DIR / "cohort_b_multimodal_splits.json"
    with open(multi_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_multimodal, f, indent=2)

    # Save data configuration in configs/
    data_config = {
        "project": "Interpretable Multimodal Framework for Cardiotoxicity Risk Assessment",
        "schema_version": "1.0.0",
        "random_seed": seed,
        "patient_identifier_status": "anonymized_technical_reference",
        "patient_level_independence_verified": False,
        "patient_identifier_limitation": "Technical row-level separation does not establish patient-level independence due to medical record anonymization.",
        "cohorts": {
            "cohort_a_clinical": {
                "source": "dataset/BC_cardiotox_clinical_variables.csv",
                "total_samples": 531,
                "positive_samples": 54,
                "split_manifest": "data/splits/cohort_a_clinical_splits.json"
            },
            "cohort_b_multimodal": {
                "source": "dataset/BC_cardiotox_clinical_and_functional_variables.csv (rows 0..269)",
                "total_samples": 270,
                "positive_samples": 27,
                "split_manifest": "data/splits/cohort_b_multimodal_splits.json"
            }
        },
        "target": TARGET_COL,
        "time_to_event": "time",
        "evaluation_metrics": [
            "PR-AUC (Average Precision)",
            "ROC-AUC",
            "Sensitivity / Recall",
            "Specificity",
            "F1-Score",
            "Balanced Accuracy"
        ]
    }
    config_path = CONFIG_DIR / "data_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)

    print(f"Splits generated successfully:\n  - {clin_manifest_path}\n  - {multi_manifest_path}\n  - {config_path}")

    return {
        "clinical_profile": profile_clinical,
        "multimodal_profile": profile_multimodal,
        "manifest_clinical": str(clin_manifest_path),
        "manifest_multimodal": str(multi_manifest_path),
        "config_path": str(config_path)
    }


if __name__ == "__main__":
    generate_and_save_all_splits()
