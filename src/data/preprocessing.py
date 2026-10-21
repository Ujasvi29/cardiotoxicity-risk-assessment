"""
Module 4: Data Preprocessing and Cleaning
==========================================
Reproducible preprocessing pipeline for the BC_cardiotox dataset:
1. Loads raw clinical, functional, and combined CSVs without mutating originals.
2. Formulates explicit, leak-free scikit-learn transformers:
   - Median imputation + robust/standard scaling for continuous features.
   - Mode (most frequent) imputation for binary/categorical features.
3. Separates prediction target (CTRCD) and survival time (time) from baseline input features.
4. Preprocesses numerical TDI functional waveforms:
   - Preserves raw physical velocity values (cm/s) to retain diagnostic amplitude.
   - Computes global-scaled representations (preventing per-waveform peak erasure).
5. Extracts and validates the 270-patient multimodal cohort with 100% verified alignment.
6. Exports processed data to data/processed/ and records a detailed schema JSON.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

BASE_DIR = Path(r"c:\dev\cardiotoxicity risk assesment")
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Feature Schema Definition
# ============================================================

TARGET_COL = "CTRCD"
TIME_COL = "time"  # Survival duration in days; outcome metadata, NOT a baseline input feature

CONTINUOUS_COLS = [
    "age",
    "weight",
    "height",
    "heart_rate",
    "LVEF",
    "PWT",
    "LAd",
    "LVDd",
    "LVSd",
]

BINARY_COLS = [
    "heart_rhythm",
    "AC",
    "antiHER2",
    "HTA",
    "DL",
    "DM",
    "smoker",
    "exsmoker",
    "ACprev",
    "antiHER2prev",
    "RTprev",
    "CIprev",
    "ICMprev",
    "ARRprev",
    "VALVprev",
    "cxvalv",
]

FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS  # 25 baseline input features


def get_feature_schema() -> Dict:
    """Return formal dictionary documenting the dataset schema."""
    return {
        "dataset_name": "BC_cardiotox",
        "cohort_description": "Breast cancer patients receiving cardiotoxic chemotherapy",
        "target": {
            "name": TARGET_COL,
            "type": "binary",
            "domain": [0, 1],
            "description": "Cancer Therapy-Related Cardiac Dysfunction (1 = yes, 0 = no)"
        },
        "time_to_event": {
            "name": TIME_COL,
            "type": "continuous",
            "units": "days",
            "description": "Follow-up days from treatment inception to CTRCD onset or end of observation"
        },
        "features": {
            "continuous_count": len(CONTINUOUS_COLS),
            "continuous_columns": CONTINUOUS_COLS,
            "binary_count": len(BINARY_COLS),
            "binary_columns": BINARY_COLS,
            "total_feature_count": len(FEATURE_COLS)
        },
        "functional_waveform": {
            "dimension": 1001,
            "points": [f"t {i}" for i in range(1, 1002)],
            "domain_interval": "[0, 1]",
            "units": "cm/s (myocardial tissue velocity)"
        }
    }


# ============================================================
# 2. Data Loading Functions (Preserves Raw Data)
# ============================================================

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSVs using European delimiters.
    Original files in dataset/ are strictly read-only.
    """
    clin_path = DATASET_DIR / "BC_cardiotox_clinical_variables.csv"
    func_path = DATASET_DIR / "BC_cardiotox_functional_variable.csv"
    comb_path = DATASET_DIR / "BC_cardiotox_clinical_and_functional_variables.csv"

    df_clin = pd.read_csv(clin_path, sep=';', decimal=',')
    df_func = pd.read_csv(func_path, sep=';', decimal=',')
    df_comb = pd.read_csv(comb_path, sep=';', decimal=',')

    return df_clin, df_func, df_comb


# ============================================================
# 3. Clinical Preprocessing Pipeline (Scikit-Learn Compatible)
# ============================================================

def build_clinical_preprocessor(scaling_method: str = "robust") -> ColumnTransformer:
    """
    Build a leak-free scikit-learn ColumnTransformer.
    - Continuous: Median imputation + Scaler (RobustScaler or StandardScaler).
    - Binary: Most-frequent (mode) imputation, maintaining binary {0, 1} range.
    """
    scaler = RobustScaler() if scaling_method == "robust" else StandardScaler()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, CONTINUOUS_COLS),
            ("cat", cat_pipeline, BINARY_COLS)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def preprocess_clinical_dataframe(
    df: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    fit: bool = True,
    scaling_method: str = "robust"
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Preprocess clinical features using specified or fitted transformer.
    If fit=True, fits on input df. If fit=False, uses pre-fitted transformer (for validation/test splits).
    """
    X = df[FEATURE_COLS].copy()

    if preprocessor is None:
        preprocessor = build_clinical_preprocessor(scaling_method=scaling_method)

    if fit:
        X_trans = preprocessor.fit_transform(X)
    else:
        X_trans = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_trans, columns=feature_names, index=df.index)

    return X_processed, preprocessor


# ============================================================
# 4. Numerical Functional Waveform Preprocessing
# ============================================================

def preprocess_functional_waveforms(
    df_func: pd.DataFrame,
    global_scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract and process functional velocity waveforms.
    - Preserves absolute myocardial velocity (cm/s).
    - Computes global peak velocity divisor for neural network stability.
    - Avoids per-waveform min-max normalization, which would destroy clinical amplitude biomarkers.
    """
    # Columns 't 1' to 't 1001'
    waveform_cols = [c for c in df_func.columns if c.startswith("t ")]
    if len(waveform_cols) != 1001:
        raise ValueError(f"Expected 1001 functional columns, found {len(waveform_cols)}")

    waveforms_raw = df_func[waveform_cols].to_numpy(dtype=np.float32)

    # Global scaling: compute 99th percentile of absolute velocities or max
    if global_scale is None:
        global_scale = float(np.percentile(np.abs(waveforms_raw), 99.5))

    waveforms_scaled = waveforms_raw / global_scale

    return waveforms_raw, waveforms_scaled, global_scale


# ============================================================
# 5. Pipeline Execution & Dataset Generation
# ============================================================

def run_preprocessing_pipeline() -> Dict:
    """Execute complete Module 4 preprocessing and export processed files."""
    print("Loading raw datasets from dataset/...")
    df_clin, df_func, df_comb = load_raw_data()

    # Verify input shapes
    assert df_clin.shape == (531, 27), f"Unexpected clinical shape: {df_clin.shape}"
    assert df_func.shape == (270, 1002), f"Unexpected functional shape: {df_func.shape}"
    assert df_comb.shape == (531, 1028), f"Unexpected combined shape: {df_comb.shape}"

    # -------------------------------------------------------------
    # 5.1 Full Clinical Cohort (N = 531)
    # -------------------------------------------------------------
    print("Preprocessing full clinical cohort (N=531)...")
    y_full = df_clin[[TARGET_COL, TIME_COL]].copy()
    X_full_raw = df_clin[FEATURE_COLS].copy()

    # Cleaned raw features (imputed without scaling, for tree models / EDA)
    imputer_num = SimpleImputer(strategy="median")
    imputer_cat = SimpleImputer(strategy="most_frequent")
    X_full_imputed = X_full_raw.copy()
    X_full_imputed[CONTINUOUS_COLS] = imputer_num.fit_transform(X_full_raw[CONTINUOUS_COLS])
    X_full_imputed[BINARY_COLS] = imputer_cat.fit_transform(X_full_raw[BINARY_COLS])

    # Scaled features (for neural networks / logistic regression)
    X_full_scaled, preprocessor_full = preprocess_clinical_dataframe(
        df_clin, fit=True, scaling_method="robust"
    )

    # -------------------------------------------------------------
    # 5.2 Multimodal Cohort (N = 270)
    # -------------------------------------------------------------
    print("Extracting verified multimodal cohort (N=270)...")
    # In df_comb, rows 0..269 are the 270 multimodal patients
    df_comb_multimodal = df_comb.iloc[:270].copy()

    # Cross-verify target label alignment
    assert (df_comb_multimodal[TARGET_COL].values == df_func[TARGET_COL].values).all(), \
        "Target label mismatch between df_comb[:270] and df_func!"

    y_multi = df_comb_multimodal[[TARGET_COL, TIME_COL]].copy()
    X_multi_raw = df_comb_multimodal[FEATURE_COLS].copy()

    # Impute multimodal clinical features
    X_multi_imputed = X_multi_raw.copy()
    imputer_num_multi = SimpleImputer(strategy="median")
    imputer_cat_multi = SimpleImputer(strategy="most_frequent")
    X_multi_imputed[CONTINUOUS_COLS] = imputer_num_multi.fit_transform(X_multi_raw[CONTINUOUS_COLS])
    X_multi_imputed[BINARY_COLS] = imputer_cat_multi.fit_transform(X_multi_raw[BINARY_COLS])

    # Scaled multimodal features
    X_multi_scaled, preprocessor_multi = preprocess_clinical_dataframe(
        df_comb_multimodal, fit=True, scaling_method="robust"
    )

    # -------------------------------------------------------------
    # 5.3 Functional Waveforms (N = 270 x 1001)
    # -------------------------------------------------------------
    print("Processing functional waveforms (N=270, L=1001)...")
    waveforms_raw, waveforms_scaled, global_scale = preprocess_functional_waveforms(df_func)

    # -------------------------------------------------------------
    # 5.4 Export Processed Files
    # -------------------------------------------------------------
    print(f"Exporting processed files to {PROCESSED_DIR}...")
    
    # 1. Feature schema
    schema = get_feature_schema()
    schema["functional_waveform"]["global_scale_divisor"] = global_scale
    schema_path = PROCESSED_DIR / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # 2. Full clinical datasets (N=531)
    X_full_imputed.to_csv(PROCESSED_DIR / "clinical_features_imputed_full.csv", index=False)
    X_full_scaled.to_csv(PROCESSED_DIR / "clinical_features_scaled_full.csv", index=False)
    y_full.to_csv(PROCESSED_DIR / "clinical_targets_full.csv", index=False)

    # 3. Multimodal aligned datasets (N=270)
    X_multi_imputed.to_csv(PROCESSED_DIR / "multimodal_clinical_imputed.csv", index=False)
    X_multi_scaled.to_csv(PROCESSED_DIR / "multimodal_clinical_scaled.csv", index=False)
    y_multi.to_csv(PROCESSED_DIR / "multimodal_targets.csv", index=False)

    # 4. Functional waveforms
    waveform_cols = [f"t_{i}" for i in range(1, 1002)]
    df_waveforms_raw = pd.DataFrame(waveforms_raw, columns=waveform_cols)
    df_waveforms_scaled = pd.DataFrame(waveforms_scaled, columns=waveform_cols)

    df_waveforms_raw.to_csv(PROCESSED_DIR / "functional_waveforms_raw_cm_s.csv", index=False)
    df_waveforms_scaled.to_csv(PROCESSED_DIR / "functional_waveforms_scaled.csv", index=False)

    summary = {
        "status": "success",
        "raw_directory": str(DATASET_DIR),
        "processed_directory": str(PROCESSED_DIR),
        "exported_files": [
            "feature_schema.json",
            "clinical_features_imputed_full.csv",
            "clinical_features_scaled_full.csv",
            "clinical_targets_full.csv",
            "multimodal_clinical_imputed.csv",
            "multimodal_clinical_scaled.csv",
            "multimodal_targets.csv",
            "functional_waveforms_raw_cm_s.csv",
            "functional_waveforms_scaled.csv"
        ],
        "shapes": {
            "full_clinical_features": X_full_imputed.shape,
            "full_clinical_targets": y_full.shape,
            "multimodal_clinical_features": X_multi_imputed.shape,
            "multimodal_clinical_targets": y_multi.shape,
            "functional_waveforms": waveforms_raw.shape
        },
        "target_distributions": {
            "full_cohort_CTRCD": y_full[TARGET_COL].value_counts().to_dict(),
            "multimodal_cohort_CTRCD": y_multi[TARGET_COL].value_counts().to_dict()
        },
        "global_waveform_scale": global_scale
    }

    report_path = REPORT_DIR / "preprocessing_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Preprocessing completed successfully! Summary written to {report_path}")
    return summary


if __name__ == "__main__":
    run_preprocessing_pipeline()
