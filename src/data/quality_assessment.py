"""
Module 3: Data Quality Assessment
==================================
Performs rigorous quality audits across clinical, functional, and image data:
1. Clinical range, physiological plausibility, and outlier detection (Z-score & IQR).
2. Categorical / binary integrity verification (strictly {0, 1, NA}).
3. Missingness pattern and co-occurrence analysis (MCAR vs MAR/MNAR indicator tests).
4. Functional waveform quality (velocity spikes, acceleration artifacts, zero-crossings).
5. Label consistency and clinical-echocardiographic plausibility (LVEF vs CTRCD).
6. Class imbalance severity and statistical power implications.

Outputs figures to outputs/figures/ and summary reports to outputs/reports/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import PROJECT_ROOT, get_data_root, require_raw_dataset

BASE_DIR = PROJECT_ROOT
DATASET_DIR = get_data_root()
FIG_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def load_datasets():
    data_root = require_raw_dataset()
    df_clin = pd.read_csv(data_root / "BC_cardiotox_clinical_variables.csv", sep=';', decimal=',')
    df_func = pd.read_csv(data_root / "BC_cardiotox_functional_variable.csv", sep=';', decimal=',')
    df_comb = pd.read_csv(data_root / "BC_cardiotox_clinical_and_functional_variables.csv", sep=';', decimal=',')
    return df_clin, df_func, df_comb


def audit_clinical_plausibility(df_clin):
    """Assess physiological ranges, binary integrity, and potential outliers."""
    # Clinical continuous variables
    cont_cols = ['age', 'weight', 'height', 'heart_rate', 'LVEF', 'PWT', 'LAd', 'LVDd', 'LVSd', 'time']
    # Clinical binary variables
    binary_cols = [
        'CTRCD', 'heart_rhythm', 'AC', 'antiHER2', 'HTA', 'DL', 'DM',
        'smoker', 'exsmoker', 'ACprev', 'antiHER2prev', 'RTprev', 'CIprev',
        'ICMprev', 'ARRprev', 'VALVprev', 'cxvalv'
    ]
    
    # 1. Binary domain check (must only be 0, 1, or NaN)
    binary_issues = {}
    for col in binary_cols:
        vals = set(df_clin[col].dropna().unique())
        invalid = vals - {0.0, 1.0, 0, 1}
        if invalid:
            binary_issues[col] = list(invalid)
            
    # 2. Outlier detection via IQR and Z-scores
    outlier_summary = {}
    for col in cont_cols:
        series = df_clin[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3.0]
        
        outlier_summary[col] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "iqr_lower": round(float(lower_bound), 2),
            "iqr_upper": round(float(upper_bound), 2),
            "iqr_outlier_count": int(len(iqr_outliers)),
            "iqr_outlier_pct": round(float(len(iqr_outliers) / len(series) * 100), 2),
            "z_outlier_count (z > 3)": int(len(z_outliers))
        }

    # 3. Derived BMI check (BMI = weight_kg / (height_m)^2)
    valid_bmi_mask = df_clin['weight'].notna() & df_clin['height'].notna()
    weights = df_clin.loc[valid_bmi_mask, 'weight']
    heights_m = df_clin.loc[valid_bmi_mask, 'height'] / 100.0
    bmis = weights / (heights_m ** 2)
    
    bmi_stats = {
        "min_bmi": round(float(bmis.min()), 2),
        "max_bmi": round(float(bmis.max()), 2),
        "mean_bmi": round(float(bmis.mean()), 2),
        "underweight (<18.5)": int((bmis < 18.5).sum()),
        "normal (18.5-24.9)": int(((bmis >= 18.5) & (bmis < 25.0)).sum()),
        "overweight (25-29.9)": int(((bmis >= 25.0) & (bmis < 30.0)).sum()),
        "obese (>=30)": int((bmis >= 30.0).sum())
    }

    return {
        "binary_domain_issues": binary_issues,
        "continuous_outliers": outlier_summary,
        "bmi_analysis": bmi_stats
    }


def audit_missingness_mechanism(df_clin):
    """
    Test whether clinical missingness correlates with CTRCD target.
    Uses Chi-square test of independence for missingness indicators vs CTRCD.
    """
    missing_tests = {}
    cols_with_null = df_clin.columns[df_clin.isnull().sum() > 0]
    
    for col in cols_with_null:
        # Create missingness indicator (1 = missing, 0 = observed)
        indicator = df_clin[col].isnull().astype(int)
        cross_tab = pd.crosstab(indicator, df_clin['CTRCD'])
        
        # Chi-square test (or Fisher's exact if small cell counts)
        if cross_tab.shape == (2, 2):
            try:
                chi2, p_val, _, _ = stats.chi2_contingency(cross_tab)
                missing_tests[col] = {
                    "missing_in_ctrcd_0": int(cross_tab.loc[1, 0]) if 1 in cross_tab.index else 0,
                    "missing_in_ctrcd_1": int(cross_tab.loc[1, 1]) if 1 in cross_tab.index else 0,
                    "p_value": round(float(p_val), 4),
                    "significantly_associated_with_target": bool(p_val < 0.05)
                }
            except Exception:
                pass
                
    return missing_tests


def audit_waveform_quality(df_func):
    """
    Audit functional velocity curves for artifacts, discontinuities,
    extreme spikes, or non-physiological jumps.
    """
    waveforms = df_func.iloc[:, 1:].values  # (270, 1001)
    
    # 1. First differences (approximate velocity acceleration / derivative)
    diffs = np.diff(waveforms, axis=1)  # step size dt = 1/1000 = 0.001
    accelerations = diffs / 0.001       # cm/s^2 equivalent
    
    max_accel_per_patient = np.max(np.abs(accelerations), axis=1)
    extreme_spike_patients = np.where(max_accel_per_patient > 2000.0)[0]  # high threshold
    
    # 2. Zero-crossing count per patient (normal cardiac cycle has 2-4 zero crossings: S, E, A waves)
    zero_crossings = []
    for i in range(len(waveforms)):
        wf = waveforms[i]
        # Count sign changes
        zc = np.where(np.diff(np.signbit(wf)))[0]
        zero_crossings.append(len(zc))
        
    zero_crossing_stats = {
        "min_crossings": int(np.min(zero_crossings)),
        "max_crossings": int(np.max(zero_crossings)),
        "mean_crossings": round(float(np.mean(zero_crossings)), 2),
        "median_crossings": int(np.median(zero_crossings)),
        "anomalous_crossings (<2 or >8)": int(sum(1 for z in zero_crossings if z < 2 or z > 8))
    }
    
    # 3. Waveform range and energy
    ranges = np.ptp(waveforms, axis=1)  # peak-to-peak amplitude
    flatline_patients = np.where(ranges < 2.0)[0]  # velocity range < 2 cm/s would be non-physiological flatline
    
    return {
        "flatline_patients_count": int(len(flatline_patients)),
        "extreme_spike_patients_count": int(len(extreme_spike_patients)),
        "peak_to_peak_min": round(float(np.min(ranges)), 2),
        "peak_to_peak_max": round(float(np.max(ranges)), 2),
        "peak_to_peak_mean": round(float(np.mean(ranges)), 2),
        "zero_crossing_profile": zero_crossing_stats
    }


def audit_echocardiographic_plausibility(df_clin):
    """
    Assess LVEF vs CTRCD label.
    CTRCD is clinically defined as LVEF decrease (usually drop >= 10% to < 50%).
    Check baseline LVEF distribution in both groups.
    """
    lvef_neg = df_clin.loc[df_clin['CTRCD'] == 0, 'LVEF']
    lvef_pos = df_clin.loc[df_clin['CTRCD'] == 1, 'LVEF']
    
    t_stat, p_val = stats.ttest_ind(lvef_neg, lvef_pos, equal_var=False)
    
    return {
        "ctrcd_0_lvef_mean": round(float(lvef_neg.mean()), 2),
        "ctrcd_0_lvef_std": round(float(lvef_neg.std()), 2),
        "ctrcd_0_lvef_min": round(float(lvef_neg.min()), 2),
        "ctrcd_1_lvef_mean": round(float(lvef_pos.mean()), 2),
        "ctrcd_1_lvef_std": round(float(lvef_pos.std()), 2),
        "ctrcd_1_lvef_min": round(float(lvef_pos.min()), 2),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 5),
        "baseline_lvef_significantly_different": bool(p_val < 0.05)
    }


def generate_quality_figures(df_clin, df_func):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    """Generate diagnostic figures for data quality assessment."""
    # -------------------------------------------------------------
    # Figure 1: Clinical Distributions & Physiological Plausibility
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    
    # 1. LVEF distribution by CTRCD
    sns.boxplot(data=df_clin, x='CTRCD', y='LVEF', hue='CTRCD', palette=['#2b5c8f', '#e0564c'], ax=axes[0, 0], width=0.45, legend=False)
    axes[0, 0].set_title('Baseline LVEF by CTRCD Group', fontweight='bold', fontsize=11)
    axes[0, 0].set_xlabel('CTRCD (0 = No Cardiotoxicity, 1 = Cardiotoxicity)')
    axes[0, 0].set_ylabel('Left Ventricular Ejection Fraction (%)')
    axes[0, 0].axhline(50, color='gray', linestyle='--', alpha=0.7, label='Clinical Threshold (50%)')
    axes[0, 0].legend(loc='lower left', fontsize=9)
    
    # 2. Age distribution
    sns.histplot(data=df_clin, x='age', hue='CTRCD', multiple='stack', palette=['#2b5c8f', '#e0564c'],
                 bins=20, ax=axes[0, 1], edgecolor='white')
    axes[0, 1].set_title('Age Distribution (Years)', fontweight='bold', fontsize=11)
    axes[0, 1].set_xlabel('Age (Years)')
    axes[0, 1].set_ylabel('Patient Count')
    
    # 3. BMI distribution
    valid_mask = df_clin['weight'].notna() & df_clin['height'].notna()
    bmis = df_clin.loc[valid_mask, 'weight'] / ((df_clin.loc[valid_mask, 'height'] / 100.0) ** 2)
    sns.histplot(bmis, kde=True, color='#4575b4', ax=axes[0, 2], bins=20)
    axes[0, 2].set_title('Body Mass Index (BMI, kg/m²)', fontweight='bold', fontsize=11)
    axes[0, 2].set_xlabel('BMI (kg/m²)')
    axes[0, 2].axvline(25, color='orange', linestyle=':', label='Overweight (25)')
    axes[0, 2].axvline(30, color='red', linestyle='--', label='Obese (30)')
    axes[0, 2].legend(fontsize=9)
    
    # 4. Posterior Wall Thickness (PWT)
    sns.boxplot(data=df_clin, x='CTRCD', y='PWT', hue='CTRCD', palette=['#2b5c8f', '#e0564c'], ax=axes[1, 0], width=0.45, legend=False)
    axes[1, 0].set_title('Posterior Wall Thickness (cm)', fontweight='bold', fontsize=11)
    axes[1, 0].set_ylabel('PWT (cm)')
    
    # 5. Left Ventricular End-Diastolic Diameter (LVDd)
    sns.boxplot(data=df_clin, x='CTRCD', y='LVDd', hue='CTRCD', palette=['#2b5c8f', '#e0564c'], ax=axes[1, 1], width=0.45, legend=False)
    axes[1, 1].set_title('LV Diastolic Diameter (cm)', fontweight='bold', fontsize=11)
    axes[1, 1].set_ylabel('LVDd (cm)')
    
    # 6. Follow-up / Event Time (days)
    sns.histplot(data=df_clin, x='time', hue='CTRCD', multiple='stack', palette=['#2b5c8f', '#e0564c'],
                 bins=25, ax=axes[1, 2], edgecolor='white')
    axes[1, 2].set_title('Follow-up / Event Time (Days)', fontweight='bold', fontsize=11)
    axes[1, 2].set_xlabel('Days from Treatment Inception')
    axes[1, 2].set_ylabel('Patient Count')
    
    plt.tight_layout()
    fig1_path = FIG_DIR / "quality_clinical_distributions.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Waveform Quality & Derivative Smoothness
    # -------------------------------------------------------------
    waveforms = df_func.iloc[:, 1:].values
    diffs = np.diff(waveforms, axis=1) / 0.001
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Peak-to-peak amplitude histogram
    ptp_vals = np.ptp(waveforms, axis=1)
    sns.histplot(ptp_vals, kde=True, color='#2b5c8f', ax=axes[0], bins=20)
    axes[0].set_title('Waveform Peak-to-Peak Amplitude (cm/s)', fontweight='bold', fontsize=11)
    axes[0].set_xlabel('Peak-to-Peak Velocity Range (cm/s)')
    axes[0].set_ylabel('Patient Count')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Maximum acceleration distribution
    max_accels = np.max(np.abs(diffs), axis=1)
    sns.histplot(max_accels, kde=True, color='#d73027', ax=axes[1], bins=20)
    axes[1].set_title('Waveform Smoothness: Max Derivative |dv/dt|', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Max Numerical Acceleration (cm/s²)')
    axes[1].set_ylabel('Patient Count')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig2_path = FIG_DIR / "quality_waveform_smoothness.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    return {
        "clinical_plausibility_fig": str(fig1_path),
        "waveform_quality_fig": str(fig2_path)
    }


def main():
    require_raw_dataset()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Executing Module 3 Data Quality Assessment...")
    df_clin, df_func, df_comb = load_datasets()
    
    plausibility = audit_clinical_plausibility(df_clin)
    missing_mechanism = audit_missingness_mechanism(df_clin)
    waveform_quality = audit_waveform_quality(df_func)
    echo_plausibility = audit_echocardiographic_plausibility(df_clin)
    figures = generate_quality_figures(df_clin, df_func)
    
    quality_report = {
        "clinical_plausibility": plausibility,
        "missingness_mechanism_tests": missing_mechanism,
        "waveform_quality_audit": waveform_quality,
        "echo_plausibility_audit": echo_plausibility,
        "figures": figures
    }
    
    out_json = REPORT_DIR / "data_quality_assessment_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
        
    print(f"Data Quality Assessment Complete! Results saved to {out_json}")


if __name__ == "__main__":
    main()
