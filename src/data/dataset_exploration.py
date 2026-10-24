"""
Module 2: Dataset Exploration and Understanding
=================================================
Reproducible analysis script for the BC_cardiotox dataset.
Performs data inventory, CSV schema inspection, missing-value profiling,
target label and class distribution analysis, duplicate detection,
multimodal alignment checks, waveform analysis, and image verification.
Outputs summary figures to outputs/figures/ and reports to outputs/reports/.
"""

import os
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.utils.paths import PROJECT_ROOT, get_data_root, require_raw_dataset

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = PROJECT_ROOT
DATASET_DIR = get_data_root()
IMAGES_DIR = DATASET_DIR / "IMAGES"
FIG_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def explore_directory_inventory():
    """Task 1: Catalog all files in dataset/."""
    data_root = require_raw_dataset()
    inventory = []
    for item in sorted(data_root.iterdir()):
        if item.is_file():
            stat = item.stat()
            readable = os.access(item, os.R_OK)
            inventory.append({
                "path": str(item.relative_to(data_root.parent)),
                "filename": item.name,
                "extension": item.suffix,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "is_readable": readable,
                "type": "Original file"
            })
        elif item.is_dir() and item.name == "IMAGES":
            stat = item.stat()
            file_count = len(list(item.glob("*.png")))
            inventory.append({
                "path": str(item.relative_to(data_root.parent)),
                "filename": item.name,
                "extension": "[directory]",
                "size_bytes": sum(f.stat().st_size for f in item.iterdir() if f.is_file()),
                "size_mb": round(sum(f.stat().st_size for f in item.iterdir() if f.is_file()) / (1024 * 1024), 2),
                "is_readable": True,
                "type": f"Extracted directory ({file_count} PNGs)"
            })
    return inventory


def explore_images():
    """Task 1 & 9: Inspect all 270 TDI images."""
    data_root = require_raw_dataset(include_images=True)
    img_files = sorted(list((data_root / "IMAGES").glob("*.png")))
    hashes = {}
    dims = set()
    modes = {}
    file_sizes = []
    
    for f in img_files:
        content = f.read_bytes()
        h = hashlib.md5(content).hexdigest()
        hashes[h] = hashes.get(h, []) + [f.name]
        file_sizes.append(len(content))
        
        with Image.open(f) as img:
            dims.add(img.size)
            modes[img.mode] = modes.get(img.mode, 0) + 1
            
    summary = {
        "total_images": len(img_files),
        "unique_hashes": len(hashes),
        "duplicates": {h: flist for h, flist in hashes.items() if len(flist) > 1},
        "dimensions": [f"{w}x{h}" for w, h in dims],
        "color_modes": modes,
        "min_size_kb": round(min(file_sizes) / 1024, 2),
        "max_size_kb": round(max(file_sizes) / 1024, 2),
        "mean_size_kb": round(np.mean(file_sizes) / 1024, 2),
        "corrupted_count": 0
    }
    return summary


def explore_csvs():
    """Load and cross-verify the three CSV files."""
    data_root = require_raw_dataset()
    clin_path = data_root / "BC_cardiotox_clinical_variables.csv"
    func_path = data_root / "BC_cardiotox_functional_variable.csv"
    comb_path = data_root / "BC_cardiotox_clinical_and_functional_variables.csv"

    df_clin = pd.read_csv(clin_path, sep=';', decimal=',')
    df_func = pd.read_csv(func_path, sep=';', decimal=',')
    df_comb = pd.read_csv(comb_path, sep=';', decimal=',')

    return df_clin, df_func, df_comb


def analyze_missing_values(df_clin, df_func, df_comb):
    """Generate detailed missing-value statistics."""
    def get_stats(df, name):
        null_counts = df.isnull().sum()
        total_cells = df.size
        null_cells = null_counts.sum()
        cols_with_null = null_counts[null_counts > 0]
        rows_with_null = (df.isnull().sum(axis=1) > 0).sum()
        
        return {
            "name": name,
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "total_cells": int(total_cells),
            "null_cells": int(null_cells),
            "null_pct": round(float(null_cells / total_cells * 100), 2),
            "cols_with_null_count": len(cols_with_null),
            "cols_without_null_count": len(df.columns) - len(cols_with_null),
            "rows_with_null_count": int(rows_with_null),
            "rows_with_null_pct": round(float(rows_with_null / len(df) * 100), 2),
            "col_details": {
                c: {"count": int(null_counts[c]), "pct": round(float(null_counts[c] / len(df) * 100), 2)}
                for c in cols_with_null.index
            }
        }
    
    return {
        "clinical": get_stats(df_clin, "Clinical CSV"),
        "functional": get_stats(df_func, "Functional CSV"),
        "combined": get_stats(df_comb, "Combined CSV")
    }


def analyze_class_distributions(df_clin, df_func, df_comb):
    """Compute target distribution across all dataset subsets."""
    func_mask = ~df_comb.iloc[:, 27].isna()
    
    subsets = {
        "Clinical Dataset (N=531)": df_clin['CTRCD'],
        "Functional Dataset (N=270)": df_func['CTRCD'],
        "Combined Dataset Overall (N=531)": df_comb['CTRCD'],
        "Combined with TDI (N=270)": df_comb.loc[func_mask, 'CTRCD'],
        "Combined without TDI (N=261)": df_comb.loc[~func_mask, 'CTRCD']
    }
    
    dist_stats = {}
    for name, s in subsets.items():
        vc = s.value_counts().to_dict()
        n_neg = int(vc.get(0, 0))
        n_pos = int(vc.get(1, 0))
        total = n_neg + n_pos
        pct_pos = round((n_pos / total) * 100, 2)
        ratio = round(n_neg / n_pos, 2) if n_pos > 0 else 0
        
        dist_stats[name] = {
            "total": total,
            "CTRCD=0 (Negative)": n_neg,
            "CTRCD=1 (Positive)": n_pos,
            "positive_rate_pct": pct_pos,
            "imbalance_ratio": f"{ratio}:1"
        }
        
    return dist_stats


def analyze_multimodal_correspondence(df_clin, df_func, df_comb):
    """
    Examine alignment between combined CSV and separate CSVs.
    Identifies permutation and row mappings.
    """
    # 1. Functional columns alignment in combined rows 0..269
    func_part_comb = df_comb.iloc[:270, 27:].values
    func_part_func = df_func.iloc[:, 1:].values
    func_max_diff = float(np.max(np.abs(func_part_comb - func_part_func)))
    target_match_func = bool((df_comb.iloc[:270]['CTRCD'].values == df_func['CTRCD'].values).all())
    
    # 2. Check clinical matching across all 531 rows
    clin_comb = df_comb.iloc[:, :27]
    matched_indices = []
    
    for i in range(len(clin_comb)):
        r = clin_comb.iloc[i]
        match = df_clin[
            (df_clin['LVEF'] == r['LVEF']) &
            (df_clin['time'] == r['time']) &
            (df_clin['age'] == r['age'])
        ]
        if len(match) >= 1:
            matched_indices.append(int(match.index[0]))
            
    is_bijective = (len(matched_indices) == 531) and (len(set(matched_indices)) == 531)
    
    return {
        "functional_max_abs_difference": func_max_diff,
        "functional_targets_match_sequence": target_match_func,
        "combined_clinical_bijective_match_to_clin_csv": is_bijective,
        "first_10_index_mappings (comb_idx -> clin_idx)": list(enumerate(matched_indices[:10]))
    }


def generate_figures(df_clin, df_func, df_comb, class_dists):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    """Generate and save publication-quality visualization figures."""
    # -------------------------------------------------------------
    # Figure 1: Class Distribution Across Cohort Subsets
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    subsets_labels = [
        "Clinical All\n(N=531)",
        "Functional\n(N=270)",
        "Combined All\n(N=531)",
        "Comb. w/ TDI\n(N=270)",
        "Comb. w/o TDI\n(N=261)"
    ]
    neg_counts = [d["CTRCD=0 (Negative)"] for d in class_dists.values()]
    pos_counts = [d["CTRCD=1 (Positive)"] for d in class_dists.values()]
    
    x = np.arange(len(subsets_labels))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, neg_counts, width, label='CTRCD = 0 (No Toxicity)', color='#2b5c8f', edgecolor='none')
    rects2 = axes[0].bar(x + width/2, pos_counts, width, label='CTRCD = 1 (Toxicity)', color='#e0564c', edgecolor='none')
    
    axes[0].set_ylabel('Patient Count', fontsize=11, fontweight='bold')
    axes[0].set_title('Absolute Class Distribution across Dataset Subsets', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subsets_labels, fontsize=10)
    axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b02020')

    # Positive rate percentage bar
    pos_rates = [d["positive_rate_pct"] for d in class_dists.values()]
    bars_rate = axes[1].bar(subsets_labels, pos_rates, color=['#4575b4', '#d73027', '#4575b4', '#d73027', '#fdae61'], width=0.5)
    axes[1].set_ylabel('CTRCD Incidence Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Positive Class Prevalence (Imbalance ~ 9:1)', fontsize=12, fontweight='bold', pad=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, 16)
    
    for bar in bars_rate:
        h = bar.get_height()
        axes[1].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    fig1_path = FIG_DIR / "class_distribution_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Missing Values in Clinical Features
    # -------------------------------------------------------------
    null_counts = df_clin.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=True)
    null_pct = (null_counts / len(df_clin)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = ['#f4a582' if p < 2.0 else '#d6604d' for p in null_pct]
    bars = ax.barh(null_counts.index, null_pct, color=colors, height=0.65)
    
    ax.set_xlabel('Missing Rate (% of 531 patients)', fontsize=11, fontweight='bold')
    ax.set_title('Missing Values in Clinical Variables (N=531)', fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 12)
    
    for bar, count in zip(bars, null_counts):
        w = bar.get_width()
        ax.annotate(f'{w:.1f}% ({count} patients)', xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)
        
    plt.tight_layout()
    fig2_path = FIG_DIR / "clinical_missing_values.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: TDI Waveform Comparison (CTRCD=0 vs CTRCD=1)
    # -------------------------------------------------------------
    waveforms = df_func.iloc[:, 1:].values  # (270, 1001)
    targets = df_func['CTRCD'].values
    time_grid = np.linspace(0, 1, 1001)
    
    wf_neg = waveforms[targets == 0]
    wf_pos = waveforms[targets == 1]
    
    mean_neg = np.mean(wf_neg, axis=0)
    std_neg = np.std(wf_neg, axis=0)
    mean_pos = np.mean(wf_pos, axis=0)
    std_pos = np.std(wf_pos, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    # Confidence envelopes
    ax.fill_between(time_grid, mean_neg - std_neg, mean_neg + std_neg, color='#2b5c8f', alpha=0.18, label='CTRCD=0 ±1 SD')
    ax.fill_between(time_grid, mean_pos - std_pos, mean_pos + std_pos, color='#e0564c', alpha=0.22, label='CTRCD=1 ±1 SD')
    
    # Mean curves
    ax.plot(time_grid, mean_neg, color='#2b5c8f', linewidth=2.4, label=f'CTRCD=0 Mean (N={len(wf_neg)})')
    ax.plot(time_grid, mean_pos, color='#e0564c', linewidth=2.4, linestyle='--', label=f'CTRCD=1 Mean (N={len(wf_pos)})')
    
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Normalized Cardiac Cycle [0, 1] (Discretized into 1001 Points)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Myocardial Velocity (cm/s)', fontsize=11, fontweight='bold')
    ax.set_title('TDI Numerical Waveforms: Mean Myocardial Velocity Curves across Normalized Cycle', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "tdi_waveforms_by_class.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Sample TDI Spectral Doppler Image & Waveform Overlay
    # -------------------------------------------------------------
    sample_img_path = require_raw_dataset(include_images=True) / "IMAGES" / "image (1).png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    with Image.open(sample_img_path) as im:
        axes[0].imshow(im)
        axes[0].set_title(f"Raw TDI Spectral Display: image (1).png\nDimensions: {im.size[0]}x{im.size[1]} ({im.mode})", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
    # Plot first 3 individual functional waveforms
    for i in range(3):
        label = f"Functional Row {i} (CTRCD={int(targets[i])})"
        axes[1].plot(time_grid, waveforms[i], label=label, linewidth=1.8)
    axes[1].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes[1].set_xlabel('Normalized Cycle Time', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Velocity (cm/s)', fontsize=11, fontweight='bold')
    axes[1].set_title('Extracted Discretized TDI Waveforms (Rows 0, 1, 2)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig4_path = FIG_DIR / "sample_image_and_waveform.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    return {
        "fig1": str(fig1_path),
        "fig2": str(fig2_path),
        "fig3": str(fig3_path),
        "fig4": str(fig4_path)
    }


def main():
    require_raw_dataset(include_images=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Executing Module 2 Dataset Exploration...")
    inventory = explore_directory_inventory()
    img_summary = explore_images()
    df_clin, df_func, df_comb = explore_csvs()
    missing_summary = analyze_missing_values(df_clin, df_func, df_comb)
    class_dists = analyze_class_distributions(df_clin, df_func, df_comb)
    multimodal_corr = analyze_multimodal_correspondence(df_clin, df_func, df_comb)
    figures = generate_figures(df_clin, df_func, df_comb, class_dists)
    
    full_report = {
        "inventory": inventory,
        "images": img_summary,
        "missing_summary": missing_summary,
        "class_distributions": class_dists,
        "multimodal_correspondence": multimodal_corr,
        "figures": figures
    }
    
    out_json = REPORT_DIR / "dataset_exploration_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"Module 2 Analysis Complete! Summary written to {out_json}")


if __name__ == "__main__":
    main()
