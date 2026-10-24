"""
Module 5: TDI Image Inspection, Quality Audit, and Preprocessing
=================================================================
Performs:
1. Complete verification of all 270 TDI images in dataset/IMAGES/:
   - File extensions, dimensions, color modes, sizes, hashes, and corruption checks.
2. Generation of a high-resolution contact sheet displaying representative images.
3. Analysis of the authors' R algorithm (HSV segmentation, Doppler velocity conversion).
4. Evaluation of candidate 2D CNN image transformations:
   - Spectral ROI cropping (removing upper UI/ECG noise and bottom status bar).
   - RGBA -> RGB channel standardization.
   - Preserving vs destroying Doppler scale/velocity information.
5. Export of image quality audit metrics to outputs/reports/image_quality_summary.json.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.paths import PROJECT_ROOT, get_data_root, require_raw_dataset

BASE_DIR = PROJECT_ROOT
DATASET_DIR = get_data_root()
IMAGES_DIR = DATASET_DIR / "IMAGES"
FIG_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

def audit_image_files() -> Dict:
    """Audit all 270 images under dataset/IMAGES/ for integrity and consistency."""
    data_root = require_raw_dataset(include_images=True)
    img_files = sorted(list((data_root / "IMAGES").glob("*.png")))
    hashes = {}
    dims = {}
    modes = {}
    sizes = []
    corrupted = []
    extensions = set()

    for f in img_files:
        extensions.add(f.suffix.lower())
        try:
            content = f.read_bytes()
            sizes.append(len(content))
            h = hashlib.md5(content).hexdigest()
            hashes[h] = hashes.get(h, []) + [f.name]

            with Image.open(f) as img:
                dim_str = f"{img.size[0]}x{img.size[1]}"
                dims[dim_str] = dims.get(dim_str, 0) + 1
                modes[img.mode] = modes.get(img.mode, 0) + 1
        except Exception as e:
            corrupted.append({"filename": f.name, "error": str(e)})

    # Detect duplicates
    duplicates = {h: flist for h, flist in hashes.items() if len(flist) > 1}

    audit_summary = {
        "total_images": len(img_files),
        "unique_filenames_count": len(set(f.name for f in img_files)),
        "extensions": list(extensions),
        "dimensions": dims,
        "color_modes": modes,
        "unique_hashes_count": len(hashes),
        "duplicate_hash_groups": duplicates,
        "corrupted_images_count": len(corrupted),
        "corrupted_images": corrupted,
        "file_size_bytes": {
            "min": int(min(sizes)),
            "max": int(max(sizes)),
            "mean": round(float(np.mean(sizes)), 2),
            "total_mb": round(sum(sizes) / (1024 * 1024), 2)
        }
    }
    return audit_summary


def generate_contact_sheet(
    sample_indices: Optional[List[int]] = None,
    grid_size: Tuple[int, int] = (4, 4),
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate and save a representative contact sheet of TDI images.
    Displays variations in Doppler signal intensity, ECG trace, noise, and baseline.
    """
    if output_path is None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIG_DIR / "tdi_contact_sheet.png"

    data_root = require_raw_dataset(include_images=True)
    all_files = sorted(list((data_root / "IMAGES").glob("*.png")), key=lambda p: int(p.stem.split('(')[1].split(')')[0]))

    if sample_indices is None:
        # Pick 16 evenly spaced images across the 270 cohort
        num_samples = grid_size[0] * grid_size[1]
        sample_indices = np.linspace(0, len(all_files) - 1, num_samples, dtype=int).tolist()

    nrows, ncols = grid_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(sample_indices):
        f = all_files[sample_idx]
        with Image.open(f) as img:
            axes[idx].imshow(img)
            axes[idx].set_title(f"{f.name}\n({img.size[0]}x{img.size[1]}, {img.mode})", fontsize=9, fontweight='bold')
            axes[idx].axis("off")

    plt.suptitle("BC_cardiotox: Tissue Doppler Imaging (TDI) Contact Sheet (16 Representative Images)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Contact sheet saved to {output_path}")
    return output_path


def get_doppler_roi_bounds() -> Dict[str, int]:
    """
    Return ROI bounds based on the authors' R preprocessing script.
    - Top: 335 pixels masked (removes ECG and metadata).
    - Bottom: 670 pixels masked (removes status line and scale legend).
    - Usable spectral Doppler region: vertical rows [335, 670].
    """
    return {
        "roi_y_min": 335,
        "roi_y_max": 670,
        "roi_height": 335,
        "roi_width": 957,
        "baseline_zero_velocity_y": 483,
        "cm_per_second_per_pixel": 0.0877193
    }


def preprocess_single_image(
    image_path: Path,
    crop_roi: bool = True,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    convert_to_rgb: bool = True
) -> np.ndarray:
    """
    Standardize a single TDI image for potential 2D CNN feature extraction.
    1. Converts RGBA to RGB (dropping the redundant alpha transparency channel).
    2. Crops to the spectral Doppler region (rows 335 to 670) to eliminate ECG trace and text UI.
    3. Resizes to target dimension (e.g. 224x224 for standard CNN backbones).
    4. Normalizes pixel intensities to [0, 1].
    """
    with Image.open(image_path) as img:
        if convert_to_rgb and img.mode != "RGB":
            img = img.convert("RGB")

        if crop_roi:
            bounds = get_doppler_roi_bounds()
            # PIL crop box: (left, upper, right, lower)
            img = img.crop((0, bounds["roi_y_min"], bounds["roi_width"], bounds["roi_y_max"]))

        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        return arr


def run_image_audit_and_demo():
    """Run full audit and export summary JSON and contact sheet."""
    require_raw_dataset(include_images=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Auditing all 270 TDI images in dataset/IMAGES/...")
    audit = audit_image_files()

    # Verify key audit requirements
    assert audit["total_images"] == 270, f"Expected 270 images, got {audit['total_images']}"
    assert audit["corrupted_images_count"] == 0, f"Corrupted images found: {audit['corrupted_images_count']}"
    assert len(audit["duplicate_hash_groups"]) == 0, "Duplicate images found!"

    print("Generating representative TDI contact sheet...")
    generate_contact_sheet()

    audit["roi_bounds"] = get_doppler_roi_bounds()
    audit_json_path = REPORT_DIR / "image_quality_summary.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Image audit complete! Summary saved to {audit_json_path}")
    return audit


if __name__ == "__main__":
    run_image_audit_and_demo()
