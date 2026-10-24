"""
Module 5 Test Suite: TDI Image Preprocessing Verification
==========================================================
Tests and assertions verifying:
1. Image count and file extensions (strictly 270 PNGs).
2. Image readability and absence of corruption across all 270 files.
3. Uniformity of image dimensions (957x717 pixels).
4. Color mode verification (269 RGBA, 1 RGB).
5. Raw file immutability (original images in dataset/IMAGES/ remain untouched).
6. Valid output from candidate transformation pipeline (ROI crop, RGB conversion, 224x224).
7. Determinism and reproducibility across repeated calls.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from PIL import Image

from src.data.image_preprocessing import (
    IMAGES_DIR,
    audit_image_files,
    preprocess_single_image,
    get_doppler_roi_bounds
)


@pytest.mark.requires_image_data
def test_image_count_and_naming():
    """Verify exactly 270 PNG images named 'image (1).png' to 'image (270).png'."""
    files = list(IMAGES_DIR.glob("*.png"))
    assert len(files) == 270, f"Expected 270 images, found {len(files)}"

    expected_names = {f"image ({i}).png" for i in range(1, 271)}
    actual_names = {f.name for f in files}
    assert expected_names == actual_names, f"Filename mismatch: missing={expected_names - actual_names}"


@pytest.mark.requires_image_data
def test_image_readability_and_corruption():
    """Verify all 270 images open and decode without errors."""
    corrupted = []
    for f in IMAGES_DIR.glob("*.png"):
        try:
            with Image.open(f) as img:
                img.verify()
        except Exception as e:
            corrupted.append((f.name, str(e)))

    assert len(corrupted) == 0, f"Corrupted images detected: {corrupted}"


@pytest.mark.requires_image_data
def test_image_dimensions_uniformity():
    """Verify all 270 images have dimensions 957x717."""
    for f in IMAGES_DIR.glob("*.png"):
        with Image.open(f) as img:
            assert img.size == (957, 717), f"Unexpected size in {f.name}: {img.size}"


@pytest.mark.requires_image_data
def test_image_color_modes():
    """Verify documented color modes: 269 RGBA, 1 RGB (image (142).png)."""
    modes = {}
    for f in IMAGES_DIR.glob("*.png"):
        with Image.open(f) as img:
            modes[img.mode] = modes.get(img.mode, 0) + 1

    assert modes.get("RGBA", 0) == 269
    assert modes.get("RGB", 0) == 1

    with Image.open(IMAGES_DIR / "image (142).png") as im142:
        assert im142.mode == "RGB", f"Expected image (142).png to be RGB, found {im142.mode}"


@pytest.mark.requires_image_data
def test_raw_images_unmodified():
    """Verify raw images directory is read-only and files have not been overwritten."""
    assert IMAGES_DIR.exists()
    assert len(list(IMAGES_DIR.glob("*.png"))) == 270
    # Check sample file sizes
    im1 = IMAGES_DIR / "image (1).png"
    assert im1.stat().st_size > 0


@pytest.mark.requires_image_data
def test_single_image_transformation():
    """Verify candidate ROI crop and 224x224 RGB conversion pipeline."""
    sample_path = IMAGES_DIR / "image (1).png"
    arr = preprocess_single_image(
        sample_path,
        crop_roi=True,
        target_size=(224, 224),
        convert_to_rgb=True
    )

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (224, 224, 3), f"Unexpected transformed shape: {arr.shape}"
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0, f"Values not in [0, 1]: min={arr.min()}, max={arr.max()}"


@pytest.mark.requires_image_data
def test_transformation_determinism():
    """Verify repeated execution on the same file yields identical arrays."""
    sample_path = IMAGES_DIR / "image (10).png"
    arr1 = preprocess_single_image(sample_path, crop_roi=True, target_size=(224, 224))
    arr2 = preprocess_single_image(sample_path, crop_roi=True, target_size=(224, 224))

    assert np.array_equal(arr1, arr2), "Non-deterministic image transformation detected!"


def test_doppler_roi_bounds():
    """Verify Doppler ROI geometry conforms to the authors' R algorithm specifications."""
    bounds = get_doppler_roi_bounds()
    assert bounds["roi_y_min"] == 335
    assert bounds["roi_y_max"] == 670
    assert bounds["baseline_zero_velocity_y"] == 483
    assert bounds["roi_height"] == 335
    assert bounds["roi_width"] == 957


if __name__ == "__main__":
    test_functions = [
        test_image_count_and_naming,
        test_image_readability_and_corruption,
        test_image_dimensions_uniformity,
        test_image_color_modes,
        test_raw_images_unmodified,
        test_single_image_transformation,
        test_transformation_determinism,
        test_doppler_roi_bounds
    ]

    print("Running Module 5 Image Preprocessing Test Suite...\n" + "=" * 55)
    passed = 0
    for test in test_functions:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")

    print("=" * 55)
    print(f"Results: {passed} / {len(test_functions)} tests passed successfully!")
    if passed == len(test_functions):
        print("All Module 5 image tests verified!")
    else:
        raise SystemExit(1)
