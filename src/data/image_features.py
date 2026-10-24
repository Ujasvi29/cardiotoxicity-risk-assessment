"""Image loading and identity-preserving feature export for Module 9."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from src.data.image_preprocessing import preprocess_single_image
from src.models.efficientnet_encoder import EfficientNetB0Encoder
from src.utils.paths import DATA_ROOT_ENV

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tensor(image_path: Path, *, crop_roi: bool = True, image_size: int = 224) -> torch.Tensor:
    """Use the established TDI preprocessing and apply ImageNet normalization."""
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    array = preprocess_single_image(image_path, crop_roi=crop_roi, target_size=(image_size, image_size), convert_to_rgb=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Preprocessor must return an HWC RGB image, got {array.shape}")
    normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def extract_image_features(
    encoder: EfficientNetB0Encoder,
    image_paths: Iterable[Path],
    *,
    batch_size: int = 16,
    crop_roi: bool = True,
    image_size: int = 224,
    output_dir: Optional[Path] = None,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Extract ordered features; optional NPY/JSON export refuses overwrites.

    Image identities are source-relative paths where possible. No patient or
    clinical labels are inferred or attached.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    paths = sorted((Path(path) for path in image_paths), key=lambda path: path.as_posix())
    if not paths:
        raise ValueError("image_paths must not be empty")
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(paths):
        raise ValueError("image_paths must not contain duplicates")

    rows = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = torch.stack([load_image_tensor(path, crop_roi=crop_roi, image_size=image_size) for path in batch_paths])
        rows.append(encoder.extract(batch).cpu().numpy())
    features = np.concatenate(rows, axis=0)
    try:
        identity_root = Path(os.path.commonpath([str(path.parent) for path in resolved_paths]))
    except ValueError:  # Different Windows drives: retain each explicit path as the identifier.
        identity_root = None
    manifest = [
        {
            "image_id": path.relative_to(identity_root).as_posix() if identity_root else path.as_posix(),
            "source_filename": path.name,
            "source_path": str(path),
        }
        for path in resolved_paths
    ]

    if output_dir is not None:
        output_dir = Path(output_dir)
        feature_path = output_dir / "image_features.npy"
        manifest_path = output_dir / "image_features_manifest.json"
        if feature_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Feature output already exists in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features, allow_pickle=False)
        manifest_path.write_text(json.dumps({
            "feature_file": feature_path.name,
            "feature_shape": list(features.shape),
            "feature_dimension": encoder.output_dim,
            "weights_status": encoder.weights_status,
            "projection_status": "random_initialized" if encoder.output_dim != encoder.feature_dim else "not_configured",
            "crop_roi": crop_roi,
            "image_size": image_size,
            "normalization": "ImageNet mean/std after existing preprocessing pixel scaling",
            "images": manifest,
            "label_status": "not attached; no verified image-to-record mapping available",
        }, indent=2), encoding="utf-8")
    return features, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unlabeled EfficientNet-B0 features from a directory of images.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--pretrained", action="store_true", help="Load cached/downloadable ImageNet weights.")
    parser.add_argument("--allow-untrained-fallback", action="store_true", help="Explicitly permit random weights if pretrained weights cannot be loaded.")
    parser.add_argument("--no-roi-crop", action="store_true", help="Resize the full image rather than applying the existing TDI spectral ROI crop.")
    args = parser.parse_args()
    from src.models.efficientnet_encoder import EncoderConfig

    if not args.image_dir.is_dir():
        parser.error(
            f"Image directory does not exist: {args.image_dir}. Pass the existing IMAGES directory, "
            f"or set {DATA_ROOT_ENV} to the dataset root and pass its IMAGES subdirectory."
        )
    paths = sorted((p for p in args.image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}), key=lambda p: p.name)
    if not paths:
        parser.error(f"No supported image files found in {args.image_dir}")
    encoder = EfficientNetB0Encoder(EncoderConfig(
        pretrained=args.pretrained,
        allow_untrained_fallback=args.allow_untrained_fallback,
        device=args.device,
    ))
    features, _ = extract_image_features(
        encoder, paths, batch_size=args.batch_size, crop_roi=not args.no_roi_crop, output_dir=args.output_dir,
    )
    print(f"Saved {len(paths)} unlabeled image vectors with shape {features.shape}; weights={encoder.weights_status}")


if __name__ == "__main__":
    main()
