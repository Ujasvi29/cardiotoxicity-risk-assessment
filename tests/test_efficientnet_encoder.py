"""Focused offline tests for Module 9 EfficientNet-B0 feature extraction."""

from pathlib import Path
import urllib.error

import numpy as np
import pytest
import torch
from torchvision.models import efficientnet_b0 as torchvision_efficientnet_b0
from PIL import Image

import src.models.efficientnet_encoder as encoder_module
from src.data.image_features import extract_image_features, load_image_tensor
from src.models.efficientnet_encoder import EncoderConfig, EfficientNetB0Encoder


@pytest.fixture(scope="module")
def encoder():
    return EfficientNetB0Encoder(EncoderConfig(device="cpu"))


def test_random_construction_and_batched_output(encoder):
    assert encoder.weights_status == "random_initialized"
    assert encoder.backbone.classifier.__class__.__name__ == "Identity"
    assert encoder.output_dim == 1280
    output = encoder.extract(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1280)
    assert torch.isfinite(output).all()


def test_single_and_repeated_inference(encoder):
    image = torch.randn(1, 3, 64, 64)
    first = encoder.extract(image)
    second = encoder.extract(image)
    assert first.shape == (1, 1280)
    torch.testing.assert_close(first, second)
    assert not first.requires_grad


@pytest.mark.parametrize("images", [torch.empty(0, 3, 64, 64), torch.randn(2, 1, 64, 64), torch.randn(3, 64, 64)])
def test_rejects_invalid_shapes(encoder, images):
    with pytest.raises(ValueError):
        encoder(images)


def test_rejects_wrong_types_and_nonfinite(encoder):
    with pytest.raises(TypeError):
        encoder([[1]])
    images = torch.randn(1, 3, 64, 64)
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        encoder(images)
    with pytest.raises(ValueError, match="at least 32"):
        encoder(torch.randn(1, 3, 16, 16))


def test_frozen_and_selective_fine_tuning():
    frozen = EfficientNetB0Encoder()
    assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())
    tuned = EfficientNetB0Encoder(EncoderConfig(trainable_blocks=1, projection_dim=64))
    assert tuned.output_dim == 64
    assert all(parameter.requires_grad for parameter in tuned.backbone.features[-1].parameters())
    assert any(parameter.requires_grad for parameter in tuned.projection.parameters())
    assert all(not parameter.requires_grad for parameter in tuned.backbone.features[0].parameters())


def test_pretrained_status_and_explicit_offline_fallback(monkeypatch):
    calls = []

    def unavailable_factory(*, weights):
        calls.append(weights)
        if weights is not None:
            raise urllib.error.URLError("offline test")
        return torchvision_efficientnet_b0(weights=None)

    monkeypatch.setattr(encoder_module, "efficientnet_b0", unavailable_factory)
    with pytest.raises(RuntimeError, match="allow_untrained_fallback=True"):
        EfficientNetB0Encoder(EncoderConfig(pretrained=True))
    assert calls == [encoder_module.EfficientNet_B0_Weights.DEFAULT]

    calls.clear()
    with pytest.warns(RuntimeWarning, match="using random initialization"):
        encoder = EfficientNetB0Encoder(EncoderConfig(pretrained=True, allow_untrained_fallback=True))
    assert calls == [encoder_module.EfficientNet_B0_Weights.DEFAULT, None]
    assert encoder.weights_status == "random_fallback_after_weight_error"


def test_pretrained_success_status_is_explicit(monkeypatch):
    observed = []

    def successful_factory(*, weights):
        observed.append(weights)
        return torchvision_efficientnet_b0(weights=None)

    monkeypatch.setattr(encoder_module, "efficientnet_b0", successful_factory)
    encoder = EfficientNetB0Encoder(EncoderConfig(pretrained=True))
    assert observed == [encoder_module.EfficientNet_B0_Weights.DEFAULT]
    assert encoder.weights_status == "imagenet_pretrained"


def test_model_configuration_errors_are_not_hidden_by_fallback(monkeypatch):
    def invalid_factory(*, weights):
        raise ValueError("invalid model configuration")

    monkeypatch.setattr(encoder_module, "efficientnet_b0", invalid_factory)
    with pytest.raises(ValueError, match="invalid model configuration"):
        EfficientNetB0Encoder(EncoderConfig(pretrained=True, allow_untrained_fallback=True))


def test_rejects_unsupported_device():
    with pytest.raises(ValueError, match="CPU or CUDA"):
        EfficientNetB0Encoder(EncoderConfig(device="mps"))
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA was requested"):
            EfficientNetB0Encoder(EncoderConfig(device="cuda"))


@pytest.mark.parametrize("kwargs", [{"trainable_blocks": -1}, {"trainable_blocks": 10}, {"projection_dim": 0}, {"pretrained": 1}])
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EfficientNetB0Encoder(EncoderConfig(**kwargs))


def test_preprocessing_integration_and_identity_preserving_export(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, color in (("b.png", (40, 80, 120)), ("a.png", (120, 80, 40))):
        Image.new("RGBA", (957, 717), color + (255,)).save(image_dir / name)
    tensor = load_image_tensor(image_dir / "a.png")
    assert tensor.shape == (3, 224, 224)
    from src.data.image_preprocessing import preprocess_single_image
    from src.data.image_features import IMAGENET_MEAN, IMAGENET_STD
    scaled = preprocess_single_image(image_dir / "a.png", crop_roi=True, target_size=(224, 224))
    expected = torch.from_numpy(((scaled - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())
    torch.testing.assert_close(tensor, expected)
    paths = sorted(image_dir.glob("*.png"), key=lambda p: p.name)
    paths.reverse()
    outdir = tmp_path / "export"
    features, manifest = extract_image_features(EfficientNetB0Encoder(), paths, batch_size=1, output_dir=outdir)
    assert features.shape == (2, 1280)
    assert [row["image_id"] for row in manifest] == ["a.png", "b.png"]
    assert np.load(outdir / "image_features.npy", allow_pickle=False).shape == (2, 1280)
    import json
    metadata = json.loads((outdir / "image_features_manifest.json").read_text(encoding="utf-8"))
    assert metadata["label_status"].startswith("not attached")
    with pytest.raises(FileExistsError):
        extract_image_features(EfficientNetB0Encoder(), paths, output_dir=outdir)


def test_grayscale_conversion_and_invalid_batch_size(tmp_path: Path, encoder):
    grayscale = tmp_path / "gray.png"
    Image.new("L", (957, 717), 128).save(grayscale)
    assert load_image_tensor(grayscale).shape == (3, 224, 224)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        extract_image_features(encoder, [grayscale], batch_size=0)
