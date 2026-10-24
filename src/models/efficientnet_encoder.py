"""EfficientNet-B0 image feature extraction without a classification head."""

from __future__ import annotations

import socket
import urllib.error
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the EfficientNet-B0 feature encoder."""

    pretrained: bool = False
    allow_untrained_fallback: bool = False
    device: str = "cpu"
    trainable_blocks: int = 0
    projection_dim: Optional[int] = None


class EfficientNetB0Encoder(nn.Module):
    """Return EfficientNet-B0 pooled features, optionally projected.

    ``trainable_blocks`` selects the last N entries in ``model.features``;
    zero freezes the complete backbone. ``forward`` preserves gradients for
    configured trainable parameters. Use :meth:`extract` for pure inference.
    """

    feature_dim = 1280

    def __init__(self, config: EncoderConfig = EncoderConfig()) -> None:
        super().__init__()
        if not isinstance(config, EncoderConfig):
            raise TypeError("config must be an EncoderConfig")
        if not isinstance(config.pretrained, bool) or not isinstance(config.allow_untrained_fallback, bool):
            raise ValueError("pretrained and allow_untrained_fallback must be boolean")
        if not isinstance(config.trainable_blocks, int) or isinstance(config.trainable_blocks, bool) or config.trainable_blocks < 0 or config.trainable_blocks > 9:
            raise ValueError("trainable_blocks must be an integer from 0 to 9")
        if config.projection_dim is not None and (not isinstance(config.projection_dim, int) or isinstance(config.projection_dim, bool) or config.projection_dim < 1):
            raise ValueError("projection_dim must be a positive integer or None")

        self.device = self._resolve_device(config.device)
        self.weights_status = "random_initialized"
        weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        try:
            model = efficientnet_b0(weights=weights)
            if config.pretrained:
                self.weights_status = "imagenet_pretrained"
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, OSError) as exc:
            if not (config.pretrained and config.allow_untrained_fallback):
                raise RuntimeError("Requested EfficientNet-B0 weights are unavailable. Set allow_untrained_fallback=True to explicitly use random weights.") from exc
            warnings.warn(f"ImageNet weights unavailable; using random initialization: {exc}", RuntimeWarning, stacklevel=2)
            model = efficientnet_b0(weights=None)
            self.weights_status = "random_fallback_after_weight_error"

        actual_dim = model.classifier[1].in_features
        if actual_dim != self.feature_dim:
            raise RuntimeError(f"Unexpected EfficientNet-B0 feature dimension: {actual_dim}")
        model.classifier = nn.Identity()
        self.backbone = model
        self.projection = nn.Identity() if config.projection_dim is None else nn.Linear(actual_dim, config.projection_dim)
        self.output_dim = actual_dim if config.projection_dim is None else config.projection_dim
        self.trainable_blocks = config.trainable_blocks

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if config.trainable_blocks:
            for block in list(self.backbone.features.children())[-config.trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        if isinstance(self.projection, nn.Linear):
            for parameter in self.projection.parameters():
                parameter.requires_grad = True
        self.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        try:
            resolved = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device: {device!r}") from exc
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable in this PyTorch environment")
        if resolved.type == "cuda" and resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {resolved.index} is unavailable; detected {torch.cuda.device_count()} device(s)")
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("device must be CPU or CUDA")
        return resolved

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4:
            raise ValueError(f"images must have shape [batch, 3, height, width], got rank {images.ndim}")
        if images.shape[0] < 1:
            raise ValueError("image batch must not be empty")
        if images.shape[1] != 3:
            raise ValueError(f"expected 3 RGB channels, got {images.shape[1]}")
        if images.shape[2] < 32 or images.shape[3] < 32:
            raise ValueError("image height and width must each be at least 32 pixels")
        if not images.is_floating_point():
            raise TypeError("images must have a floating point dtype")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        images = images.to(device=self.device, dtype=torch.float32)
        features = self.backbone(images)
        output = self.projection(features)
        if output.ndim != 2 or output.shape != (images.shape[0], self.output_dim):
            raise RuntimeError(f"Unexpected encoder output shape: {tuple(output.shape)}")
        return output

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Run evaluation-mode extraction without recording gradients."""
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                return self(images)
        finally:
            self.train(was_training)
