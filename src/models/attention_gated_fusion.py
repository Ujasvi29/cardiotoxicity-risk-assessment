"""Two-modality attention and gated representation fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AttentionGatedFusionConfig:
    """Fusion dimensions; defaults match the current Module 9 and 10 outputs."""

    image_dim: int = 1280
    clinical_dim: int = 32
    fusion_dim: int = 128
    output_dim: int = 128
    seed: int | None = None
    device: str = "cpu"


class AttentionGatedFusion(nn.Module):
    """Fuse paired image and clinical embeddings into one representation.

    Inputs are ``image_embeddings [B, image_dim]`` and
    ``clinical_embeddings [B, clinical_dim]``. Each is projected to
    ``fusion_dim``. Additive attention scores the two projected modality
    tokens and softmax-normalizes across the modality axis (axis 1), producing
    ``[B, 2]`` weights. A vector gate ``[B, fusion_dim]`` then controls the
    image versus clinical contribution for each shared-space coordinate.
    The fused output has shape ``[B, output_dim]``.

    Both modalities are required. Inputs must already be paired in the same
    verified record order; this module does not create or infer pairings.
    """

    def __init__(self, config: AttentionGatedFusionConfig = AttentionGatedFusionConfig()) -> None:
        super().__init__()
        if not isinstance(config, AttentionGatedFusionConfig):
            raise TypeError("config must be an AttentionGatedFusionConfig")
        dimensions = (config.image_dim, config.clinical_dim, config.fusion_dim, config.output_dim)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("image_dim, clinical_dim, fusion_dim, and output_dim must be positive integers")
        if config.seed is not None and (
            not isinstance(config.seed, int)
            or isinstance(config.seed, bool)
            or config.seed < -(2**63)
            or config.seed > 2**64 - 1
        ):
            raise ValueError("seed must be None or an integer in [-2**63, 2**64 - 1]")
        try:
            device = torch.device(config.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid device: {config.device!r}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"detected {torch.cuda.device_count()} device(s)"
            )
        if device.type == "cpu" and device.index is not None:
            raise ValueError("CPU device must not include an index; use 'cpu'")
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be CPU or CUDA")

        self.config = config
        self.device = device
        # Construct Linear layers inside fork_rng: their default initialization
        # must not consume or reset the caller's global random stream.
        # Layers are initialized on CPU and moved to the requested device below,
        # so only the CPU generator participates in parameter initialization.
        with torch.random.fork_rng():
            if config.seed is not None:
                torch.random.default_generator.manual_seed(config.seed)
            self.image_projection = self._projection(config.image_dim, config.fusion_dim)
            self.clinical_projection = self._projection(config.clinical_dim, config.fusion_dim)
            self.attention_hidden = nn.Linear(config.fusion_dim, config.fusion_dim)
            self.attention_score = nn.Linear(config.fusion_dim, 1, bias=False)
            self.gate = nn.Linear(2 * config.fusion_dim, config.fusion_dim)
            self.output_projection = nn.Identity() if config.output_dim == config.fusion_dim else nn.Linear(config.fusion_dim, config.output_dim)
        self.to(device)

    @staticmethod
    def _projection(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def _validate(self, image_embeddings: torch.Tensor, clinical_embeddings: torch.Tensor) -> None:
        for name, value, expected_dim in (
            ("image_embeddings", image_embeddings, self.config.image_dim),
            ("clinical_embeddings", clinical_embeddings, self.config.clinical_dim),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape [batch, {expected_dim}], got rank {value.ndim}")
            if value.shape[0] == 0:
                raise ValueError(f"{name} batch must not be empty")
            if value.shape[1] != expected_dim:
                raise ValueError(f"{name} expected feature dimension {expected_dim}, got {value.shape[1]}")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            if value.device != self.device:
                raise ValueError(f"{name} is on {value.device}, but fusion module is on {self.device}")
        if image_embeddings.shape[0] != clinical_embeddings.shape[0]:
            raise ValueError("image and clinical embedding batches must have the same size and record ordering")
        if image_embeddings.device != clinical_embeddings.device:
            raise ValueError("image and clinical embeddings must be on the same device")

    def forward(
        self,
        image_embeddings: torch.Tensor,
        clinical_embeddings: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return fused vectors, optionally including attention and gate values.

        The detailed form is intended for tests and diagnostics. It contains
        only model-derived weights and representations, not patient metadata.
        """
        self._validate(image_embeddings, clinical_embeddings)
        image = self.image_projection(image_embeddings.to(dtype=torch.float32))
        clinical = self.clinical_projection(clinical_embeddings.to(dtype=torch.float32))

        # Two modality tokens [B, 2, D]. Scalar additive attention normalizes
        # over token/modality axis 1, never over embedding coordinates.
        tokens = torch.stack((image, clinical), dim=1)
        scores = self.attention_score(torch.tanh(self.attention_hidden(tokens))).squeeze(-1)
        attention = torch.softmax(scores, dim=1)

        # Per-coordinate gate. g weights image and (1-g) weights clinical.
        gate = torch.sigmoid(self.gate(torch.cat((image, clinical), dim=1)))
        fused_shared = attention[:, 0:1] * gate * image + attention[:, 1:2] * (1.0 - gate) * clinical
        fused = self.output_projection(fused_shared)
        if not torch.isfinite(fused).all():
            raise RuntimeError("fusion produced non-finite values from finite inputs")
        if return_details:
            return {"fused": fused, "attention_weights": attention, "gate": gate}
        return fused

    def configuration(self) -> dict[str, Any]:
        """Return JSON-compatible fusion configuration."""
        return asdict(self.config)
