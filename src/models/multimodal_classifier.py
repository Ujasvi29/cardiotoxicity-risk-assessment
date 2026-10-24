"""CTRCD logits from Module 11 fused image and clinical representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from src.models.attention_gated_fusion import AttentionGatedFusion


CTRCD_CLASS_LABELS = (0, 1)


@dataclass(frozen=True)
class MultimodalClassifierConfig:
    """Classification-head settings for the verified binary CTRCD schema."""

    hidden_dims: tuple[int, ...] = (64,)
    dropout: float = 0.1
    seed: int | None = None


class MultimodalCTRCDClassifier(nn.Module):
    """Reuse attention-gated fusion and add a two-logit CTRCD head.

    Inputs are already-produced and correctly paired image and clinical
    embeddings. The module does not load raw modalities, align records, train,
    or infer labels from dataset order. ``forward`` returns logits ordered as
    class 0 then class 1. ``predict_proba`` explicitly returns their softmax
    probabilities in that same order.
    """

    num_classes = len(CTRCD_CLASS_LABELS)

    def __init__(
        self,
        fusion: AttentionGatedFusion | None = None,
        config: MultimodalClassifierConfig = MultimodalClassifierConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(config, MultimodalClassifierConfig):
            raise TypeError("config must be a MultimodalClassifierConfig")
        if not isinstance(config.hidden_dims, tuple) or any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in config.hidden_dims
        ):
            raise ValueError("hidden_dims must be a tuple of positive integers")
        if not isinstance(config.dropout, (int, float)) or isinstance(config.dropout, bool) or not 0 <= config.dropout < 1:
            raise ValueError("dropout must be a number in [0, 1)")
        if config.seed is not None and (
            not isinstance(config.seed, int)
            or isinstance(config.seed, bool)
            or config.seed < -(2**63)
            or config.seed > 2**64 - 1
        ):
            raise ValueError("seed must be None or an integer in [-2**63, 2**64 - 1]")
        if fusion is None:
            fusion = AttentionGatedFusion()
        if not isinstance(fusion, AttentionGatedFusion):
            raise TypeError("fusion must be an AttentionGatedFusion")

        self.fusion = fusion
        self.config = config
        input_dim = fusion.config.output_dim
        widths = (input_dim, *config.hidden_dims, self.num_classes)
        with torch.random.fork_rng():
            if config.seed is not None:
                torch.random.default_generator.manual_seed(config.seed)
            layers: list[nn.Module] = []
            for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(left, right))
                if index < len(widths) - 2:
                    layers.extend((nn.ReLU(), nn.Dropout(config.dropout)))
            self.classifier = nn.Sequential(*layers)
        self.classifier.to(fusion.device)

    def _classify(self, fused: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(fused)
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise RuntimeError(f"unexpected classifier output shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("classifier produced non-finite logits")
        return logits

    def forward(
        self,
        image_embeddings: torch.Tensor,
        clinical_embeddings: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return ``[batch, 2]`` logits; optionally include fusion diagnostics."""
        fused = self.fusion(image_embeddings, clinical_embeddings, return_details=return_diagnostics)
        if return_diagnostics:
            assert isinstance(fused, dict)
            logits = self._classify(fused["fused"])
            return {
                "logits": logits,
                "attention_weights": fused["attention_weights"],
                "gate": fused["gate"],
            }
        assert isinstance(fused, torch.Tensor)
        return self._classify(fused)

    def predict_proba(self, image_embeddings: torch.Tensor, clinical_embeddings: torch.Tensor) -> torch.Tensor:
        """Return `[P(CTRCD=0), P(CTRCD=1)]` in inference mode."""
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                return torch.softmax(self(image_embeddings, clinical_embeddings), dim=1)
        finally:
            self.train(was_training)

    def predict(self, image_embeddings: torch.Tensor, clinical_embeddings: torch.Tensor) -> torch.Tensor:
        """Return predicted schema labels 0 or 1, without selecting a threshold."""
        probabilities = self.predict_proba(image_embeddings, clinical_embeddings)
        return probabilities.argmax(dim=1)

    def configuration(self) -> dict[str, Any]:
        """Return classifier and nested fusion settings for run metadata."""
        return {"classifier": asdict(self.config), "fusion": self.fusion.configuration(), "class_labels": list(CTRCD_CLASS_LABELS)}
