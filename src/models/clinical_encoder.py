"""Clinical feature preprocessing and MLP representation encoder.

The encoder returns a learned representation only. It has no prediction head
or training objective; callers own any future protocol-specific training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch import nn

from src.data.preprocessing import (
    BINARY_COLS,
    CONTINUOUS_COLS,
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    build_clinical_preprocessor,
)


@dataclass(frozen=True)
class ClinicalEncoderConfig:
    """MLP settings. Defaults are engineering choices, not study protocol."""

    input_dim: int = len(FEATURE_COLS)
    hidden_dims: tuple[int, ...] = (128, 64)
    clinical_embedding_dim: int = 32
    activation: str = "relu"
    dropout: float = 0.1
    seed: int | None = None
    device: str = "cpu"


class ClinicalFeaturePreprocessor:
    """Fit the project's existing imputation/scaling pipeline on training rows.

    Input must contain exactly the documented baseline features, in any order.
    This strict interface prevents IDs, targets, follow-up time, and unknown
    post-baseline columns from silently entering the representation.
    """

    def __init__(self, scaling_method: str = "robust") -> None:
        if scaling_method not in {"robust", "standard"}:
            raise ValueError("scaling_method must be 'robust' or 'standard'")
        self.scaling_method = scaling_method
        self._transformer: ColumnTransformer | None = None
        self.feature_names_in_: tuple[str, ...] | None = None

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("clinical inputs must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("clinical input must contain at least one record")
        if not frame.columns.is_unique:
            raise ValueError("clinical input contains duplicate column names")
        forbidden = {TARGET_COL, TIME_COL}.intersection(frame.columns)
        id_columns = [c for c in frame.columns if any(token in str(c).lower() for token in ("id", "identifier"))]
        if forbidden or id_columns:
            names = sorted(forbidden.union(id_columns))
            raise ValueError(f"clinical inputs contain forbidden target, follow-up, or identifier columns: {names}")
        missing = [c for c in FEATURE_COLS if c not in frame.columns]
        extra = [c for c in frame.columns if c not in FEATURE_COLS]
        if missing or extra:
            raise ValueError(f"clinical schema mismatch; missing={missing}, unsupported={extra}")
        ordered = frame.loc[:, FEATURE_COLS].copy()
        for column in FEATURE_COLS:
            if not pd.api.types.is_numeric_dtype(ordered[column]):
                raise TypeError(f"clinical feature {column!r} must be numeric; encode it under a documented schema first")
            values = ordered[column].to_numpy(dtype=np.float64, na_value=np.nan)
            if not np.isfinite(values[~np.isnan(values)]).all():
                raise ValueError(f"clinical feature {column!r} contains infinite values")
        for column in BINARY_COLS:
            values = set(ordered[column].dropna().unique())
            if not values.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f"binary clinical feature {column!r} must contain only 0, 1, or missing values")
        return ordered

    def fit(self, frame: pd.DataFrame) -> "ClinicalFeaturePreprocessor":
        ordered = self._validate_frame(frame)
        transformer = build_clinical_preprocessor(self.scaling_method)
        transformer.fit(ordered)
        self._transformer = transformer
        self.feature_names_in_ = tuple(FEATURE_COLS)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._transformer is None:
            raise RuntimeError("preprocessor is not fitted; call fit on training data first")
        ordered = self._validate_frame(frame)
        values = self._transformer.transform(ordered)
        return pd.DataFrame(values, columns=self._transformer.get_feature_names_out(), index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def to_schema(self) -> dict[str, Any]:
        if self._transformer is None:
            raise RuntimeError("preprocessor is not fitted")
        return {
            "feature_columns": list(self.feature_names_in_ or ()),
            "continuous_columns": list(CONTINUOUS_COLS),
            "binary_columns": list(BINARY_COLS),
            "scaling_method": self.scaling_method,
            "missing_values": {"continuous": "median imputation", "binary": "most_frequent imputation"},
            "fit_scope": "training partition only; caller is responsible for partitioning before fit",
        }


def _activation(name: str) -> nn.Module:
    activations: Mapping[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]()
    except (KeyError, AttributeError) as exc:
        raise ValueError("activation must be one of: relu, gelu, tanh") from exc


class ClinicalMLPEncoder(nn.Module):
    """Map preprocessed baseline feature vectors to fixed-dimensional vectors."""

    def __init__(self, config: ClinicalEncoderConfig = ClinicalEncoderConfig()) -> None:
        super().__init__()
        if not isinstance(config, ClinicalEncoderConfig):
            raise TypeError("config must be a ClinicalEncoderConfig")
        dims = (config.input_dim, *config.hidden_dims, config.clinical_embedding_dim)
        if not config.hidden_dims or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in dims):
            raise ValueError("input_dim, hidden_dims, and clinical_embedding_dim must be positive integers")
        if not isinstance(config.dropout, (int, float)) or not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        try:
            device = torch.device(config.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid device: {config.device!r}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be CPU or CUDA")
        self.config = config
        self.device = device
        self.input_dim = config.input_dim
        self.output_dim = config.clinical_embedding_dim
        # Isolate seeded initialization so constructing a seeded model does not
        # reset or consume the caller's global CPU/CUDA random streams.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(config.seed)
            layers: list[nn.Module] = []
            for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
                layers.append(nn.Linear(left, right))
                if index < len(dims) - 2:
                    layers.extend((_activation(config.activation), nn.Dropout(config.dropout)))
            self.network = nn.Sequential(*layers)
        self.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError(f"features must have shape [batch, {self.input_dim}], got rank {features.ndim}")
        if features.shape[0] < 1:
            raise ValueError("clinical feature batch must not be empty")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} clinical features, got {features.shape[1]}")
        if not features.is_floating_point():
            raise TypeError("features must have a floating point dtype")
        if not torch.isfinite(features).all():
            raise ValueError("features contain non-finite values")
        output = self.network(features.to(device=self.device, dtype=torch.float32))
        if output.shape != (features.shape[0], self.output_dim):
            raise RuntimeError(f"unexpected encoder output shape: {tuple(output.shape)}")
        return output

    def extract(self, features: torch.Tensor) -> torch.Tensor:
        """Extract embeddings in eval/inference mode, restoring prior mode."""
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                return self(features)
        finally:
            self.train(was_training)

    def configuration(self) -> dict[str, Any]:
        """JSON-compatible model configuration for reproducibility metadata."""
        return asdict(self.config)
