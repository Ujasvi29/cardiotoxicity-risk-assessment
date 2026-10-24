"""Reproducible training and validation utilities for Module 12 classifiers.

This layer consumes caller-supplied, already paired image and clinical
embeddings. It deliberately does not load data or infer multimodal mappings.
"""

from __future__ import annotations

import json
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.attention_gated_fusion import AttentionGatedFusion, AttentionGatedFusionConfig
from src.models.multimodal_classifier import (
    CTRCD_CLASS_LABELS,
    MultimodalClassifierConfig,
    MultimodalCTRCDClassifier,
)
from src.training.class_imbalance import compute_class_weights


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and epoch settings for one model fit."""

    max_epochs: int = 100
    patience: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    class_weighting: bool = False


@dataclass(frozen=True)
class TrialParameters:
    """Settings sampled for one independent tuning trial."""

    learning_rate: float
    weight_decay: float
    dropout: float
    hidden_dims: tuple[int, ...]
    batch_size: int
    seed: int


@dataclass(frozen=True)
class SearchSpace:
    """Small default random-search space; values are candidates, not optima."""

    learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    weight_decays: tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    dropouts: tuple[float, ...] = (0.0, 0.1, 0.3)
    hidden_dims: tuple[tuple[int, ...], ...] = ((32,), (64,), (64, 32))
    batch_sizes: tuple[int, ...] = (16, 32)


@dataclass(frozen=True)
class TuningConfig:
    """Reproducibility and compute limits for validation-only search."""

    trial_count: int = 6
    search_seed: int = 1042
    max_epochs: int = 60
    patience: int = 8
    device: str = "cpu"
    class_weighting: bool = False


class TrialFailure(RuntimeError):
    """An expected, recoverable failure for one hyperparameter trial.

    A model factory may raise this when a candidate is intentionally rejected.
    Other exceptions are recorded for diagnostics and then re-raised so bugs
    are not silently treated as ordinary failed trials.
    """


class PairedEmbeddingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Aligned embedding rows and CTRCD labels supplied by a verified caller.

    This class checks row counts and values but cannot establish record
    identity. The caller is responsible for supplying provenance-verified
    same-record ordering and split membership.
    """

    def __init__(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        labels: torch.Tensor,
        *,
        image_record_ids: Sequence[str],
        clinical_record_ids: Sequence[str],
        label_record_ids: Sequence[str],
        patient_group_ids: Sequence[str],
    ) -> None:
        _validate_arrays(image, clinical, labels, "dataset")
        row_count = int(labels.shape[0])
        identity_columns = {
            "image_record_ids": image_record_ids,
            "clinical_record_ids": clinical_record_ids,
            "label_record_ids": label_record_ids,
            "patient_group_ids": patient_group_ids,
        }
        for name, values in identity_columns.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence of opaque strings")
            if len(values) != row_count or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must provide one non-empty opaque string per record")
        if list(image_record_ids) != list(clinical_record_ids) or list(image_record_ids) != list(label_record_ids):
            raise ValueError("image, clinical, and label record identifiers must match exactly in the same order")
        if len(set(image_record_ids)) != row_count:
            raise ValueError("record identifiers must be unique within a partition")
        self.image = image.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        self.clinical = clinical.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        self.labels = labels.detach().to(device="cpu", dtype=torch.long).contiguous().clone()
        # Identifiers are retained only for integrity checks and are never logged.
        self.record_ids = tuple(image_record_ids)
        self.patient_group_ids = tuple(patient_group_ids)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.image[index], self.clinical[index], self.labels[index]


def _validate_arrays(image: torch.Tensor, clinical: torch.Tensor, labels: torch.Tensor, name: str) -> None:
    for label, tensor in (("image embeddings", image), ("clinical embeddings", clinical), ("labels", labels)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} {label} must be a torch.Tensor")
    if image.ndim != 2 or clinical.ndim != 2:
        raise ValueError(f"{name} image and clinical embeddings must be rank-2 matrices")
    if labels.ndim != 1:
        raise ValueError(f"{name} labels must be a rank-1 vector")
    if image.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if image.shape[0] != clinical.shape[0] or image.shape[0] != labels.shape[0]:
        raise ValueError(f"{name} image, clinical, and label sample counts must match")
    if image.shape[1] < 1 or clinical.shape[1] < 1:
        raise ValueError(f"{name} embedding feature dimensions must be positive")
    if not image.is_floating_point() or not clinical.is_floating_point():
        raise TypeError(f"{name} embeddings must use floating-point dtypes")
    if labels.dtype != torch.long:
        raise TypeError(f"{name} labels must use torch.int64 (torch.long)")
    for label, tensor in (("image embeddings", image), ("clinical embeddings", clinical)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} {label} contain non-finite values")
    if not torch.isin(labels, torch.tensor(CTRCD_CLASS_LABELS, device=labels.device)).all():
        raise ValueError(f"{name} labels must contain only the classes {CTRCD_CLASS_LABELS}")


def make_paired_embedding_dataset(
    image_embeddings: torch.Tensor,
    clinical_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    image_record_ids: Sequence[str],
    clinical_record_ids: Sequence[str],
    label_record_ids: Sequence[str],
    patient_group_ids: Sequence[str],
) -> PairedEmbeddingDataset:
    """Create checked rows from an explicit caller-provided record mapping.

    The record identifiers must be opaque, non-sensitive keys. Their equality
    verifies order consistency, not the real-world correctness of the mapping.
    """
    return PairedEmbeddingDataset(
        image_embeddings,
        clinical_embeddings,
        labels,
        image_record_ids=image_record_ids,
        clinical_record_ids=clinical_record_ids,
        label_record_ids=label_record_ids,
        patient_group_ids=patient_group_ids,
    )


def _check_partition_disjoint(left: Dataset, right: Dataset, left_name: str, right_name: str) -> None:
    if not isinstance(left, PairedEmbeddingDataset) or not isinstance(right, PairedEmbeddingDataset):
        raise TypeError("leakage checks require PairedEmbeddingDataset instances with record and patient group IDs")
    if set(left.record_ids).intersection(right.record_ids):
        raise ValueError(f"{left_name} and {right_name} contain overlapping record identifiers")
    if set(left.patient_group_ids).intersection(right.patient_group_ids):
        raise ValueError(f"{left_name} and {right_name} contain overlapping patient groups")


def _check_model_device(model: MultimodalCTRCDClassifier, device: torch.device) -> None:
    parameter_devices = {parameter.device for parameter in model.parameters()}
    if parameter_devices and parameter_devices != {device}:
        raise ValueError("model parameters must already be on the trainer's configured device")
    # AttentionGatedFusion stores the unindexed form of `cuda` when requested;
    # normalize that runtime check to the actual current CUDA device.
    if model.fusion.device.type == "cuda" and model.fusion.device.index is None and device.type == "cuda":
        model.fusion.device = device


def _validate_training_config(config: TrainingConfig) -> torch.device:
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    for name in ("max_epochs", "patience", "batch_size"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(config.learning_rate, bool) or not isinstance(config.learning_rate, Real) or not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(config.weight_decay, bool) or not isinstance(config.weight_decay, Real) or not np.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool) or not -(2**63) <= config.seed <= 2**64 - 1:
        raise ValueError("seed must be an integer in [-2**63, 2**64 - 1]")
    try:
        device = torch.device(config.device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device: {config.device!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is not None and device.index < 0:
        raise ValueError("CUDA device index must be nonnegative")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU device must not include an index; use 'cpu'")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA")
    if not isinstance(config.class_weighting, bool):
        raise ValueError("class_weighting must be a boolean")
    return device


def _check_dataset(dataset: Dataset, name: str, model: MultimodalCTRCDClassifier) -> None:
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{name} must be a torch Dataset created from verified paired tensors")
    if len(dataset) < 1:
        raise ValueError(f"{name} must not be empty")
    # Check every row before allocating a training loop, including custom datasets.
    image_dim = clinical_dim = None
    for item in dataset:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(f"{name} items must be (image_embedding, clinical_embedding, label)")
        image, patient_features, target = item
        if not all(isinstance(value, torch.Tensor) for value in (image, patient_features, target)):
            raise TypeError(f"{name} items must contain tensors")
        if image.ndim != 1 or patient_features.ndim != 1 or target.ndim != 0:
            raise ValueError(f"{name} items must contain two feature vectors and one scalar label")
        if image_dim is None:
            image_dim, clinical_dim = image.shape[0], patient_features.shape[0]
        if image.shape[0] != image_dim or patient_features.shape[0] != clinical_dim:
            raise ValueError(f"{name} embedding feature dimensions must be consistent across records")
        if not image.is_floating_point() or not patient_features.is_floating_point():
            raise TypeError(f"{name} embeddings must use floating-point dtypes")
        if not torch.isfinite(image).all() or not torch.isfinite(patient_features).all():
            raise ValueError(f"{name} embeddings contain non-finite values")
        if target.dtype != torch.long:
            raise TypeError(f"{name} labels must use torch.int64 (torch.long)")
        target_value = int(target.item())
        if target_value not in CTRCD_CLASS_LABELS:
            raise ValueError(f"{name} labels must contain only the classes {CTRCD_CLASS_LABELS}")
    if image_dim != model.fusion.config.image_dim:
        raise ValueError(f"{name} image feature dimension does not match the classifier's fusion configuration")
    if clinical_dim != model.fusion.config.clinical_dim:
        raise ValueError(f"{name} clinical feature dimension does not match the classifier's fusion configuration")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.shape != (len(labels), 2) or len(labels) == 0:
        raise ValueError("metrics require non-empty labels and an (N, 2) probability matrix")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, CTRCD_CLASS_LABELS).all():
        raise ValueError("metrics labels must use class order (0, 1)")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("metric probabilities must be finite values in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("each metric probability row must sum to 1")
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    negative_count = tn + fp
    positive_count = tp + fn
    predicted_positive = tp + fp
    predicted_negative = tn + fn
    precision = tp / predicted_positive if predicted_positive else None
    sensitivity = tp / positive_count if positive_count else None
    specificity = tn / negative_count if negative_count else None
    negative_predictive_value = tn / predicted_negative if predicted_negative else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    has_both_classes = positive_count > 0 and negative_count > 0
    roc_auc = float(roc_auc_score(labels, probabilities[:, 1])) if has_both_classes else None
    average_precision = float(average_precision_score(labels, probabilities[:, 1])) if positive_count else None
    balanced_accuracy = (sensitivity + specificity) / 2 if has_both_classes else None
    one_hot = np.eye(2, dtype=np.float64)[labels]
    brier_score = float(np.mean(np.square(probabilities - one_hot).sum(axis=1)))
    return {
        "count": int(len(labels)),
        "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "class_proportions": {"0": float(negative_count / len(labels)), "1": float(positive_count / len(labels))},
        "positive_class": {"label": 1, "meaning": "CTRCD-positive"},
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_class_1": precision,
        "recall_class_1_sensitivity": sensitivity,
        "specificity_class_0": specificity,
        "negative_predictive_value_class_0": negative_predictive_value,
        "f1_class_1": f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "confusion_denominators": {"negative": negative_count, "positive": positive_count},
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "average_precision_class_1": average_precision,
        "brier_score": brier_score,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


class MultimodalTrainer:
    """Train and validate Module 12 models using supplied paired embeddings."""

    def __init__(self, config: TrainingConfig = TrainingConfig()) -> None:
        self.config = config
        self.device = _validate_training_config(config)

    def fit(
        self,
        model: MultimodalCTRCDClassifier,
        train_data: Dataset,
        validation_data: Dataset,
        *,
        output_dir: Path | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit one model; only validation ROC-AUC selects/restores checkpoints."""
        if not isinstance(model, MultimodalCTRCDClassifier):
            raise TypeError("model must be a MultimodalCTRCDClassifier")
        _check_dataset(train_data, "train_data", model)
        _check_dataset(validation_data, "validation_data", model)
        if train_data is validation_data:
            raise ValueError("train_data and validation_data must be separate partitions")
        _check_partition_disjoint(train_data, validation_data, "train_data", "validation_data")
        train_labels = torch.stack([train_data[i][2] for i in range(len(train_data))])
        val_labels = torch.stack([validation_data[i][2] for i in range(len(validation_data))])
        if torch.unique(train_labels).numel() != 2:
            raise ValueError("train_data must contain both CTRCD classes for binary classifier training")
        if torch.unique(val_labels).numel() != 2:
            raise ValueError("validation_data must contain both CTRCD classes to select by ROC-AUC")
        _check_model_device(model, self.device)
        if output_dir is not None:
            output_dir = Path(output_dir)
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite non-empty training output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        metadata = run_metadata or {}
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("run_metadata must contain JSON-safe, non-sensitive values") from exc

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        validation_generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 1)
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(validation_data, batch_size=self.config.batch_size, shuffle=False, generator=validation_generator)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        class_weights = None
        if self.config.class_weighting:
            parameter = next(model.parameters())
            class_weights = compute_class_weights(
                train_labels,
                class_labels=CTRCD_CLASS_LABELS,
                device=self.device,
                dtype=parameter.dtype,
            )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        validation_criterion = nn.CrossEntropyLoss()
        histories: dict[str, list[float]] = {"train_loss": [], "validation_loss": [], "validation_roc_auc": [], "validation_accuracy": []}
        best_auc = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []

        # Isolate dropout/model stochasticity from the caller's RNG streams.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.config.seed)
            if self.device.type == "cuda":
                torch.cuda.default_generators[devices[0]].manual_seed(self.config.seed)
            for epoch in range(1, self.config.max_epochs + 1):
                model.train()
                train_loss_sum = 0.0
                train_count = 0
                for image, clinical, target in train_loader:
                    image = image.to(self.device)
                    clinical = clinical.to(self.device)
                    target = target.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(image, clinical)
                    loss = criterion(logits, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.detach()) * int(target.shape[0])
                    train_count += int(target.shape[0])

                val_loss, val_labels_np, val_probabilities = self._predict(model, val_loader, validation_criterion)
                validation_metrics = _metrics(val_labels_np, val_probabilities)
                histories["train_loss"].append(train_loss_sum / train_count)
                histories["validation_loss"].append(val_loss)
                histories["validation_roc_auc"].append(validation_metrics["roc_auc"])
                histories["validation_accuracy"].append(validation_metrics["accuracy"])

                if validation_metrics["roc_auc"] > best_auc:
                    best_auc = validation_metrics["roc_auc"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.config.patience:
                        break

        assert best_state is not None
        model.load_state_dict(best_state)
        result: dict[str, Any] = {
            "status": "completed",
            "best_epoch": best_epoch,
            "best_validation_roc_auc": best_auc,
            "best_validation_metrics": _metrics(*self._predict_arrays(model, validation_data)),
            "training_class_counts": {
                "0": int((train_labels == 0).sum()),
                "1": int((train_labels == 1).sum()),
            },
            "training_class_proportions": {
                "0": int((train_labels == 0).sum()) / len(train_labels),
                "1": int((train_labels == 1).sum()) / len(train_labels),
            },
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
            "validation_roc_auc_reliability_note": (
                "Validation ranking is highly uncertain because one or both class counts are below 5."
                if min(int((val_labels == 0).sum()), int((val_labels == 1).sum())) < 5
                else None
            ),
            "history": histories,
            "training_config": asdict(self.config),
            "model_config": model.configuration(),
            "optimizer": {"name": "AdamW", "learning_rate": self.config.learning_rate, "weight_decay": self.config.weight_decay},
            "class_order": list(CTRCD_CLASS_LABELS),
            "split_integrity_checks": {
                "record_id_order_checked_across_modalities_and_labels": True,
                "record_ids_unique_within_partition": True,
                "record_and_patient_group_overlap_checked_between_train_and_validation": True,
                "real_world_pairing_provenance_verified_by_trainer": False,
            },
            "software": {"python": platform.python_version(), "torch": str(torch.__version__)},
            "run_metadata": metadata,
        }
        if output_dir is not None:
            checkpoint = {
                "format_version": 1,
                "model_state_dict": best_state,
                "metadata": result,
            }
            _atomic_torch_save(checkpoint, output_dir / "best_model.pt")
            _write_json(result, output_dir / "history.json")
            result["checkpoint_path"] = str(output_dir / "best_model.pt")
        return result

    def _predict(self, model: MultimodalCTRCDClassifier, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
        was_training = model.training
        model.eval()
        loss_sum = 0.0
        count = 0
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        try:
            with torch.inference_mode():
                for image, clinical, target in loader:
                    image, clinical, target = image.to(self.device), clinical.to(self.device), target.to(self.device)
                    logits = model(image, clinical)
                    loss = criterion(logits, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite validation loss")
                    loss_sum += float(loss) * int(target.shape[0])
                    count += int(target.shape[0])
                    labels.append(target.cpu().numpy())
                    probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        finally:
            model.train(was_training)
        return loss_sum / count, np.concatenate(labels), np.concatenate(probabilities)

    def _predict_arrays(self, model: MultimodalCTRCDClassifier, data: Dataset) -> tuple[np.ndarray, np.ndarray]:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 2)
        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False, generator=generator)
        _, labels, probabilities = self._predict(model, loader, nn.CrossEntropyLoss())
        return labels, probabilities

    def evaluate(
        self,
        model: MultimodalCTRCDClassifier,
        test_data: Dataset,
        *,
        train_data: Dataset,
        validation_data: Dataset,
    ) -> dict[str, Any]:
        """Evaluate a locked model once on caller-supplied held-out test data."""
        if not isinstance(model, MultimodalCTRCDClassifier):
            raise TypeError("model must be a MultimodalCTRCDClassifier")
        _check_model_device(model, self.device)
        _check_dataset(test_data, "test_data", model)
        _check_dataset(train_data, "train_data", model)
        _check_dataset(validation_data, "validation_data", model)
        _check_partition_disjoint(train_data, validation_data, "train_data", "validation_data")
        _check_partition_disjoint(train_data, test_data, "train_data", "test_data")
        _check_partition_disjoint(validation_data, test_data, "validation_data", "test_data")
        labels, probabilities = self._predict_arrays(model, test_data)
        return _metrics(labels, probabilities)


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[MultimodalCTRCDClassifier, dict[str, Any]]:
    """Safely restore a Module 12 checkpoint containing tensors and primitives."""
    target = torch.device(device)
    target = _validate_training_config(TrainingConfig(device=str(target)))
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError as exc:  # Older supported torch versions lack weights_only.
        raise RuntimeError("safe checkpoint loading requires a PyTorch version with weights_only=True") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported or malformed multimodal checkpoint")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("class_order") != list(CTRCD_CLASS_LABELS):
        raise ValueError("checkpoint class order or metadata is invalid")
    model_config = metadata.get("model_config")
    if (
        not isinstance(model_config, dict)
        or model_config.get("class_labels") != list(CTRCD_CLASS_LABELS)
        or not isinstance(model_config.get("fusion"), dict)
        or not isinstance(model_config.get("classifier"), dict)
        or not isinstance(payload.get("model_state_dict"), dict)
    ):
        raise ValueError("checkpoint model configuration is invalid")
    try:
        fusion_data = dict(model_config["fusion"])
        fusion_data["device"] = str(target)
        fusion = AttentionGatedFusion(AttentionGatedFusionConfig(**fusion_data))
        classifier_data = dict(model_config["classifier"])
        classifier_data["hidden_dims"] = tuple(classifier_data["hidden_dims"])
        model = MultimodalCTRCDClassifier(fusion, MultimodalClassifierConfig(**classifier_data))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint model configuration is malformed") from exc
    model.to(target)
    model.fusion.device = target
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint model weights do not match the recorded model configuration") from exc
    return model, metadata


def make_trial_model(base_model: MultimodalCTRCDClassifier, parameters: TrialParameters, *, device: str = "cpu") -> MultimodalCTRCDClassifier:
    """Build fresh fusion/classifier weights for a tuning trial."""
    resolved_device = str(_validate_training_config(TrainingConfig(device=device)))
    fusion_config = replace(base_model.fusion.config, seed=parameters.seed, device=resolved_device)
    fusion = AttentionGatedFusion(fusion_config)
    classifier_config = MultimodalClassifierConfig(
        hidden_dims=parameters.hidden_dims,
        dropout=parameters.dropout,
        seed=parameters.seed + 1,
    )
    return MultimodalCTRCDClassifier(fusion, classifier_config)


class MultimodalHyperparameterSearch:
    """Reproducible random search selected only by validation ROC-AUC.

    There is intentionally no test-data argument: test observations cannot be
    consulted during a search trial.
    """

    def __init__(
        self,
        config: TuningConfig = TuningConfig(),
        search_space: SearchSpace = SearchSpace(),
    ) -> None:
        if not isinstance(config, TuningConfig):
            raise TypeError("config must be a TuningConfig")
        if not isinstance(config.trial_count, int) or isinstance(config.trial_count, bool) or config.trial_count < 1:
            raise ValueError("trial_count must be a positive integer")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (config.max_epochs, config.patience)):
            raise ValueError("max_epochs and patience must be positive integers")
        if not isinstance(config.class_weighting, bool):
            raise ValueError("class_weighting must be a boolean")
        if not isinstance(config.search_seed, int) or isinstance(config.search_seed, bool) or not -(2**63) <= config.search_seed <= 2**64 - 1:
            raise ValueError("search_seed must be an integer in [-2**63, 2**64 - 1]")
        if config.search_seed + config.trial_count - 1 > 2**64 - 1:
            raise ValueError("search_seed plus trial_count exceeds the supported seed range")
        if not isinstance(search_space, SearchSpace):
            raise TypeError("search_space must be a SearchSpace")
        self.config = config
        self.space = search_space
        self.device = _validate_training_config(TrainingConfig(device=config.device))
        for name in ("learning_rates", "weight_decays", "dropouts", "hidden_dims", "batch_sizes"):
            if not getattr(search_space, name):
                raise ValueError(f"search_space.{name} must not be empty")
        if any(not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in search_space.learning_rates):
            raise ValueError("search_space.learning_rates must contain finite positive values")
        if any(not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value < 0 for value in search_space.weight_decays):
            raise ValueError("search_space.weight_decays must contain finite nonnegative values")
        if any(not isinstance(value, Real) or isinstance(value, bool) or not 0 <= value < 1 for value in search_space.dropouts):
            raise ValueError("search_space.dropouts must contain values in [0, 1)")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in search_space.batch_sizes):
            raise ValueError("search_space.batch_sizes must contain positive integers")
        if any(not isinstance(shape, tuple) or any(not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in shape) for shape in search_space.hidden_dims):
            raise ValueError("search_space.hidden_dims must contain tuples of positive integers")

    def search(
        self,
        model_factory: Callable[[TrialParameters], MultimodalCTRCDClassifier],
        train_data: Dataset,
        validation_data: Dataset,
        *,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run independent trials; best trial is chosen solely by validation AUC."""
        rng = random.Random(self.config.search_seed)
        if output_dir is not None:
            output_dir = Path(output_dir)
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite non-empty tuning output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        best_trial: dict[str, Any] | None = None
        seen_models: list[MultimodalCTRCDClassifier] = []
        for index in range(self.config.trial_count):
            params = TrialParameters(
                learning_rate=float(rng.choice(self.space.learning_rates)),
                weight_decay=float(rng.choice(self.space.weight_decays)),
                dropout=float(rng.choice(self.space.dropouts)),
                hidden_dims=tuple(rng.choice(self.space.hidden_dims)),
                batch_size=int(rng.choice(self.space.batch_sizes)),
                seed=self.config.search_seed + index,
            )
            record: dict[str, Any] = {
                "trial": index + 1,
                "parameters": asdict(params),
                "seed": params.seed,
                "class_weighting": self.config.class_weighting,
            }
            try:
                model = model_factory(params)
                if not isinstance(model, MultimodalCTRCDClassifier):
                    raise TypeError("model_factory must return a fresh MultimodalCTRCDClassifier")
                if any(model is existing for existing in seen_models):
                    raise ValueError("model_factory reused a model instance; each trial requires fresh model weights")
                seen_models.append(model)
                trainer_config = TrainingConfig(
                    max_epochs=self.config.max_epochs,
                    patience=self.config.patience,
                    batch_size=params.batch_size,
                    learning_rate=params.learning_rate,
                    weight_decay=params.weight_decay,
                    seed=params.seed,
                    device=self.config.device,
                    class_weighting=self.config.class_weighting,
                )
                trial_dir = output_dir / f"trial_{index + 1:03d}" if output_dir is not None else None
                result = MultimodalTrainer(trainer_config).fit(model, train_data, validation_data, output_dir=trial_dir)
                record.update({
                    "status": "completed",
                    "best_epoch": result["best_epoch"],
                    "validation_roc_auc": result["best_validation_roc_auc"],
                    "validation_metrics": result["best_validation_metrics"],
                    "validation_roc_auc_reliability_note": result["validation_roc_auc_reliability_note"],
                    "history": result["history"],
                    "checkpoint_path": result.get("checkpoint_path"),
                })
                if best_trial is None or record["validation_roc_auc"] > best_trial["validation_roc_auc"]:
                    best_trial = record
            except TrialFailure as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "selected": False})
            except Exception as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "selected": False})
                records.append(record)
                report = {
                    "selection_metric": "validation_roc_auc",
                    "search_seed": self.config.search_seed,
                    "trial_count": self.config.trial_count,
                    "class_weighting": self.config.class_weighting,
                    "best_trial": best_trial,
                    "trials": records,
                }
                if output_dir is not None:
                    _write_json(report, output_dir / "trials.json")
                raise
            records.append(record)

        for record in records:
            record["selected"] = bool(best_trial is not None and record["trial"] == best_trial["trial"])

        report = {
            "selection_metric": "validation_roc_auc",
            "search_seed": self.config.search_seed,
            "trial_count": self.config.trial_count,
            "class_weighting": self.config.class_weighting,
            "best_trial": best_trial,
            "trials": records,
        }
        if output_dir is not None:
            _write_json(report, output_dir / "trials.json")
        if best_trial is None:
            raise RuntimeError(f"all hyperparameter trials failed; see recorded trial errors: {records}")
        return report
