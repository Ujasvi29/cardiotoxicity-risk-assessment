"""Training-history diagnostics and locked-checkpoint evaluation helpers.

History analysis is descriptive and never changes a model or selects a model.
Final test evaluation delegates partition checks and inference to Module 13.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from src.models.multimodal_classifier import CTRCD_CLASS_LABELS, MultimodalCTRCDClassifier
from src.training.multimodal_trainer import (
    MultimodalTrainer,
    TrainingConfig,
    load_checkpoint,
)


_HISTORY_KEYS = (
    "train_loss",
    "validation_loss",
    "validation_roc_auc",
    "validation_accuracy",
)


def _finite_series(value: Any, name: str) -> list[float]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"history field {name!r} must be a one-dimensional sequence")
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"history field {name!r} must be a sequence of finite numbers")
    if not value:
        raise ValueError(f"history field {name!r} must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError(f"history field {name!r} must contain only finite numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"history field {name!r} contains a non-finite value")
    if name.endswith("loss") and any(item < 0 for item in result):
        raise ValueError(f"history field {name!r} contains a negative loss")
    if name in {"validation_roc_auc", "validation_accuracy"} and any(not 0 <= item <= 1 for item in result):
        raise ValueError(f"history field {name!r} must stay within [0, 1]")
    return result


def _trend(values: list[float]) -> dict[str, Any]:
    differences = [right - left for left, right in zip(values[:-1], values[1:])]
    nonzero_directions = [1 if item > 0 else -1 for item in differences if item != 0]
    direction_changes = sum(
        left != right for left, right in zip(nonzero_directions[:-1], nonzero_directions[1:])
    )
    return {
        "start": values[0],
        "end": values[-1],
        "change": values[-1] - values[0],
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "direction_changes": direction_changes,
        "direction": "decreasing" if values[-1] < values[0] else "increasing" if values[-1] > values[0] else "unchanged",
    }


def analyze_training_history(
    history: Mapping[str, Any] | None,
    *,
    training_config: Mapping[str, Any] | None = None,
    best_epoch: int | None = None,
    selection_metric: str | None = None,
    trend_window: int = 3,
) -> dict[str, Any]:
    """Summarize training/validation trends without changing model selection.

    A possible overfitting pattern requires training loss to fall, validation
    loss to rise, and a recorded validation metric not to improve over the
    same recent window. This is a heuristic description, not a diagnosis.
    Loss gaps are calculated only when metadata explicitly says class
    weighting was disabled, because Module 14 uses weighted training loss and
    unweighted validation loss.
    """
    if not isinstance(trend_window, int) or isinstance(trend_window, bool) or trend_window < 2:
        raise ValueError("trend_window must be an integer of at least 2")
    output: dict[str, Any] = {
        "status": "insufficient_evidence",
        "epochs_analyzed": 0,
        "best_epoch": best_epoch if isinstance(best_epoch, int) and not isinstance(best_epoch, bool) else None,
        "selection_metric": selection_metric,
        "training_validation_loss_gap": {"status": "unavailable", "final": None, "by_epoch": None},
        "trends": {},
        "assessment": "Insufficient history to assess training and validation behavior.",
        "warnings": [],
        "recommendations": [],
    }
    if history is None:
        output["warnings"].append("No training history was supplied; no overfitting assessment was made.")
        return output
    if not isinstance(history, Mapping):
        output["status"] = "invalid_history"
        output["warnings"].append("History must be a mapping of metric names to epoch sequences.")
        return output

    series: dict[str, list[float]] = {}
    try:
        for name in _HISTORY_KEYS:
            if name in history:
                series[name] = _finite_series(history[name], name)
    except ValueError as exc:
        output["status"] = "invalid_history"
        output["warnings"].append(str(exc))
        return output
    if not series:
        output["warnings"].append("No recognized training or validation history fields were supplied.")
        return output
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1:
        output["status"] = "inconsistent_history"
        output["warnings"].append("Recognized history fields have different epoch counts; trends were not aligned or inferred.")
        return output

    epoch_count = lengths.pop()
    output["epochs_analyzed"] = epoch_count
    if best_epoch is not None and (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or best_epoch < 1
        or best_epoch > epoch_count
    ):
        output["warnings"].append("best_epoch is missing from or inconsistent with the supplied history.")
        output["best_epoch"] = None
    output["trends"] = {name: _trend(values) for name, values in series.items()}

    class_weighting = training_config.get("class_weighting") if isinstance(training_config, Mapping) else None
    if "train_loss" in series and "validation_loss" in series:
        if class_weighting is False:
            gaps = [validation - train for train, validation in zip(series["train_loss"], series["validation_loss"])]
            output["training_validation_loss_gap"] = {
                "status": "comparable_approximate",
                "meaning": "validation loss minus epoch-averaged training loss",
                "final": gaps[-1],
                "by_epoch": gaps,
            }
        else:
            reason = "weighted training loss differs from unweighted validation loss" if class_weighting is True else "class_weighting metadata is unavailable"
            output["training_validation_loss_gap"] = {
                "status": "not_comparable",
                "reason": reason,
                "final": None,
                "by_epoch": None,
            }
    else:
        output["warnings"].append("Both train_loss and validation_loss are needed to compare their trends.")

    if epoch_count < trend_window:
        output["warnings"].append(
            f"Only {epoch_count} epochs are available; the configured trend window requires {trend_window}."
        )
        return output
    if not {"train_loss", "validation_loss"}.issubset(series):
        output["warnings"].append("Loss divergence cannot be assessed without both loss series.")
        return output

    recent_train = series["train_loss"][-trend_window:]
    recent_validation = series["validation_loss"][-trend_window:]
    train_decreasing = recent_train[-1] < recent_train[0]
    validation_increasing = recent_validation[-1] > recent_validation[0]
    metric_name = next((key for key in ("validation_roc_auc", "validation_accuracy") if key in series), None)
    metric_not_improving = False
    if metric_name is not None:
        recent_metric = series[metric_name][-trend_window:]
        metric_not_improving = recent_metric[-1] <= recent_metric[0]
        output["validation_metric_for_trend"] = metric_name

    if class_weighting is not False:
        output["status"] = "inconclusive_loss_comparability"
        output["assessment"] = (
            "Training and validation loss scales cannot be compared because class-weighting metadata is enabled or unavailable. "
            "Recorded validation metrics remain descriptive, but this history is insufficient for a loss-divergence assessment."
        )
        output["warnings"].append("Set and retain class_weighting metadata before comparing training and validation loss trends.")
        return output

    divergence = train_decreasing and validation_increasing
    if divergence and metric_name is not None and metric_not_improving:
        output["status"] = "possible_overfitting_pattern"
        output["assessment"] = (
            "Recent training loss decreased while validation loss increased and the recorded validation metric did not improve. "
            "This combination is consistent with a possible overfitting pattern, but is not a diagnosis."
        )
        output["warnings"].append("Recent loss and validation-metric trends diverge; confirm with independent split provenance and repeated evaluation.")
        output["recommendations"].extend((
            "Verify patient-level split integrity and mapping provenance before interpreting the gap.",
            "Review the existing validation-based early-stopping decision; consider stronger regularization or a simpler model only if the pattern persists.",
        ))
    elif divergence:
        output["status"] = "loss_divergence_observed"
        output["assessment"] = "Training loss fell while validation loss rose; no aligned validation metric was available to support a stronger overfitting assessment."
        output["warnings"].append("Loss divergence alone is insufficient to diagnose overfitting.")
    else:
        output["status"] = "no_clear_recent_divergence"
        output["assessment"] = "The recent loss trends do not show training loss falling while validation loss rises; this does not establish generalization."
        if not train_decreasing and not validation_increasing:
            output["warnings"].append("Loss trends are flat or jointly worsening; inspect optimization and model capacity before drawing conclusions.")
            output["assessment"] = "The supplied loss trends do not indicate clear overfitting divergence and may reflect an optimization stall or underfitting; the history alone cannot distinguish them."
            output["recommendations"].append("Review training convergence, feature quality, and model capacity; avoid changing settings based only on this heuristic.")
    return output


def analyze_training_result(
    result: Mapping[str, Any] | None,
    *,
    checkpoint_identity: str | None = None,
    trend_window: int = 3,
) -> dict[str, Any]:
    """Analyze the result mapping returned by ``MultimodalTrainer.fit``."""
    if not isinstance(result, Mapping):
        return analyze_training_history(None, trend_window=trend_window)
    analysis = analyze_training_history(
        result.get("history"),
        training_config=result.get("training_config"),
        best_epoch=result.get("best_epoch"),
        selection_metric="validation_roc_auc" if "best_validation_roc_auc" in result else None,
        trend_window=trend_window,
    )
    best_auc = result.get("best_validation_roc_auc")
    if best_auc is not None and (
        isinstance(best_auc, bool)
        or not isinstance(best_auc, Real)
        or not math.isfinite(float(best_auc))
        or not 0 <= float(best_auc) <= 1
    ):
        analysis["warnings"].append("best_validation_roc_auc is invalid and was omitted.")
        best_auc = None
    analysis["best_validation_roc_auc"] = float(best_auc) if best_auc is not None else None
    is_sha256 = (
        isinstance(checkpoint_identity, str)
        and checkpoint_identity.startswith("sha256:")
        and len(checkpoint_identity) == 71
        and all(character in "0123456789abcdef" for character in checkpoint_identity[7:])
    )
    analysis["checkpoint_identity"] = checkpoint_identity if is_sha256 else None
    if checkpoint_identity is not None and not is_sha256:
        analysis["warnings"].append(
            "checkpoint_identity must be a SHA-256 digest; the supplied value was omitted to avoid exposing paths or sensitive identifiers."
        )
    return analysis


def _architecture_signature(model: MultimodalCTRCDClassifier) -> dict[str, Any]:
    config = model.configuration()
    fusion = config["fusion"]
    classifier = config["classifier"]
    return {
        "class_labels": tuple(config["class_labels"]),
        "fusion": tuple(fusion[name] for name in ("image_dim", "clinical_dim", "fusion_dim", "output_dim")),
        "classifier": (tuple(classifier["hidden_dims"]), classifier["dropout"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_selected_checkpoint(
    checkpoint_path: str | Path,
    test_data: Any,
    *,
    train_data: Any,
    validation_data: Any,
    device: str = "cpu",
    expected_model: MultimodalCTRCDClassifier | None = None,
) -> dict[str, Any]:
    """Safely load and evaluate a locked checkpoint on a held-out test split.

    This function reports a SHA-256 checkpoint identity without returning its
    filesystem path. Test metrics are never used for selection. Split checks
    delegate to Module 13 and are explicitly marked structural; mapping and
    patient-identity provenance cannot be inferred from opaque IDs here.
    """
    path = Path(checkpoint_path)
    model, metadata = load_checkpoint(path, device=device)
    if expected_model is not None:
        if not isinstance(expected_model, MultimodalCTRCDClassifier):
            raise TypeError("expected_model must be a MultimodalCTRCDClassifier")
        if _architecture_signature(expected_model) != _architecture_signature(model):
            raise ValueError("checkpoint architecture is incompatible with expected_model")
    if metadata.get("class_order") != list(CTRCD_CLASS_LABELS):
        raise ValueError("checkpoint class order is incompatible with the CTRCD evaluation schema")

    trainer = MultimodalTrainer(TrainingConfig(device=device))
    metrics = trainer.evaluate(
        model,
        test_data,
        train_data=train_data,
        validation_data=validation_data,
    )
    matrix = metrics["confusion_matrix_labels_0_1"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    class_metrics = {
        "0": {
            "precision": tn / (tn + fn) if tn + fn else None,
            "recall": tn / (tn + fp) if tn + fp else None,
            "f1": 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else None,
        },
        "1": {
            "precision": metrics["precision_class_1"],
            "recall": metrics["recall_class_1_sensitivity"],
            "f1": metrics["f1_class_1"],
        },
    }
    return {
        "evaluation_scope": "held_out_test_only; not for model or threshold selection",
        "checkpoint_sha256": _sha256(path),
        "class_order": list(CTRCD_CLASS_LABELS),
        "positive_class": {"label": 1, "meaning": "CTRCD-positive"},
        "metrics": metrics,
        "per_class_metrics": class_metrics,
        "split_integrity": {
            "record_id_disjointness": "structurally_checked",
            "patient_group_id_disjointness": "structurally_checked",
            "patient_identity_provenance": "unverified",
            "image_clinical_pairing_provenance": "unverified",
            "patient_level_independence": "unverified",
        },
        "cohort_provenance": "not supplied; unverified",
        "real_cohort_evidence": "not claimed",
    }
