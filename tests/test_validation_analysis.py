"""Offline Module 15 tests; all model data are synthetic tensors."""

import copy
import inspect

import numpy as np
import pytest
import torch

from src.evaluation.validation_analysis import (
    analyze_training_history,
    analyze_training_result,
    evaluate_selected_checkpoint,
)
from src.models.attention_gated_fusion import AttentionGatedFusion, AttentionGatedFusionConfig
from src.models.multimodal_classifier import MultimodalCTRCDClassifier, MultimodalClassifierConfig
from src.training.multimodal_trainer import (
    MultimodalTrainer,
    TrainingConfig,
    _metrics,
    make_paired_embedding_dataset,
)


def synthetic_dataset(labels, *, prefix, group_prefix=None):
    labels = torch.as_tensor(labels, dtype=torch.long)
    count = len(labels)
    record_ids = [f"{prefix}-record-{index}" for index in range(count)]
    return make_paired_embedding_dataset(
        torch.randn(count, 4),
        torch.randn(count, 3),
        labels,
        image_record_ids=record_ids,
        clinical_record_ids=list(record_ids),
        label_record_ids=list(record_ids),
        patient_group_ids=[f"{group_prefix or prefix}-patient-{index}" for index in range(count)],
    )


def small_model(hidden_dims=(4,)):
    fusion = AttentionGatedFusion(AttentionGatedFusionConfig(
        image_dim=4, clinical_dim=3, fusion_dim=5, output_dim=5, seed=30,
    ))
    return MultimodalCTRCDClassifier(
        fusion,
        MultimodalClassifierConfig(hidden_dims=hidden_dims, dropout=0.1, seed=31),
    )


def test_history_detects_multi_signal_overfitting_pattern_and_reports_gap():
    history = {
        "train_loss": [1.2, 0.9, 0.7, 0.5],
        "validation_loss": [1.1, 0.8, 0.9, 1.0],
        "validation_roc_auc": [0.65, 0.72, 0.70, 0.68],
        "validation_accuracy": [0.60, 0.70, 0.68, 0.66],
    }
    result = analyze_training_history(
        history,
        training_config={"class_weighting": False},
        best_epoch=2,
        selection_metric="validation_roc_auc",
    )
    assert result["status"] == "possible_overfitting_pattern"
    assert result["best_epoch"] == 2
    assert result["selection_metric"] == "validation_roc_auc"
    assert result["training_validation_loss_gap"]["status"] == "comparable_approximate"
    assert result["training_validation_loss_gap"]["final"] == pytest.approx(0.5)
    assert result["validation_metric_for_trend"] == "validation_roc_auc"
    assert result["recommendations"]


def test_stable_improving_history_does_not_warn_of_overfitting():
    result = analyze_training_history({
        "train_loss": [1.0, 0.8, 0.6],
        "validation_loss": [1.1, 0.9, 0.7],
        "validation_roc_auc": [0.55, 0.63, 0.72],
    }, training_config={"class_weighting": False})
    assert result["status"] == "no_clear_recent_divergence"
    assert "overfitting" not in result["assessment"].lower()


def test_history_parser_accepts_numpy_epoch_arrays():
    result = analyze_training_history({
        "train_loss": np.asarray([1.0, 0.8, 0.6]),
        "validation_loss": np.asarray([1.1, 0.9, 0.7]),
    }, training_config={"class_weighting": False})
    assert result["epochs_analyzed"] == 3
    assert result["trends"]["train_loss"]["end"] == pytest.approx(0.6)


def test_flat_history_is_reported_as_stall_or_underfitting_ambiguity():
    result = analyze_training_history({
        "train_loss": [0.7, 0.7, 0.7],
        "validation_loss": [0.8, 0.8, 0.8],
        "validation_accuracy": [0.5, 0.5, 0.5],
    }, training_config={"class_weighting": False})
    assert result["status"] == "no_clear_recent_divergence"
    assert "optimization stall or underfitting" in result["assessment"]
    assert result["training_validation_loss_gap"]["status"] == "comparable_approximate"


@pytest.mark.parametrize("history", [None, {}, {"other_metric": [1.0, 2.0]}])
def test_missing_or_empty_history_returns_insufficient_evidence(history):
    result = analyze_training_history(history)
    assert result["status"] == "insufficient_evidence"
    assert result["epochs_analyzed"] == 0
    assert result["warnings"]


@pytest.mark.parametrize(
    "history,status",
    [
        ({"train_loss": [1.0, float("nan")]}, "invalid_history"),
        ({"train_loss": [1.0, float("inf")]}, "invalid_history"),
        ({"train_loss": [1.0, 0.9], "validation_loss": [1.0]}, "inconsistent_history"),
        ({"train_loss": [], "validation_loss": []}, "invalid_history"),
        ({"train_loss": [1.0, True]}, "invalid_history"),
        ({"train_loss": [-0.1, 0.0]}, "invalid_history"),
        ({"validation_roc_auc": [0.6, 1.2]}, "invalid_history"),
    ],
)
def test_malformed_history_is_explicit_and_never_filled_in(history, status):
    result = analyze_training_history(history)
    assert result["status"] == status
    assert result["epochs_analyzed"] == 0
    assert result["warnings"]


def test_short_history_and_missing_metrics_are_inconclusive():
    result = analyze_training_history({"train_loss": [1.0, 0.9], "validation_loss": [1.1, 1.0]})
    assert result["status"] == "insufficient_evidence"
    assert result["epochs_analyzed"] == 2
    assert any("trend window" in warning for warning in result["warnings"])


def test_weighted_training_loss_gap_is_not_misrepresented_as_comparable():
    result = analyze_training_history(
        {"train_loss": [1.0, 0.8, 0.7], "validation_loss": [1.1, 0.9, 0.8]},
        training_config={"class_weighting": True},
    )
    assert result["status"] == "inconclusive_loss_comparability"
    assert result["training_validation_loss_gap"]["status"] == "not_comparable"
    assert result["training_validation_loss_gap"]["final"] is None


def test_fit_result_analysis_uses_best_epoch_without_test_data():
    result = analyze_training_result({
        "history": {
            "train_loss": [1.0, 0.8, 0.7],
            "validation_loss": [1.1, 0.9, 0.8],
            "validation_roc_auc": [0.6, 0.7, 0.75],
        },
        "training_config": {"class_weighting": False},
        "best_epoch": 3,
        "best_validation_roc_auc": 0.75,
    }, checkpoint_identity=f"sha256:{'a' * 64}")
    assert result["best_epoch"] == 3
    assert result["best_validation_roc_auc"] == pytest.approx(0.75)
    assert result["checkpoint_identity"] == f"sha256:{'a' * 64}"
    assert "test_data" not in inspect.signature(analyze_training_result).parameters


def test_fit_result_analysis_omits_sensitive_checkpoint_path():
    result = analyze_training_result({}, checkpoint_identity="C:/private/patient/checkpoint.pt")
    assert result["checkpoint_identity"] is None
    assert all("patient" not in warning for warning in result["warnings"])


def test_metric_values_use_probabilities_and_class_one_is_positive():
    metrics = _metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]]),
    )
    assert metrics["positive_class"] == {"label": 1, "meaning": "CTRCD-positive"}
    assert metrics["class_counts"] == {"0": 2, "1": 2}
    assert metrics["confusion_matrix_labels_0_1"] == [[2, 0], [0, 2]]
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision_class_1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([0, 2], [[0.8, 0.2], [0.2, 0.8]]),
        ([0, 1], [[0.8], [0.2]]),
        ([0, 1], [[float("nan"), 0.0], [0.2, 0.8]]),
        ([0, 1], [[1.1, -0.1], [0.2, 0.8]]),
        ([0, 1], [[0.8, 0.3], [0.2, 0.8]]),
    ],
)
def test_invalid_labels_or_probabilities_are_rejected(labels, probabilities):
    with pytest.raises(ValueError):
        _metrics(np.asarray(labels), np.asarray(probabilities))


def test_one_class_metrics_are_none_when_undefined():
    metrics = _metrics(np.asarray([0, 0]), np.asarray([[0.8, 0.2], [0.6, 0.4]]))
    assert metrics["roc_auc"] is None
    assert metrics["recall_class_1_sensitivity"] is None
    assert metrics["average_precision_class_1"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["specificity_class_0"] == 1.0


def test_checkpoint_evaluation_is_final_only_safe_and_reports_unverified_provenance(tmp_path):
    train = synthetic_dataset([0, 1, 0, 1, 0, 1], prefix="train")
    validation = synthetic_dataset([0, 1, 0, 1], prefix="val")
    test = synthetic_dataset([0, 1, 0, 1], prefix="test")
    model = small_model()
    checkpoint = tmp_path / "selected.pt"
    MultimodalTrainer(TrainingConfig(max_epochs=1, patience=1, batch_size=4, seed=44)).fit(
        model, train, validation, output_dir=tmp_path / "fit",
    )
    # Evaluate the actual saved checkpoint and ensure the provided architecture matches.
    checkpoint.write_bytes((tmp_path / "fit" / "best_model.pt").read_bytes())
    result = evaluate_selected_checkpoint(
        checkpoint,
        test,
        train_data=train,
        validation_data=validation,
        expected_model=small_model(),
    )
    assert len(result["checkpoint_sha256"]) == 64
    assert "checkpoint_path" not in result
    assert result["evaluation_scope"].startswith("held_out_test_only")
    assert result["class_order"] == [0, 1]
    assert set(result["per_class_metrics"]) == {"0", "1"}
    assert result["split_integrity"]["record_id_disjointness"] == "structurally_checked"
    assert result["split_integrity"]["patient_identity_provenance"] == "unverified"
    assert result["split_integrity"]["patient_level_independence"] == "unverified"
    assert result["real_cohort_evidence"] == "not claimed"
    assert result["cohort_provenance"] == "not supplied; unverified"


def test_selected_checkpoint_evaluation_supports_one_class_test_split(tmp_path):
    train = synthetic_dataset([0, 1, 0, 1, 0, 1], prefix="train")
    validation = synthetic_dataset([0, 1, 0, 1], prefix="val")
    test = synthetic_dataset([0, 0, 0], prefix="test")
    model = small_model()
    out_dir = tmp_path / "fit"
    MultimodalTrainer(TrainingConfig(max_epochs=1, patience=1, batch_size=4)).fit(
        model, train, validation, output_dir=out_dir,
    )
    result = evaluate_selected_checkpoint(
        out_dir / "best_model.pt", test, train_data=train, validation_data=validation,
    )
    assert result["metrics"]["count"] == 3
    assert result["metrics"]["roc_auc"] is None
    tn, fp = result["metrics"]["confusion_matrix_labels_0_1"][0]
    assert result["metrics"]["specificity_class_0"] == pytest.approx(tn / (tn + fp))
    assert result["per_class_metrics"]["1"]["recall"] is None


def test_checkpoint_architecture_mismatch_and_split_overlap_are_rejected(tmp_path):
    train = synthetic_dataset([0, 1, 0, 1, 0, 1], prefix="train")
    validation = synthetic_dataset([0, 1, 0, 1], prefix="val")
    test_with_overlap = synthetic_dataset([0, 1], prefix="test", group_prefix="train")
    out_dir = tmp_path / "fit"
    MultimodalTrainer(TrainingConfig(max_epochs=1, patience=1, batch_size=4)).fit(
        small_model(), train, validation, output_dir=out_dir,
    )
    with pytest.raises(ValueError, match="architecture is incompatible"):
        evaluate_selected_checkpoint(
            out_dir / "best_model.pt", synthetic_dataset([0, 1], prefix="test2"),
            train_data=train, validation_data=validation, expected_model=small_model(hidden_dims=(6,)),
        )
    with pytest.raises(ValueError, match="overlapping patient groups"):
        evaluate_selected_checkpoint(
            out_dir / "best_model.pt", test_with_overlap,
            train_data=train, validation_data=validation,
        )


def test_evaluation_does_not_change_in_memory_model_parameters():
    train = synthetic_dataset([0, 1, 0, 1, 0, 1], prefix="train")
    validation = synthetic_dataset([0, 1, 0, 1], prefix="val")
    test = synthetic_dataset([0, 1], prefix="test")
    model = small_model()
    original_state = copy.deepcopy(model.state_dict())
    original_mode = model.training
    metrics = MultimodalTrainer(TrainingConfig()).evaluate(
        model, test, train_data=train, validation_data=validation,
    )
    assert metrics["count"] == 2
    assert model.training == original_mode
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name])
