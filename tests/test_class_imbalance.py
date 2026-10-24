"""Synthetic tests for training-only inverse-frequency class weighting."""

import numpy as np
import pytest
import torch
from torch import nn

from src.models.attention_gated_fusion import AttentionGatedFusion, AttentionGatedFusionConfig
from src.models.multimodal_classifier import MultimodalCTRCDClassifier, MultimodalClassifierConfig
from src.training.class_imbalance import compute_class_weights
from src.training.multimodal_trainer import (
    MultimodalHyperparameterSearch,
    MultimodalTrainer,
    TrainingConfig,
    TuningConfig,
    _metrics,
    make_paired_embedding_dataset,
)


def paired_data(labels, *, prefix):
    labels = torch.as_tensor(labels, dtype=torch.long)
    count = len(labels)
    return make_paired_embedding_dataset(
        torch.randn(count, 4),
        torch.randn(count, 3),
        labels,
        image_record_ids=[f"{prefix}-r{i}" for i in range(count)],
        clinical_record_ids=[f"{prefix}-r{i}" for i in range(count)],
        label_record_ids=[f"{prefix}-r{i}" for i in range(count)],
        patient_group_ids=[f"{prefix}-p{i}" for i in range(count)],
    )


def tiny_model():
    fusion = AttentionGatedFusion(AttentionGatedFusionConfig(
        image_dim=4, clinical_dim=3, fusion_dim=5, output_dim=5, seed=20,
    ))
    return MultimodalCTRCDClassifier(
        fusion,
        MultimodalClassifierConfig(hidden_dims=(4,), dropout=0.0, seed=21),
    )


@pytest.mark.parametrize(
    "labels,expected",
    [
        ([0, 0, 1, 1], [1.0, 1.0]),
        ([0, 0, 0, 0, 0, 0, 1, 1], [2 / 3, 2.0]),
        ([0] * 99 + [1], [0.5050505, 50.0]),
    ],
)
def test_inverse_frequency_weights_are_normalized_and_finite(labels, expected):
    weights = compute_class_weights(torch.tensor(labels, dtype=torch.long))
    torch.testing.assert_close(weights, torch.tensor(expected), rtol=1e-5, atol=1e-6)
    assert weights.dtype == torch.float32
    assert weights.device.type == "cpu"
    assert torch.isfinite(weights).all()
    class_counts = torch.bincount(torch.tensor(labels), minlength=2)
    mean_sample_weight = (weights * class_counts).sum() / len(labels)
    torch.testing.assert_close(mean_sample_weight, torch.tensor(1.0))


def test_weights_are_reproducible_and_class_order_matches_logits():
    labels = torch.tensor([1, 0, 0, 0, 1, 0])
    first = compute_class_weights(labels, class_labels=(0, 1))
    second = compute_class_weights(labels.clone(), class_labels=(0, 1))
    torch.testing.assert_close(first, second)
    assert first[0] < first[1]
    assert compute_class_weights(labels, dtype=torch.float64).dtype == torch.float64


def test_weighting_configuration_must_be_boolean():
    with pytest.raises(ValueError, match="class_weighting"):
        MultimodalTrainer(TrainingConfig(class_weighting=1))
    with pytest.raises(ValueError, match="class_weighting"):
        MultimodalHyperparameterSearch(TuningConfig(class_weighting=1))


@pytest.mark.parametrize(
    "labels,error,message",
    [
        (torch.empty(0, dtype=torch.long), ValueError, "empty"),
        (torch.tensor([[0, 1]], dtype=torch.long), ValueError, "rank-1"),
        (torch.tensor([0.0, 1.0]), TypeError, "integer dtype"),
        (torch.tensor([False, True]), TypeError, "integer dtype"),
        (torch.tensor([0, 2]), ValueError, "only class IDs"),
        (torch.tensor([1, 1]), ValueError, "absent classes"),
    ],
)
def test_invalid_or_single_class_training_labels_fail_clearly(labels, error, message):
    with pytest.raises(error, match=message):
        compute_class_weights(labels)


def test_weighted_cross_entropy_is_finite_and_backpropagates():
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    weights = compute_class_weights(labels)
    loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_trainer_uses_only_train_labels_and_default_keeps_unweighted_mode():
    train = paired_data([0] * 8 + [1] * 2, prefix="train")
    validation = paired_data([0, 1, 0, 1], prefix="validation")
    test = paired_data([1, 1, 1, 1], prefix="test")
    trainer = MultimodalTrainer(TrainingConfig(
        max_epochs=1, patience=1, batch_size=5, seed=9, class_weighting=True,
    ))
    result = trainer.fit(tiny_model(), train, validation)
    expected = compute_class_weights(train.labels)
    np.testing.assert_allclose(result["class_weights"], expected.numpy())
    assert result["training_class_counts"] == {"0": 8, "1": 2}
    assert result["training_class_proportions"] == {"0": 0.8, "1": 0.2}
    different_validation_prevalence = paired_data([0, 0, 0, 1], prefix="validation-skewed")
    second_result = trainer.fit(tiny_model(), train, different_validation_prevalence)
    np.testing.assert_allclose(second_result["class_weights"], result["class_weights"])
    # Evaluation labels are not inputs to the weight calculation; one-class
    # test evaluation remains reportable with undefined ranking/sensitivity.
    test_metrics = trainer.evaluate(
        tiny_model(), test, train_data=train, validation_data=validation,
    )
    assert test_metrics["class_counts"] == {"0": 0, "1": 4}
    assert test_metrics["roc_auc"] is None
    assert test_metrics["recall_class_1_sensitivity"] == 1.0
    assert test_metrics["specificity_class_0"] is None
    assert test_metrics["positive_class"] == {"label": 1, "meaning": "CTRCD-positive"}

    default_result = MultimodalTrainer(TrainingConfig(max_epochs=1, patience=1, batch_size=5)).fit(
        tiny_model(), train, validation,
    )
    assert default_result["class_weights"] is None


def test_metrics_report_undefined_values_as_none_and_brier_for_valid_probs():
    result = _metrics(
        np.asarray([0, 0]),
        np.asarray([[0.75, 0.25], [0.6, 0.4]]),
    )
    assert result["roc_auc"] is None
    assert result["recall_class_1_sensitivity"] is None
    assert result["balanced_accuracy"] is None
    assert result["specificity_class_0"] == 1.0
    assert result["class_proportions"] == {"0": 1.0, "1": 0.0}
    assert result["brier_score"] == pytest.approx((0.25**2 + 0.25**2 + 0.4**2 + 0.4**2) / 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_weights_can_be_created_on_cuda_for_cuda_loss():
    weights = compute_class_weights(torch.tensor([0, 0, 1]), device="cuda")
    assert weights.device.type == "cuda"
    assert torch.isfinite(weights).all()
