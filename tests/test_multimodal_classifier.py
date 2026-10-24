"""Offline Module 12 tests use deterministic synthetic embeddings only."""

import pytest
import torch

from src.models.attention_gated_fusion import AttentionGatedFusion, AttentionGatedFusionConfig
from src.models.multimodal_classifier import (
    CTRCD_CLASS_LABELS,
    MultimodalClassifierConfig,
    MultimodalCTRCDClassifier,
)


@pytest.fixture(autouse=True)
def deterministic_torch_rng():
    state = torch.random.get_rng_state()
    torch.manual_seed(412)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


@pytest.fixture
def classifier():
    fusion = AttentionGatedFusion(AttentionGatedFusionConfig(
        image_dim=12, clinical_dim=5, fusion_dim=16, output_dim=10, seed=13,
    ))
    return MultimodalCTRCDClassifier(
        fusion=fusion,
        config=MultimodalClassifierConfig(hidden_dims=(8,), dropout=0.2, seed=31),
    )


def test_binary_logits_shape_dtype_and_batch_sizes(classifier):
    assert CTRCD_CLASS_LABELS == (0, 1)
    for count in (1, 4):
        logits = classifier(torch.randn(count, 12), torch.randn(count, 5))
        assert logits.shape == (count, 2)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()


def test_forward_propagates_gradients_through_classifier_and_fusion(classifier):
    image = torch.randn(3, 12, requires_grad=True)
    clinical = torch.randn(3, 5, requires_grad=True)
    logits = classifier(image, clinical)
    targets = torch.tensor([0, 1, 0])
    torch.nn.functional.cross_entropy(logits, targets).backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert clinical.grad is not None and clinical.grad.abs().sum() > 0
    assert classifier.classifier[-1].weight.grad is not None
    assert classifier.fusion.gate.weight.grad is not None


def test_optional_diagnostics_preserve_fusion_weights_and_gate(classifier):
    output = classifier(torch.randn(2, 12), torch.randn(2, 5), return_diagnostics=True)
    assert set(output) == {"logits", "attention_weights", "gate"}
    assert output["logits"].shape == (2, 2)
    assert output["attention_weights"].shape == (2, 2)
    torch.testing.assert_close(output["attention_weights"].sum(dim=1), torch.ones(2))
    assert output["gate"].shape == (2, 16)


def test_predict_proba_normalizes_and_predict_uses_declared_class_order(classifier):
    image, clinical = torch.randn(3, 12), torch.randn(3, 5)
    probabilities = classifier.predict_proba(image, clinical)
    predicted = classifier.predict(image, clinical)
    assert probabilities.shape == (3, 2)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))
    assert torch.equal(predicted, probabilities.argmax(dim=1))
    assert set(predicted.tolist()).issubset(set(CTRCD_CLASS_LABELS))
    assert not probabilities.requires_grad


def test_prediction_restores_training_mode(classifier):
    classifier.train()
    classifier.predict_proba(torch.randn(1, 12), torch.randn(1, 5))
    assert classifier.training
    classifier.eval()
    classifier.predict_proba(torch.randn(1, 12), torch.randn(1, 5))
    assert not classifier.training


def test_seeded_classifier_initialization_is_reproducible_and_rng_safe():
    fusion_config = AttentionGatedFusionConfig(image_dim=12, clinical_dim=5, fusion_dim=16, output_dim=10, seed=3)
    config = MultimodalClassifierConfig(hidden_dims=(7,), seed=0)
    torch.manual_seed(981)
    expected = torch.rand(3)
    torch.manual_seed(981)
    first = MultimodalCTRCDClassifier(AttentionGatedFusion(fusion_config), config)
    observed = torch.rand(3)
    second = MultimodalCTRCDClassifier(AttentionGatedFusion(fusion_config), config)
    torch.testing.assert_close(observed, expected)
    for left, right in zip(first.classifier.parameters(), second.classifier.parameters()):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("kwargs", [
    {"hidden_dims": (8, 0)}, {"hidden_dims": (True,)}, {"hidden_dims": [8]},
    {"dropout": -0.1}, {"dropout": 1.0}, {"dropout": True},
    {"seed": True}, {"seed": 1.2}, {"seed": 2**64},
])
def test_rejects_invalid_classifier_configuration(kwargs):
    with pytest.raises(ValueError):
        MultimodalCTRCDClassifier(config=MultimodalClassifierConfig(**kwargs))


@pytest.mark.parametrize("image,clinical,error", [
    (None, torch.randn(1, 5), TypeError),
    (torch.empty(0, 12), torch.empty(0, 5), ValueError),
    (torch.randn(2, 12), torch.randn(1, 5), ValueError),
    (torch.randn(1, 11), torch.randn(1, 5), ValueError),
    (torch.randn(1, 12), torch.randn(1, 4), ValueError),
    (torch.ones(1, 12, dtype=torch.long), torch.randn(1, 5), TypeError),
])
def test_missing_and_incompatible_inputs_are_rejected(classifier, image, clinical, error):
    with pytest.raises(error):
        classifier(image, clinical)


def test_cpu_execution():
    model = MultimodalCTRCDClassifier()
    logits = model(torch.randn(1, 1280), torch.randn(1, 32))
    assert logits.device.type == "cpu"
    if torch.cuda.is_available():
        fusion = AttentionGatedFusion(AttentionGatedFusionConfig(
            image_dim=12, clinical_dim=5, fusion_dim=16, output_dim=10, device="cuda",
        ))
        cuda_model = MultimodalCTRCDClassifier(fusion=fusion)
        cuda_logits = cuda_model(torch.randn(1, 12, device="cuda"), torch.randn(1, 5, device="cuda"))
        assert cuda_logits.device.type == "cuda"
