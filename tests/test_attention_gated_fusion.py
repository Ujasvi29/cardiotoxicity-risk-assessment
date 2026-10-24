"""Offline tests for attention-gated multimodal representation fusion."""

import pytest
import torch

from src.models.attention_gated_fusion import AttentionGatedFusion, AttentionGatedFusionConfig


@pytest.fixture(autouse=True)
def deterministic_torch_rng():
    state = torch.random.get_rng_state()
    torch.manual_seed(117)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


@pytest.fixture
def fusion():
    return AttentionGatedFusion()


def test_default_module9_module10_dimensions_and_batch_shapes(fusion):
    assert fusion.configuration()["image_dim"] == 1280
    assert fusion.configuration()["clinical_dim"] == 32
    assert fusion.configuration()["fusion_dim"] == fusion.configuration()["output_dim"] == 128
    for batch_size in (1, 4):
        result = fusion(torch.randn(batch_size, 1280), torch.randn(batch_size, 32))
        assert result.shape == (batch_size, 128)
        assert torch.isfinite(result).all()


def test_attention_normalizes_over_two_modalities_and_gate_is_per_coordinate(fusion):
    details = fusion(torch.randn(3, 1280), torch.randn(3, 32), return_details=True)
    attention = details["attention_weights"]
    gate = details["gate"]
    assert attention.shape == (3, 2)
    assert torch.all(attention >= 0) and torch.all(attention <= 1)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))
    assert gate.shape == (3, 128)
    assert torch.all(gate > 0) and torch.all(gate < 1)
    assert details["fused"].shape == (3, 128)


def test_both_modalities_influence_output_and_receive_gradients(fusion):
    image = torch.randn(2, 1280, requires_grad=True)
    clinical = torch.randn(2, 32, requires_grad=True)
    output = fusion(image, clinical)
    output.square().sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert clinical.grad is not None and torch.isfinite(clinical.grad).all() and clinical.grad.abs().sum() > 0
    for layer in (fusion.image_projection, fusion.clinical_projection, fusion.attention_hidden, fusion.attention_score, fusion.gate):
        if isinstance(layer, torch.nn.Linear):
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


def test_distinct_inputs_change_fused_representation(fusion):
    fusion.eval()
    image = torch.zeros(1, 1280)
    clinical = torch.zeros(1, 32)
    baseline = fusion(image, clinical)
    image_changed = fusion(torch.ones_like(image), clinical)
    clinical_changed = fusion(image, torch.ones_like(clinical))
    assert not torch.allclose(baseline, image_changed)
    assert not torch.allclose(baseline, clinical_changed)


def test_seeded_init_is_reproducible_without_global_rng_side_effect():
    config = AttentionGatedFusionConfig(seed=44, fusion_dim=12, output_dim=7)
    torch.manual_seed(321)
    expected = torch.rand(4)
    torch.manual_seed(321)
    first = AttentionGatedFusion(config)
    observed = torch.rand(4)
    second = AttentionGatedFusion(config)
    assert torch.equal(expected, observed)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)
    assert first(torch.randn(1, 1280), torch.randn(1, 32)).shape == (1, 7)


def test_unseeded_initialization_does_not_consume_global_rng():
    torch.manual_seed(852)
    expected = torch.rand(3)
    torch.manual_seed(852)
    AttentionGatedFusion(AttentionGatedFusionConfig(seed=None, fusion_dim=12))
    observed = torch.rand(3)
    torch.testing.assert_close(observed, expected)


def test_train_eval_mode_and_cpu_execution(fusion):
    inputs = (torch.randn(2, 1280), torch.randn(2, 32))
    fusion.train()
    train_out = fusion(*inputs)
    assert fusion.training and train_out.device.type == "cpu"
    fusion.eval()
    eval_out = fusion(*inputs)
    assert not fusion.training
    torch.testing.assert_close(train_out, eval_out)


@pytest.mark.parametrize("image,clinical,error", [
    (None, torch.randn(1, 32), TypeError),
    (torch.empty(0, 1280), torch.empty(0, 32), ValueError),
    (torch.randn(1, 1280, 1), torch.randn(1, 32), ValueError),
    (torch.randn(1, 1279), torch.randn(1, 32), ValueError),
    (torch.randn(1, 1280), torch.randn(1, 31), ValueError),
    (torch.randn(2, 1280), torch.randn(1, 32), ValueError),
    (torch.ones(1, 1280, dtype=torch.int64), torch.randn(1, 32), TypeError),
    (torch.randn(1, 1280), None, TypeError),
])
def test_rejects_invalid_or_missing_modality_inputs(fusion, image, clinical, error):
    with pytest.raises(error):
        fusion(image, clinical)


@pytest.mark.parametrize("which", ["image", "clinical"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rejects_nonfinite_inputs(fusion, which, bad):
    image, clinical = torch.randn(1, 1280), torch.randn(1, 32)
    (image if which == "image" else clinical)[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        fusion(image, clinical)


@pytest.mark.parametrize("kwargs", [
    {"image_dim": 0}, {"clinical_dim": -2}, {"fusion_dim": 0}, {"output_dim": -1},
])
def test_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError, match="positive integers"):
        AttentionGatedFusion(AttentionGatedFusionConfig(**kwargs))


@pytest.mark.parametrize("seed", [True, False, 1.5, "12", -(2**63) - 1, 2**64])
def test_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed must be None or an integer"):
        AttentionGatedFusion(AttentionGatedFusionConfig(seed=seed))


def test_seed_integer_and_none_are_supported():
    assert AttentionGatedFusion(AttentionGatedFusionConfig(seed=0)).configuration()["seed"] == 0
    assert AttentionGatedFusion(AttentionGatedFusionConfig(seed=None)).configuration()["seed"] is None


@pytest.mark.parametrize("seed", [-(2**63), 2**64 - 1])
def test_seed_range_endpoints_are_accepted(seed):
    model = AttentionGatedFusion(AttentionGatedFusionConfig(seed=seed, fusion_dim=4))
    assert model.configuration()["seed"] == seed


def test_rejects_invalid_cuda_index_and_indexed_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA device index 1 is unavailable"):
        AttentionGatedFusion(AttentionGatedFusionConfig(device="cuda:1"))
    with pytest.raises(ValueError, match="CPU device must not include an index"):
        AttentionGatedFusion(AttentionGatedFusionConfig(device="cpu:0"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_floating_input_precision_is_normalized_to_float32(fusion, dtype):
    result = fusion(torch.ones(1, 1280, dtype=dtype), torch.ones(1, 32, dtype=dtype))
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_rejects_inputs_on_wrong_device_and_cuda_is_conditional():
    fusion = AttentionGatedFusion()
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="fusion module"):
            fusion(torch.randn(1, 1280, device="cuda"), torch.randn(1, 32, device="cuda"))
        cuda_fusion = AttentionGatedFusion(AttentionGatedFusionConfig(device="cuda"))
        assert cuda_fusion(torch.randn(1, 1280, device="cuda"), torch.randn(1, 32, device="cuda")).device.type == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA was requested"):
            AttentionGatedFusion(AttentionGatedFusionConfig(device="cuda"))
