"""Offline Module 10 tests using explicitly synthetic clinical fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import BINARY_COLS, CONTINUOUS_COLS, FEATURE_COLS, TARGET_COL, TIME_COL
from src.models.clinical_encoder import ClinicalEncoderConfig, ClinicalFeaturePreprocessor, ClinicalMLPEncoder


@pytest.fixture
def synthetic_clinical():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({c: rng.normal(size=24) for c in CONTINUOUS_COLS})
    for i, column in enumerate(BINARY_COLS):
        frame[column] = rng.integers(0, 2, size=24)
    frame.loc[0, CONTINUOUS_COLS[0]] = np.nan
    frame.loc[1, BINARY_COLS[0]] = np.nan
    return frame[FEATURE_COLS]


def test_default_encoder_shapes_batch_one_and_multiple():
    encoder = ClinicalMLPEncoder()
    assert encoder.configuration()["hidden_dims"] == (128, 64)
    assert encoder.output_dim == 32
    assert encoder(torch.randn(1, 25)).shape == (1, 32)
    assert encoder(torch.randn(7, 25)).shape == (7, 32)


def test_custom_architecture_activation_dropout_and_seed_does_not_reset_global_rng():
    torch.manual_seed(202)
    expected = torch.rand(3)
    torch.manual_seed(202)
    first = ClinicalMLPEncoder(ClinicalEncoderConfig(hidden_dims=(16, 8), clinical_embedding_dim=5, activation="gelu", dropout=0.25, seed=9))
    observed = torch.rand(3)
    second = ClinicalMLPEncoder(ClinicalEncoderConfig(hidden_dims=(16, 8), clinical_embedding_dim=5, activation="gelu", dropout=0.25, seed=9))
    assert torch.equal(expected, observed)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b)
    assert first(torch.randn(2, 25)).shape == (2, 5)
    assert any(isinstance(layer, torch.nn.GELU) for layer in first.modules())
    assert any(isinstance(layer, torch.nn.Dropout) and layer.p == 0.25 for layer in first.modules())


@pytest.mark.parametrize("bad", [torch.empty(0, 25), torch.randn(2, 24), torch.randn(2, 25, 1), torch.ones(1, 25, dtype=torch.int64)])
def test_encoder_rejects_invalid_tensor_inputs(bad):
    encoder = ClinicalMLPEncoder()
    with pytest.raises((TypeError, ValueError)):
        encoder(bad)


def test_encoder_rejects_nonfinite_and_non_tensor():
    encoder = ClinicalMLPEncoder()
    invalid = torch.zeros(1, 25)
    invalid[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        encoder(invalid)
    with pytest.raises(TypeError, match="torch.Tensor"):
        encoder([[1]])


@pytest.mark.parametrize("kwargs", [
    {"input_dim": 0}, {"hidden_dims": ()}, {"hidden_dims": (8, 0)},
    {"clinical_embedding_dim": -1}, {"dropout": 1.0}, {"activation": "sigmoid"},
])
def test_invalid_encoder_configuration(kwargs):
    with pytest.raises(ValueError):
        ClinicalMLPEncoder(ClinicalEncoderConfig(**kwargs))


def test_cpu_and_conditional_cuda():
    encoder = ClinicalMLPEncoder(ClinicalEncoderConfig(device="cpu"))
    assert encoder(torch.zeros(1, 25)).device.type == "cpu"
    if torch.cuda.is_available():
        cuda_encoder = ClinicalMLPEncoder(ClinicalEncoderConfig(device="cuda"))
        assert cuda_encoder(torch.zeros(1, 25, device="cuda")).device.type == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA was requested"):
            ClinicalMLPEncoder(ClinicalEncoderConfig(device="cuda"))


def test_training_gradients_and_inference_mode_restore():
    encoder = ClinicalMLPEncoder()
    encoder.train()
    output = encoder(torch.randn(3, 25))
    output.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())
    extracted = encoder.extract(torch.randn(2, 25))
    assert extracted.shape == (2, 32) and not extracted.requires_grad
    assert encoder.training
    encoder.eval()
    encoder.extract(torch.randn(1, 25))
    assert not encoder.training


def test_preprocessor_fits_training_partition_only_and_stabilizes_order(synthetic_clinical):
    train = synthetic_clinical.iloc[:16].copy()
    test = synthetic_clinical.iloc[16:].copy()
    preprocessor = ClinicalFeaturePreprocessor().fit(train)
    original_center = preprocessor._transformer.named_transformers_["num"].named_steps["scaler"].center_.copy()
    transformed = preprocessor.transform(test[FEATURE_COLS[::-1]])
    np.testing.assert_array_equal(original_center, preprocessor._transformer.named_transformers_["num"].named_steps["scaler"].center_)
    assert list(transformed.columns) == FEATURE_COLS
    assert transformed.shape == (8, 25) and np.isfinite(transformed.to_numpy()).all()
    assert preprocessor.to_schema()["fit_scope"].startswith("training partition only")


@pytest.mark.parametrize("extra", [TARGET_COL, TIME_COL, "patient_id", "CTRCD_outcome_after_baseline"])
def test_preprocessor_rejects_targets_identifiers_and_unknown_fields(synthetic_clinical, extra):
    frame = synthetic_clinical.copy()
    frame[extra] = 0
    with pytest.raises(ValueError, match="forbidden|schema mismatch"):
        ClinicalFeaturePreprocessor().fit(frame)


def test_preprocessor_rejects_missing_non_numeric_and_invalid_binary_schema(synthetic_clinical):
    with pytest.raises(ValueError, match="missing"):
        ClinicalFeaturePreprocessor().fit(synthetic_clinical.drop(columns=[FEATURE_COLS[0]]))
    non_numeric = synthetic_clinical.copy()
    non_numeric[CONTINUOUS_COLS[0]] = "not numeric"
    with pytest.raises(TypeError, match="must be numeric"):
        ClinicalFeaturePreprocessor().fit(non_numeric)
    invalid_binary = synthetic_clinical.copy()
    invalid_binary[BINARY_COLS[0]] = 2
    with pytest.raises(ValueError, match="only 0, 1"):
        ClinicalFeaturePreprocessor().fit(invalid_binary)


def test_preprocessor_transform_requires_fit(synthetic_clinical):
    with pytest.raises(RuntimeError, match="not fitted"):
        ClinicalFeaturePreprocessor().transform(synthetic_clinical)
