"""Offline training tests use explicitly synthetic paired embeddings."""

import copy
import inspect
import json
import types

import numpy as np
import pytest
import torch

from src.models.attention_gated_fusion import AttentionGatedFusion, AttentionGatedFusionConfig
from src.models.multimodal_classifier import MultimodalCTRCDClassifier
from src.training.multimodal_trainer import (
    MultimodalHyperparameterSearch,
    MultimodalTrainer,
    SearchSpace,
    TrainingConfig,
    TrialParameters,
    TuningConfig,
    TrialFailure,
    load_checkpoint,
    make_paired_embedding_dataset,
    make_trial_model,
)


@pytest.fixture(autouse=True)
def deterministic_torch_rng():
    state = torch.random.get_rng_state()
    torch.manual_seed(616)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


def synthetic_data(count=24, *, seed=52, group_prefix=None):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count, dtype=torch.long) % 2
    image = torch.randn(count, 12, generator=generator) + labels[:, None].float() * 0.4
    clinical = torch.randn(count, 5, generator=generator) + labels[:, None].float() * 0.4
    ids = [f"synthetic-{seed}-record-{index}" for index in range(count)]
    groups = [f"{group_prefix or f'synthetic-{seed}-patient'}-{index}" for index in range(count)]
    return make_paired_embedding_dataset(
        image,
        clinical,
        labels,
        image_record_ids=ids,
        clinical_record_ids=list(ids),
        label_record_ids=list(ids),
        patient_group_ids=groups,
    )


def small_model(seed=10):
    fusion = AttentionGatedFusion(AttentionGatedFusionConfig(
        image_dim=12, clinical_dim=5, fusion_dim=8, output_dim=6, seed=seed,
    ))
    from src.models.multimodal_classifier import MultimodalClassifierConfig
    return MultimodalCTRCDClassifier(fusion, MultimodalClassifierConfig(hidden_dims=(5,), dropout=0.1, seed=seed + 1))


def small_trainer(**kwargs):
    settings = {"max_epochs": 4, "patience": 2, "batch_size": 8, "learning_rate": 1e-2, "seed": 77}
    settings.update(kwargs)
    return MultimodalTrainer(TrainingConfig(**settings))


def test_training_updates_parameters_and_returns_finite_history():
    model = small_model()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = small_trainer().fit(model, synthetic_data(), synthetic_data(10, seed=53))
    assert result["status"] == "completed"
    assert 1 <= result["best_epoch"] <= 4
    assert result["class_order"] == [0, 1]
    assert all(np.isfinite(values).all() for values in result["history"].values())
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert result["optimizer"]["name"] == "AdamW"


def test_validation_evaluation_does_not_update_weights_and_restores_mode():
    model = small_model()
    model.train()
    before = copy.deepcopy(model.state_dict())
    metrics = small_trainer().evaluate(
        model,
        synthetic_data(12, seed=54),
        train_data=synthetic_data(seed=52),
        validation_data=synthetic_data(10, seed=53),
    )
    assert model.training
    assert metrics["count"] == 12
    assert metrics["confusion_denominators"] == {"negative": 6, "positive": 6}
    assert set(metrics["class_counts"]) == {"0", "1"}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_early_stopping_and_best_state_restoration(monkeypatch):
    model = small_model()
    trainer = small_trainer(max_epochs=6, patience=1)
    snapshots = []
    calls = {"count": 0}

    def controlled_validation(self, active_model, loader, criterion):
        calls["count"] += 1
        if calls["count"] == 1:
            snapshots.append(copy.deepcopy(active_model.state_dict()))
            probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        else:
            probabilities = np.asarray([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        return 0.5, np.asarray([0, 1, 0, 1]), probabilities

    monkeypatch.setattr(trainer, "_predict", types.MethodType(controlled_validation, trainer))
    result = trainer.fit(model, synthetic_data(), synthetic_data(4, seed=55))
    assert result["best_epoch"] == 1
    assert len(result["history"]["train_loss"]) == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshots[0][name])


def test_checkpoint_round_trip_preserves_logits_and_class_order(tmp_path):
    model = small_model()
    image, clinical = torch.randn(3, 12), torch.randn(3, 5)
    output_dir = tmp_path / "run"
    result = small_trainer().fit(model, synthetic_data(), synthetic_data(10, seed=56), output_dir=output_dir)
    restored, metadata = load_checkpoint(output_dir / "best_model.pt")
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored(image, clinical), model(image, clinical))
    assert metadata["class_order"] == [0, 1]
    assert result["checkpoint_path"].endswith("best_model.pt")
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history["history"]["train_loss"]) >= result["best_epoch"]
    with pytest.raises(FileExistsError):
        small_trainer().fit(small_model(), synthetic_data(), synthetic_data(10, seed=56), output_dir=output_dir)


def test_hyperparameter_search_uses_fresh_models_and_optimizers_and_records_trials(tmp_path, monkeypatch):
    base = small_model()
    built = []
    optimizers = []
    adamw = torch.optim.AdamW

    def tracked_adamw(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", tracked_adamw)

    def factory(parameters: TrialParameters):
        model = make_trial_model(base, parameters)
        built.append(model)
        return model

    search = MultimodalHyperparameterSearch(
        TuningConfig(trial_count=3, search_seed=88, max_epochs=2, patience=1),
        SearchSpace(
            learning_rates=(1e-2,), weight_decays=(0.0,), dropouts=(0.0,),
            hidden_dims=((4,),), batch_sizes=(8,),
        ),
    )
    result = search.search(factory, synthetic_data(), synthetic_data(10, seed=57), output_dir=tmp_path / "tuning")
    assert len(built) == 3 and len({id(model) for model in built}) == 3
    assert len(optimizers) == 3 and len({id(optimizer) for optimizer in optimizers}) == 3
    assert len(result["trials"]) == 3
    assert all(row["status"] == "completed" and "history" in row for row in result["trials"])
    assert sum(row["selected"] for row in result["trials"]) == 1
    assert result["best_trial"]["validation_roc_auc"] == max(row["validation_roc_auc"] for row in result["trials"])
    saved = json.loads((tmp_path / "tuning" / "trials.json").read_text(encoding="utf-8"))
    assert saved["selection_metric"] == "validation_roc_auc"
    assert all((tmp_path / "tuning" / f"trial_{index:03d}" / "best_model.pt").is_file() for index in range(1, 4))
    assert "test_data" not in inspect.signature(search.search).parameters


def test_failed_trials_are_recorded_not_silently_discarded():
    search = MultimodalHyperparameterSearch(
        TuningConfig(trial_count=2, max_epochs=1, patience=1),
        SearchSpace(learning_rates=(1e-2,), weight_decays=(0.0,), dropouts=(0.0,), hidden_dims=((4,),), batch_sizes=(8,)),
    )
    calls = {"count": 0}

    def factory(parameters):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TrialFailure("synthetic trial failure")
        return make_trial_model(small_model(), parameters)

    result = search.search(factory, synthetic_data(), synthetic_data(10, seed=58))
    assert [row["status"] for row in result["trials"]] == ["failed", "completed"]
    assert "synthetic trial failure" in result["trials"][0]["error"]


def test_search_rejects_single_class_validation_before_training():
    labels = torch.zeros(6, dtype=torch.long)
    ids = [f"single-{index}" for index in range(6)]
    single_class = make_paired_embedding_dataset(
        torch.randn(6, 12), torch.randn(6, 5), labels,
        image_record_ids=ids,
        clinical_record_ids=list(ids),
        label_record_ids=list(ids),
        patient_group_ids=[f"single-patient-{index}" for index in range(6)],
    )
    trainer = small_trainer(max_epochs=1)
    with pytest.raises(ValueError, match="both CTRCD classes"):
        trainer.fit(small_model(), synthetic_data(), single_class)
    with pytest.raises(ValueError, match="must contain both CTRCD classes"):
        trainer.evaluate(
            small_model(), single_class,
            train_data=synthetic_data(seed=52),
            validation_data=synthetic_data(8, seed=59),
        )


@pytest.mark.parametrize("image,clinical,labels,error", [
    (torch.empty(0, 12), torch.empty(0, 5), torch.empty(0, dtype=torch.long), ValueError),
    (torch.randn(3, 12), torch.randn(2, 5), torch.tensor([0, 1, 0]), ValueError),
    (torch.randn(3, 12), torch.randn(3, 5), torch.tensor([0, 1, 2]), ValueError),
    (torch.randn(3, 12), torch.randn(3, 5), torch.tensor([0.0, 1.0, 0.0]), TypeError),
    (torch.full((3, 12), float("nan")), torch.randn(3, 5), torch.tensor([0, 1, 0]), ValueError),
])
def test_invalid_dataset_arrays_are_rejected(image, clinical, labels, error):
    count = int(labels.shape[0])
    ids = [f"invalid-record-{index}" for index in range(count)]
    with pytest.raises(error):
        make_paired_embedding_dataset(
            image, clinical, labels,
            image_record_ids=ids,
            clinical_record_ids=list(ids),
            label_record_ids=list(ids),
            patient_group_ids=[f"invalid-patient-{index}" for index in range(count)],
        )


def test_trainer_rejects_dimensions_incompatible_with_fusion():
    incompatible = synthetic_data()
    incompatible.image = torch.randn(24, 11)
    with pytest.raises(ValueError, match="image feature dimension"):
        small_trainer(max_epochs=1).fit(small_model(), incompatible, synthetic_data(8, seed=60))


def test_train_and_validation_must_be_distinct_partitions():
    dataset = synthetic_data()
    with pytest.raises(ValueError, match="separate partitions"):
        small_trainer(max_epochs=1).fit(small_model(), dataset, dataset)


def test_record_alignment_and_patient_group_overlap_are_rejected():
    image = torch.randn(4, 12)
    clinical = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 0, 1])
    ids = ["r0", "r1", "r2", "r3"]
    with pytest.raises(ValueError, match="must match exactly"):
        make_paired_embedding_dataset(
            image, clinical, labels,
            image_record_ids=ids,
            clinical_record_ids=["r1", "r0", "r2", "r3"],
            label_record_ids=ids,
            patient_group_ids=["p0", "p1", "p2", "p3"],
        )
    train = synthetic_data(seed=52)
    validation = synthetic_data(8, seed=72, group_prefix="synthetic-52-patient")
    with pytest.raises(ValueError, match="overlapping patient groups"):
        small_trainer(max_epochs=1).fit(small_model(), train, validation)
    validation = synthetic_data(8, seed=73)
    validation.record_ids = train.record_ids[:8]
    with pytest.raises(ValueError, match="overlapping record identifiers"):
        small_trainer(max_epochs=1).fit(small_model(), train, validation)


def test_training_partition_must_contain_both_classes():
    labels = torch.zeros(6, dtype=torch.long)
    ids = [f"train-single-{index}" for index in range(6)]
    single_class = make_paired_embedding_dataset(
        torch.randn(6, 12), torch.randn(6, 5), labels,
        image_record_ids=ids,
        clinical_record_ids=list(ids),
        label_record_ids=list(ids),
        patient_group_ids=[f"train-single-patient-{index}" for index in range(6)],
    )
    with pytest.raises(ValueError, match="train_data must contain both"):
        small_trainer(max_epochs=1).fit(small_model(), single_class, synthetic_data(8, seed=74))


def test_training_config_validation_and_cpu_execution():
    result = small_trainer(max_epochs=1).fit(small_model(), synthetic_data(), synthetic_data(8, seed=59))
    assert result["status"] == "completed"
    for kwargs in ({"batch_size": 0}, {"max_epochs": True}, {"learning_rate": float("nan")}, {"weight_decay": -1}):
        with pytest.raises(ValueError):
            MultimodalTrainer(TrainingConfig(**kwargs))
    for seed in (True, 1.5, "3", 2**64):
        with pytest.raises(ValueError, match="seed"):
            MultimodalTrainer(TrainingConfig(seed=seed))
    with pytest.raises(ValueError, match="device"):
        MultimodalTrainer(TrainingConfig(device="cuda:-1"))


def test_dataset_rejects_malformed_patient_groups_and_id_columns():
    data = synthetic_data(4)
    kwargs = {
        "image_record_ids": list(data.record_ids),
        "clinical_record_ids": list(data.record_ids),
        "label_record_ids": list(data.record_ids),
    }
    with pytest.raises(ValueError, match="patient_group_ids"):
        make_paired_embedding_dataset(
            data.image, data.clinical, data.labels,
            **kwargs, patient_group_ids=["p0", "", "p2", "p3"],
        )
    with pytest.raises(TypeError, match="sequence"):
        make_paired_embedding_dataset(
            data.image, data.clinical, data.labels,
            **kwargs, patient_group_ids="not-a-column",
        )


def test_unexpected_search_errors_are_recorded_then_reraised(tmp_path):
    search = MultimodalHyperparameterSearch(
        TuningConfig(trial_count=2, max_epochs=1, patience=1),
        SearchSpace(learning_rates=(1e-2,), weight_decays=(0.0,), dropouts=(0.0,), hidden_dims=((4,),), batch_sizes=(8,)),
    )
    calls = {"count": 0}

    def factory(parameters):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TypeError("synthetic programming error")
        return make_trial_model(small_model(), parameters)

    with pytest.raises(TypeError, match="synthetic programming error"):
        search.search(factory, synthetic_data(), synthetic_data(10, seed=76), output_dir=tmp_path / "search")
    report = json.loads((tmp_path / "search" / "trials.json").read_text(encoding="utf-8"))
    assert report["trials"][0]["status"] == "error"
    assert "TypeError: synthetic programming error" in report["trials"][0]["error"]


def test_metrics_reject_non_normalized_and_one_class_inputs():
    from src.training.multimodal_trainer import _metrics

    with pytest.raises(ValueError, match="sum to 1"):
        _metrics(np.asarray([0, 1]), np.asarray([[0.8, 0.3], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="both classes"):
        _metrics(np.asarray([0, 0]), np.asarray([[0.8, 0.2], [0.7, 0.3]]))


def test_training_reproducibility_preserves_global_rng():
    train, validation = synthetic_data(), synthetic_data(10, seed=75)
    first_model, second_model = small_model(seed=91), small_model(seed=91)
    trainer = small_trainer(seed=92, max_epochs=3)
    torch.manual_seed(501)
    expected_random_values = torch.rand(3)
    torch.manual_seed(501)
    first = trainer.fit(first_model, train, validation)
    observed_random_values = torch.rand(3)
    second = trainer.fit(second_model, train, validation)
    torch.testing.assert_close(observed_random_values, expected_random_values)
    assert first["history"] == second["history"]
    for left, right in zip(first_model.parameters(), second_model.parameters()):
        torch.testing.assert_close(left, right)
