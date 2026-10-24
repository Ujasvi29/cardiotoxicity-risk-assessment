"""Training-label-derived class weights for binary CTRCD cross-entropy."""

from __future__ import annotations

import torch

from src.models.multimodal_classifier import CTRCD_CLASS_LABELS


def compute_class_weights(
    training_labels: torch.Tensor,
    *,
    class_labels: tuple[int, ...] = CTRCD_CLASS_LABELS,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return inverse-frequency weights normalized to mean 1.

    For ``K`` classes, ``N`` labels, and class count ``n_c``, each weight is
    ``N / (K * n_c)`` in ``class_labels`` order. Every class must be present.
    The caller should pass training-partition labels only. By default, weights
    remain on the labels' device; callers may request the loss's device/dtype.
    """
    if not isinstance(training_labels, torch.Tensor):
        raise TypeError("training_labels must be a torch.Tensor")
    if training_labels.ndim != 1:
        raise ValueError("training_labels must be a rank-1 tensor")
    if training_labels.numel() == 0:
        raise ValueError("training_labels must not be empty")
    if training_labels.dtype == torch.bool or training_labels.is_floating_point() or training_labels.is_complex():
        raise TypeError("training_labels must use an integer dtype")
    if not isinstance(class_labels, tuple) or len(class_labels) != 2:
        raise ValueError("class_labels must be a two-class tuple in classifier logit order")
    if any(not isinstance(label, int) or isinstance(label, bool) for label in class_labels):
        raise TypeError("class_labels must contain integer class IDs")
    if len(set(class_labels)) != len(class_labels):
        raise ValueError("class_labels must contain distinct class IDs")
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if dtype not in supported_dtypes:
        raise TypeError("dtype must be float16, bfloat16, float32, or float64")
    if not torch.isin(training_labels, torch.tensor(class_labels, device=training_labels.device, dtype=training_labels.dtype)).all():
        raise ValueError(f"training_labels must contain only class IDs {class_labels}")

    counts = torch.stack([(training_labels == label).sum() for label in class_labels]).to(torch.float64)
    if (counts == 0).any():
        absent = [label for label, count in zip(class_labels, counts.tolist()) if count == 0]
        raise ValueError(f"training_labels must contain every class; absent classes: {absent}")
    total = float(training_labels.numel())
    values = total / (len(class_labels) * counts)
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("calculated class weights must be finite and positive")
    output_device = training_labels.device if device is None else torch.device(device)
    weights = values.to(device=output_device, dtype=dtype)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("class weights are not representable as finite positive values in the requested dtype")
    return weights
