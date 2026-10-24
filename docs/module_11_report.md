# Module 11: Attention-Based Gated Multimodal Fusion

## Scope and evidence

This module combines paired image and clinical embeddings into a representation. It does not add a CTRCD prediction head, loss, training loop, evaluation procedure, or multimodal identity mapping. The repository has no separate Module 11 architecture specification or roadmap. The attention and gate below are the smallest implementation choices used to satisfy this module request; they are not asserted to be a previously approved study protocol.

The inspected interfaces establish the default input dimensions: Module 9 EfficientNet-B0 emits 1,280 features unless its optional projection is configured, and Module 10's clinical encoder emits 32 features by default. The Module 10 preprocessing schema specifies 25 baseline fields and excludes `CTRCD` and `time`. The raw dataset, generated split artifacts, image files, and a verified image-to-clinical mapping are absent from this checkout. Thus, no real paired multimodal inputs exist here and no integration on real records was attempted.

## Attention and gated fusion

Given image input `I ∈ R^(B×1280)` and clinical input `C ∈ R^(B×32)`, learned projections produce `H_i, H_c ∈ R^(B×128)`. The defaults match Module 9 and Module 10 outputs; all dimensions are configurable. If an input already has the shared dimension, its projection is identity.

The two representations form modality tokens `H = stack(H_i, H_c) ∈ R^(B×2×128)`. Additive attention computes:

```text
s_b,m = wᵀ tanh(W H_b,m)                 (m ∈ {image, clinical})
α_b,m = softmax(s_b,·)_m                 (softmax over modality axis, axis 1)
```

The attention vector `α ∈ R^(B×2)` is nonnegative and sums to one across the two modalities per record. It weights the contribution of each projected modality.

A sigmoid vector gate is computed from their concatenation:

```text
g = sigmoid(W_g [H_i ; H_c] + b_g)        g ∈ (0,1)^(B×128)
Z = α_image * g ⊙ H_i + α_clinical * (1-g) ⊙ H_c
F = W_o Z + b_o                           F ∈ R^(B×128)
```

Here `g` determines the per-coordinate image share and `1-g` the clinical share; attention additionally scales each modality as a whole. The output projection is identity for the default 128-dimensional output and learned if `output_dim` differs from `fusion_dim`. `forward(..., return_details=True)` optionally returns `fused`, `attention_weights`, and `gate` for diagnostics. The module has no prediction head.

## Configuration and input behavior

Defaults: image dimension 1,280, clinical dimension 32, shared dimension 128, output dimension 128, CPU, and unspecified seed. These are implementation defaults grounded in the current encoder defaults and a configurable shared space; the repository does not prescribe the shared/output sizes or initialization seed. A seed must be `None` or an integer in `[-2**63, 2**64 - 1]` (booleans are rejected). Seeded and unseeded initialization both preserve the caller's global RNG state. Parameters initialize on CPU and then move to the configured device, so initialization does not consume CUDA RNG state.

Both named inputs are required tensors shaped `[batch, configured_dim]`. Empty batches, invalid ranks/dimensions, non-floating and non-finite values, mismatched batch sizes, and inputs not on the configured device are rejected. Indexed CPU devices are rejected; CUDA availability and explicit device indices are validated. Floating inputs are intentionally converted to float32 before the learned layers, whose parameters use float32 by default; this component does not promise mixed-precision execution. The module never fills absent modalities. Its caller must provide verified, same-order paired records. No ordering or identity is inferred here. CPU is supported; CUDA is conditional on the installed PyTorch environment. Gradients pass through projections, attention, gate, and output projection.

## Testing and integration

Focused tests use only deterministic synthetic tensors to verify shapes, normalization, gate range, modality influence, gradients, seed behavior, mode behavior, validation, missing-modality rejection, and conditional device handling. The real dataset and mapping are unavailable, so integration with actual paired cohort records was not tested. Synthetic tests do not establish clinical validity, performance, or that the proposed design is the intended final research protocol.

For real-data integration, the project first needs a provenance-verified image-to-record mapping and matching split assignments. Confirm that each paired record's clinical inputs are baseline-only, use training-fitted preprocessing, then configure dimensions to the actual encoder outputs. The patient-level independence limitation from anonymized identifiers remains unresolved.

## Files

- `src/models/attention_gated_fusion.py`: reusable attention and gated fusion component.
- `tests/test_attention_gated_fusion.py`: offline synthetic unit tests.
- `docs/module_11_report.md`: this report.

No existing encoder, raw data, labels, or patient mappings were changed. No commit or push is part of this work.

## Verification results

Executed using `.venv\\Scripts\\python.exe`:

- `-m pytest -q tests/test_attention_gated_fusion.py` — **36 passed**.
- Relevant encoder/path regressions (`test_efficientnet_encoder.py`, `test_clinical_encoder.py`, `test_paths.py`) — **45 passed**.
- `-m pytest -q` — **88 passed, 53 skipped, 0 failed**. Skips are the existing dataset-dependent integration checks requiring raw CSVs/images or generated processed, split, and mapping artifacts absent from this checkout. The new fusion tests did not skip.
- `-m compileall -q src tests` — passed.
- `git --no-pager diff --check` — passed.

No real paired multimodal data was used. CUDA was unavailable; conditional CUDA behavior was not exercised.
