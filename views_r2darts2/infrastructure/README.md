# infrastructure

The **infrastructure** layer provides the cross-cutting concerns that cannot belong to any single domain layer: reproducibility enforcement, training stability monitoring, temporal encoding, runtime patches, and typed exception hierarchy. Every other layer imports from here; this layer imports from nothing inside the package except `math` (for `WarmupCAWR`).

---

## Files

| File | Class(es) | Responsibility |
|------|-----------|----------------|
| `reproducibility_gate.py` | `ReproducibilityGate` | Central invariant enforcer — config, data, temporal, and physics audits |
| `callbacks.py` | 9 callback classes | PyTorch Lightning callbacks for training monitoring and stability |
| `encoders.py` | Module-level functions | Cyclic (sin/cos) time encoders for all VIEWS temporal resolutions |
| `patches.py` | Functions | Runtime monkey-patches for Darts/PyTorch compatibility bugs |
| `exceptions.py` | 8 exception classes | Typed exception hierarchy for all reproducibility failures |

---

## ReproducibilityGate

The single most important class in the package. A namespace of nested static-method classes that enforce physical, temporal, and configuration invariants. **Every other class calls into this at initialization** — no object can be created, trained, or evaluated without passing through the gate.

### Nested namespaces

| Namespace | Methods | What it checks |
|-----------|---------|----------------|
| `ReproducibilityGate.Config` | `audit_manifest()` | All `CORE_GENOME` keys present and non-null; algorithm-specific genes present |
| `ReproducibilityGate.Config` | `audit_architecture()` | `output_chunk_length == len(steps)`; model-specific shape constraints |
| `ReproducibilityGate.Data` | `audit_dataframe_schema()` | Target and feature columns exist in the source dataframe |
| `ReproducibilityGate.Temporal` | `audit_continuity()` | Train end + 1 == test start (no gap, no overlap) |
| `ReproducibilityGate.Physics` | `audit_tensor()` | No NaN or Inf in input tensors at system boundaries |

### CORE_GENOME

The minimum set of config keys required by every experiment regardless of model:

```
random_state, steps, run_type, name, algorithm, loss_function,
lr, weight_decay, batch_size, n_epochs, optimizer_cls, lr_scheduler_cls,
early_stopping_patience, early_stopping_min_delta, gradient_clip_val,
num_samples, mc_dropout
```

Missing any of these raises `MissingHyperparameterError` before the first tensor is allocated.

### NULLABLE_PARAMS

Some Darts constructor args are validly `None` (e.g., `pooling_kernel_sizes=None` means auto-compute). These are whitelisted so the manifest audit doesn't reject them as "missing":

```
hidden_fc_sizes, pooling_kernel_sizes, n_freq_downsample,
categorical_embedding_sizes, temporal_hidden_size_past, temporal_hidden_size_future
```

---

## Exception hierarchy

All exceptions inherit from `ReproducibilityError(Exception)` for easy catch-all handling:

| Exception | Raised when |
|-----------|-------------|
| `MissingHyperparameterError` | A mandatory config key is absent or None |
| `ArchitectureMismatchError` | `output_chunk_length != len(steps)` or model shape mismatch |
| `TemporalDiscontinuityError` | Train and test sets are not contiguous at `t+1` |
| `DataLeakageError` | Test-period time indices appear inside a training tensor |
| `DataStarvationError` | Training window is shorter than the minimum required history |
| `NumericalSanityError` | NaN, Inf, or extreme outlier detected in data or predictions |
| `TemporalHoleError` | Missing months detected in a historical sequence |
| `PredictionHorizonError` | Forecast attempts to extend beyond ground-truth boundary |

---

## Callbacks

Nine PyTorch Lightning `Callback` subclasses are wired into every training run by `ModelCatalog`. All callbacks are defined at module level (not nested) so they are picklable by `torch.save`.

| Callback | Trigger | What it does |
|----------|---------|-------------|
| `TrainingStepPatchCallback` | `on_fit_start` | Monkey-patches `pl_module.training_step` to expose `last_predictions` and `last_targets` for downstream callbacks (Darts' default `training_step` discards predictions) |
| `NaNDetectionCallback` | Each training step | Inspects loss value; raises `NumericalSanityError` immediately on NaN — halts training before corrupt weights propagate |
| `GradientHealthCallback` | Each training step | Computes global gradient norm; logs to W&B; warns if norm > threshold (explosion) or < floor (vanishing) |
| `WeightNormCallback` | Each epoch end | Tracks L2 norm of all named parameters; logs per-layer norm evolution |
| `LossStabilityCallback` | Each epoch end | Detects plateau (loss unchanged for N epochs) and loss spikes (sudden increase); logs warnings |
| `RevINMonitorCallback` | Each epoch end | Monitors RevIN affine parameters (γ, β per series); warns if they drift outside expected range |
| `PredictionSanityCallback` | Each epoch end | Validates output shape and value range; uses stored `last_predictions` from `TrainingStepPatchCallback` |
| `YHatBarCallback` | Each epoch end | Computes per-series mean prediction (ŷ̄) vs per-series mean target (ȳ); logs calibration ratio |
| `EpochTimingCallback` | Each epoch | Records wall-clock time per epoch for performance tracking |

---

## Encoders

Cyclic (sin/cos) time encoders for all VIEWS temporal resolutions. Defined at module level for picklability — Darts serialises `add_encoders` configs via `torch.save`, which requires all callables to be importable by `(module, qualname)`.

| Resolution | Config key | Functions | Period |
|------------|-----------|-----------|--------|
| Monthly (`cm`, `pgm`) | `month_id` | `month_sin`, `month_cos` | 12 |
| Weekly (`cw`, `pgw`) | `week_id` | `week_of_year_sin`, `week_of_year_cos` | 52 |
| Daily (`cd`, `pgd`) | `day_id` | `day_of_week_sin`, `day_of_week_cos`, `day_of_year_sin`, `day_of_year_cos` | 7 / 365 |
| Yearly (`cy`, `pgy`) | — | None (no intra-year cycle) | — |

All use the convention `(idx - 1) % period` so that the first integer in each series maps to phase 0.

`ModelCatalog` reads `config["level"]` (e.g., `"cm"`) to select the correct encoder set and injects it into the Darts `add_encoders` dict when `use_cyclic_encoders=True`.

---

## Patches

Runtime monkey-patches applied at `DartsForecastingModelManager.__init__()` via `apply_all_patches()`:

### `apply_torch_load_patch()`
Overrides `torch.load` to default `weights_only=False`. Required because Darts saves full model artifacts (including non-tensor objects like scalers) that cannot be loaded with the PyTorch 2.x secure default of `weights_only=True`. Applied idempotently — skipped if already patched.

### N-BEATS dropout patch
Fixes a Darts upstream bug where `_Block.__init__` does not correctly forward the `dropout` argument when `MonteCarloDropout` is enabled. The patch replaces `_Block.__init__` with a corrected version that passes `dropout_fn=MonteCarloDropout(p)` to each FC layer as intended.
