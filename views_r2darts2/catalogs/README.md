# catalogs

The **catalogs** layer is the Genomic Firewall between the DNA manifest (a plain Python `dict`) and live PyTorch/Darts objects. Every model, loss function, optimizer, and scheduler in the system is instantiated through exactly one catalog class — never directly. This enforces that no object can be created without a validated config.

---

## Files

| File | Class | Responsibility |
|------|-------|----------------|
| `model_catalog.py` | `ModelCatalog` | Translates `config["algorithm"]` → a fully wired Darts `TorchForecastingModel` with callbacks and loss |
| `loss_catalog.py` | `LossCatalog` | Translates `config["loss_function"]` → a `torch.nn.Module` with correct constructor kwargs |
| `optimizer_catalog.py` | `OptimizerCatalog` | Translates `config["optimizer_cls"]` → a `torch.optim` class and its validated kwargs |
| `scheduler_catalog.py` | `SchedulerCatalog` | Translates `config["lr_scheduler_cls"]` → a `torch.optim.lr_scheduler` class and its kwargs |

---

## Design contract

Every catalog follows the same three-step pattern:

1. **Receive config** — the raw DNA dict is injected at `__init__`
2. **Validate via `ReproducibilityGate`** — mandatory genome keys are audited; missing keys raise `MissingHyperparameterError` before any object is created
3. **Return a concrete object** — `get_*()` returns the instantiated class, never a string or a partial

```python
from views_r2darts2.catalogs.loss_catalog import LossCatalog
from views_r2darts2.catalogs.optimizer_catalog import OptimizerCatalog
from views_r2darts2.catalogs.scheduler_catalog import SchedulerCatalog
from views_r2darts2.catalogs.model_catalog import ModelCatalog

loss      = LossCatalog(config).get_loss()
opt_cls   = OptimizerCatalog(config).get_optimizer_cls()
opt_kw    = OptimizerCatalog(config).get_optimizer_kwargs()
sched_cls = SchedulerCatalog(config).get_scheduler_cls()
model     = ModelCatalog(config).get_model("NHiTSModel")
```

---

## LossCatalog

Maps `config["loss_function"]` to a `torch.nn.Module`. Supports every loss in `views_r2darts2.math` plus standard PyTorch losses.

**Supported values for `config["loss_function"]`:**

| Name | Class | Status |
|------|-------|--------|
| `SpotlightLossLogcosh` | `SpotlightLossLogcosh` | ⭐ Production |
| `SpotlightLoss` | `SpotlightLoss` | ⭐ Production |
| `PrismLoss` | `PrismLoss` | Research |
| `SpotlightFocalLoss` | `SpotlightFocalLoss` | Research |
| `SentinelLoss` | `SentinelLoss` | Research |
| `WeightedPenaltyHuberLoss` | `WeightedPenaltyHuberLoss` | Legacy |
| `WeightedHuberLoss` | `WeightedHuberLoss` | Legacy |
| `TimeAwareWeightedHuberLoss` | `TimeAwareWeightedHuberLoss` | Legacy |
| `TweedieLoss` | `TweedieLoss` | Legacy |
| `AsymmetricQuantileLoss` | `AsymmetricQuantileLoss` | Legacy |
| `ZeroInflatedLoss` | `ZeroInflatedLoss` | Legacy |
| `ShrinkageLoss` | `ShrinkageLoss` | Legacy |
| `SpikeFocalLoss` | `SpikeFocalLoss` | Legacy |
| `MSELoss` | `torch.nn.MSELoss` | Baseline |
| `L1Loss` | `torch.nn.L1Loss` | Baseline |
| `HuberLoss` | `torch.nn.HuberLoss` | Baseline |
| `SmoothL1Loss` | `torch.nn.SmoothL1Loss` | Baseline |
| `PoissonNLLLoss` | `torch.nn.PoissonNLLLoss` | Baseline |

The catalog reads `config.get("delta")`, `config.get("non_zero_threshold")`, etc. and passes only the kwargs the target class's `__init__` accepts, using introspection. Unknown keys are silently dropped; missing required keys raise `ValueError`.

---

## OptimizerCatalog

Maps `config["optimizer_cls"]` to a `torch.optim` class via `getattr(torch.optim, name)`. Validates that the required genome keys (`lr`, `weight_decay`, etc.) are present using `ReproducibilityGate.Config.OPTIMIZER_GENOMES`. Optimizer kwargs are **always sourced from the top-level config** — duplicate `optimizer_kwargs` nested dicts are ignored to prevent stale values.

**Supported:** `AdamW`, `Adam`, `SGD`, `RMSprop`, and any other `torch.optim` class by name.

---

## SchedulerCatalog

Maps `config["lr_scheduler_cls"]` to a scheduler class. Handles the translation from `lr_scheduler_*` prefixed config keys to the scheduler's actual constructor kwarg names via an internal `_KWARG_MAP`. Also provides `_STATIC_KWARGS` (e.g., `{"mode": "min"}` for `ReduceLROnPlateau`).

**Supported schedulers:**

| Config value | Class |
|---|---|
| `ReduceLROnPlateau` | `torch.optim.lr_scheduler.ReduceLROnPlateau` |
| `CosineAnnealingWarmRestarts` | `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` |
| `WarmupCAWR` | `views_r2darts2.math.warmup_cawr.WarmupCAWR` |
| `StepLR` | `torch.optim.lr_scheduler.StepLR` |
| `ExponentialLR` | `torch.optim.lr_scheduler.ExponentialLR` |

---

## ModelCatalog

The most complex catalog. Given a config, it:

1. Resolves the loss function via `LossCatalog`
2. Resolves the optimizer via `OptimizerCatalog`
3. Resolves the scheduler via `SchedulerCatalog`
4. Injects cyclic encoders if `use_cyclic_encoders=True`
5. Wires all mandatory Fortress callbacks: `NaNDetectionCallback`, `GradientHealthCallback`, `WeightNormCallback`, `LossStabilityCallback`, `RevINMonitorCallback`, `PredictionSanityCallback`, `TrainingStepPatchCallback`, `YHatBarCallback`, `EpochTimingCallback`, plus `EarlyStopping` and `LearningRateMonitor`
6. Audits the constructed model for architectural compatibility (output chunk length, static covariate support)
7. Returns a fully initialised `TorchForecastingModel` ready for `.fit()`

**Supported algorithms** (value of `config["algorithm"]`):

`NBEATSModel`, `NHiTSModel`, `TFTModel`, `TCNModel`, `BlockRNNModel`, `TransformerModel`, `TSMixerModel`, `NLinearModel`, `TiDEModel`, `DLinearModel`
