# Loss Cards

One card per custom loss function: the formula, the mandatory DNA genes, the scale-sensitivity of
each parameter, and — where an audit was run — the behavioural profile (cowardice signal, seed
sensitivity). Cards are specifications, not implementation docs; the implementation is under
`views_r2darts2/math/` and the registry is `LossCatalog` + `ReproducibilityGate.Config.LOSS_GENOMES`.

**No card lists a constructor default.** Every gene in `LOSS_GENOMES` is mandatory (ADR-003). Where
a card shows a value, it is a typical starting point that must still be declared in the DNA.

## Cards (8 losses, 9 files)

| Loss | Card(s) | Module |
|---|---|---|
| `AsymmetricQuantileLoss` | `asymmetric_quantile_loss_spec.md` | `views_r2darts2/math/asymmetric_quantile_loss.py` |
| `ShrinkageLoss` | `shrinkage_loss_spec.md` | `views_r2darts2/math/shrinkage_loss.py` |
| `SpikeFocalLoss` | `spike_focal_loss_spec.md` | `views_r2darts2/math/spike_focal_loss.py` |
| `TimeAwareWeightedHuberLoss` | `time_aware_weighted_huber_loss_spec.md` | `views_r2darts2/math/time_aware_weighted_huber_loss.py` |
| `TweedieLoss` | `tweedie_loss_spec.md`, `tweedie.md` (short form + audit) | `views_r2darts2/math/tweedie_loss.py` |
| `WeightedHuberLoss` | `weighted_huber_loss_spec.md` | `views_r2darts2/math/weighted_huber_loss.py` |
| `WeightedPenaltyHuberLoss` | `weighted_penalty_huber_loss_spec.md`, `weighted_penalty_huber.md` (short form + audit) | `views_r2darts2/math/weighted_penalty_huber_loss.py` |
| `ZeroInflatedLoss` | `zero_inflated_loss_spec.md` | `views_r2darts2/math/zero_inflated_loss.py` |

## Registered losses with no card (12 of 20 — register C-41)

The **production loss family** is entirely uncarded:

- `SpotlightLoss` — `views_r2darts2/math/spotlight_loss.py`
- `SpotlightLossLogcosh` — `views_r2darts2/math/spotlight_loss_logcosh.py`
- `SpotlightLossAsinh` — `views_r2darts2/math/spotlight_loss_asinh.py`
- `SpotlightLossHuber` — `views_r2darts2/math/spotlight_loss_huber.py`
- `SpotlightLossPowerLaw` — `views_r2darts2/math/spotlight_loss_power_law.py`
- `SpotlightFocalLoss` — `views_r2darts2/math/spotlight_focal_loss.py`
- `PrismLoss` — `views_r2darts2/math/prism_loss.py`
- `SentinelLoss` — `views_r2darts2/math/sentinel_loss.py`
- `CharbonnierLoss` — `views_r2darts2/math/charbonnier_loss.py`

Passthrough / thin wrappers, low priority:

- `HuberLoss` — `views_r2darts2/math/huber_loss.py`
- `LogCoshLoss` — `views_r2darts2/math/logcosh_loss.py`
- `MSELoss` — `views_r2darts2/math/mse_loss.py`

`views_r2darts2/math/README.md` (on the code side) gives an architectural overview of the Spotlight
family and is the best current substitute for the missing cards. Writing the cards is blocked on the
same gap as register C-06: the family has no behavioural tests to draw an audit profile from.

## Related

- `reports/guides/loss_function_tuning_guide.md` — scale-dependent parameter tuning per pipeline
- `tests/losses/harness.py` — the integrity harness the audit profiles were produced with
