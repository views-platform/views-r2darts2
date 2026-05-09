import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import functools
from typing import Optional, Tuple
from darts.logging import raise_if_not, raise_log, get_logger
from darts.models.forecasting.nbeats import (
    _GType,
    _TrendGenerator,
    _SeasonalityGenerator,
    ACTIVATIONS,
    _Block,
)
from darts.utils.torch import MonteCarloDropout

logger = logging.getLogger(__name__)
darts_logger = get_logger(__name__)

# --- 1. PyTorch Load Patch (weights_only safety) ---

def apply_torch_load_patch():
    """
    Overrides torch.load to ensure weights_only=False by default.
    Necessary for loading full model artifacts in many environments.
    """
    if hasattr(torch, "load") and not getattr(torch.load, "monkeypatched", False):
        # Save original if not already saved
        if not hasattr(torch, "__original_load__"):
            # Try to get clean version from conftest if in test session
            try:
                from tests.conftest import CLEAN_TORCH_LOAD
                torch.__original_load__ = CLEAN_TORCH_LOAD
            except (ImportError, ModuleNotFoundError):
                orig = torch.load
                # Avoid capturing a Mock during testing
                if "Mock" not in str(type(orig)):
                    torch.__original_load__ = orig
                else:
                    return # Already contaminated, skip

        def custom_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return torch.__original_load__(*args, **kwargs)

        custom_torch_load.monkeypatched = True
        torch.load = custom_torch_load
        logger.info("Successfully patched torch.load (weights_only=False default).")

# --- 2. N-BEATS Dropout Patch ---

def _patched_block_init(
    self,
    num_layers: int,
    layer_width: int,
    nr_params: int,
    expansion_coefficient_dim: int,
    input_chunk_length: int,
    target_length: int,
    g_type: _GType,
    batch_norm: bool,
    dropout: float,
    activation: str,
):
    super(_Block, self).__init__()
    self.num_layers = num_layers
    self.layer_width = layer_width
    self.target_length = target_length
    self.nr_params = nr_params
    self.g_type = g_type
    self.dropout_val = dropout
    self.batch_norm = batch_norm
    raise_if_not(activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}")
    self.activation = getattr(nn, activation)()
    self.fc_stack = nn.ModuleList()
    self.bn_stack = nn.ModuleList()
    self.dropout_stack = nn.ModuleList()
    self.fc_stack.append(nn.Linear(input_chunk_length, layer_width))
    for _ in range(num_layers - 1):
        self.fc_stack.append(nn.Linear(layer_width, layer_width))
    if self.batch_norm:
        self.bn_stack.extend(
            [nn.BatchNorm1d(num_features=layer_width) for _ in range(num_layers)]
        )
    if self.dropout_val > 0:
        self.dropout_stack.extend(
            [MonteCarloDropout(p=self.dropout_val) for _ in range(num_layers)]
        )
    if g_type == _GType.SEASONALITY:
        self.backcast_linear_layer = nn.Linear(
            layer_width, 2 * int(input_chunk_length / 2 - 1) + 1
        )
        self.forecast_linear_layer = nn.Linear(
            layer_width, nr_params * (2 * int(target_length / 2 - 1) + 1)
        )
    else:
        self.backcast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)
        self.forecast_linear_layer = nn.Linear(
            layer_width, nr_params * expansion_coefficient_dim
        )
    if g_type == _GType.GENERIC:
        self.backcast_g = nn.Linear(expansion_coefficient_dim, input_chunk_length)
        self.forecast_g = nn.Linear(expansion_coefficient_dim, target_length)
    elif g_type == _GType.TREND:
        self.backcast_g = _TrendGenerator(expansion_coefficient_dim, input_chunk_length)
        self.forecast_g = _TrendGenerator(expansion_coefficient_dim, target_length)
    elif g_type == _GType.SEASONALITY:
        self.backcast_g = _SeasonalityGenerator(input_chunk_length)
        self.forecast_g = _SeasonalityGenerator(target_length)
    else:
        raise_log(ValueError("g_type not supported"), darts_logger)


def _patched_block_forward(self, x):
    batch_size = x.shape[0]
    for i in range(self.num_layers):
        x = self.fc_stack[i](x)
        if self.batch_norm:
            x = self.bn_stack[i](x)
        x = self.activation(x)
        if self.dropout_val > 0:
            x = self.dropout_stack[i](x)
    theta_backcast = self.backcast_linear_layer(x)
    theta_forecast = self.forecast_linear_layer(x)
    theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)
    x_hat = self.backcast_g(theta_backcast)
    y_hat = self.forecast_g(theta_forecast)
    y_hat = y_hat.reshape(x.shape[0], self.target_length, self.nr_params)
    return x_hat, y_hat


def apply_nbeats_patch():
    """
    Patches Darts NBEATSModel to correctly use MonteCarloDropout in its blocks.
    """
    try:
        _Block.__init__ = _patched_block_init
        _Block.forward = _patched_block_forward
        logger.info("Successfully patched Darts NBEATSModel.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during N-BEATS patching: {e}")


# --- 3. RINorm Scale-Only Normalization Patch ---
#
# ═══════════════════════════════════════════════════════════════════════
# PROBLEM (v1–v3: mean-shift RevIN → over-stationarization)
# ═══════════════════════════════════════════════════════════════════════
#
# All previous variants subtract the instance mean in the forward pass
# and add it back in the inverse:
#
#   v1: z = (x − μ) / σ          → ŷ = ẑ·σ + μ
#   v2: z = asinh((sinh(x)−μ_r)/σ_r) → ŷ = asinh(sinh(ẑ)·σ_r + μ_r)
#   v3: z = asinh(sinh(x−μ)/σ_c) → ŷ = asinh(sinh(ẑ)·σ_c) + μ
#
# In all cases, at ẑ=0 the model gets a FREE mean prediction:
#   v1: ŷ = μ        v2: ŷ = asinh(μ_r)      v3: ŷ = μ_asinh
#
# Liu et al. (NeurIPS 2022, "Non-stationary Transformers") term this
# OVER-STATIONARIZATION: "the stationarized series deprived of inherent
# non-stationarity can be less instructive for real-world bursty events
# forecasting."
#
# For 90% zero-inflated conflict data, this produces SMOOTHED forecasts:
# the model learns ẑ≈0 is "good enough" (free mean covers most of the
# loss), and never develops the capacity to predict actual conflict
# dynamics. The temporal structure is erased.
#
# ═══════════════════════════════════════════════════════════════════════
# FIX: TEMPORAL RMS INSTANCE NORMALIZATION (v4)
# ═══════════════════════════════════════════════════════════════════════
#
# Inspired by RMSNorm (Zhang & Sennrich, 2019 — used in LLaMA, Gemma):
# mean subtraction is unnecessary; scale-only normalization works as
# well as or better than full LayerNorm.
#
# Forward:
#   RMS = √(E_t[x²] + ε)          — root mean square over time
#   z = x / RMS                    — scale-only normalization
#
# Inverse:
#   ŷ = ẑ · RMS                    — rescale to original magnitude
#
# ═══════════════════════════════════════════════════════════════════════
# WHY RMS, NOT STD
# ═══════════════════════════════════════════════════════════════════════
#
# std(x) measures spread around the mean. For a flat-but-nonzero series
# (sustained conflict at x≈3.0±0.01), std≈0.01 → z=3.0/0.01=300.
# Catastrophic scale mismatch between flat and variable series.
#
# RMS = √(μ² + σ²) captures BOTH level and variability:
#   - Flat conflict (x≈3):   RMS≈3.0  → z≈1.0     ✓
#   - Variable conflict:     RMS≈√(μ²+σ²) → z∈[-1,3]  ✓
#   - All-zero (peace):      RMS≈√ε   → z≈0, ŷ≈0   ✓
#
# ═══════════════════════════════════════════════════════════════════════
# KEY PROPERTIES
# ═══════════════════════════════════════════════════════════════════════
#
# 1. NO FREE MEAN: at ẑ=0, ŷ=0 (not μ). The model must predict actual
#    conflict levels — no lazy equilibrium at the instance mean.
#
# 2. LEVEL PRESERVATION: z retains the sign and relative magnitude of x.
#    A conflict country (z≈1) looks different from a peace country (z≈0).
#    The model can discriminate without needing separate statistics.
#
# 3. GRADIENT: ∂ŷ/∂ẑ = RMS — constant per series, bounded by data
#    magnitude. No exponential growth, no sinh/cosh in the computation.
#    In asinh-space, max x≈10 → max RMS≈10 → max gradient=10.
#
# 4. NO CLIPPING/CAPPING: purely linear scaling. No hardcoded thresholds.
#    ε is the standard numerical stability constant (1e-5 from Darts).
#
# 5. ROUND-TRIP: z = x/RMS → ẑ·RMS = x (exact when ẑ=z).
#
# 6. MC DROPOUT: ŷ = ẑ·RMS. E[ŷ] = E[ẑ]·RMS. No Jensen bias
#    (linear function of the stochastic variable ẑ).
#
# ═══════════════════════════════════════════════════════════════════════
# Z-RANGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
#
# For Ukraine (x_asinh∈[1, 8], μ≈5, σ≈2):
#   RMS ≈ √(25+4) ≈ 5.4
#   z = x/5.4 ∈ [0.19, 1.48]  — compact, centered around 1
#
# For low conflict (x_asinh∈[0, 2], rare spikes):
#   RMS ≈ 0.5
#   z = x/0.5 ∈ [0, 4]  — spikes visible, zeros preserved
#
# For peaceful (x_asinh≈0 for all t):
#   RMS ≈ √ε ≈ 0.003
#   z = 0/0.003 = 0, ŷ = 0·0.003 = 0  — exact zero preservation
#
# ═══════════════════════════════════════════════════════════════════════
# IMPLEMENTATION NOTES
# ═══════════════════════════════════════════════════════════════════════
#
# - self.mean stores zeros (for Darts state_dict API compatibility).
# - self.stdev stores RMS (temporal root mean square).
# - No sinh/cosh/asinh operations → no overflow risk → no clamps.
# - Backwards-compatible: no new state_dict keys.
# - Affine γ/β are per-channel (shared across batch), cannot learn
#   per-instance levels, so no risk of re-introducing the mean shortcut.


def apply_rinorm_compression_patch():
    """
    Patches Darts RINorm with temporal RMS instance normalization (v4).

    Scale-only normalization using root mean square over time. No mean
    subtraction — preserves the absolute level that distinguishes
    conflict series from peace baselines.

    Addresses the over-stationarization problem (Liu et al., NeurIPS 2022)
    by retaining non-stationary level information in the normalized
    representation. Uses RMS instead of std following RMSNorm
    (Zhang & Sennrich, 2019) to handle flat-but-nonzero series cleanly.

    Forward:  z = x / RMS(x)       RMS = √(mean_t(x²) + ε)
    Inverse:  ŷ = ẑ · RMS(x)

    At ẑ=0: ŷ = 0 (not μ). No forecast smoothing.
    Gradient: ∂ŷ/∂ẑ = RMS — constant, bounded, no exponential growth.
    Round-trip: exact (z·RMS = x).
    No hardcoded values. No clipping. Backwards-compatible state_dict.
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm, '_raw_space_patched', False):
        return  # Already patched

    def _rms_forward(self, x: torch.Tensor):
        # x is in asinh-space: (batch, input_chunk_length, n_targets)
        calc_dims = tuple(range(1, x.ndim - 1))

        # Root mean square over time dimensions.
        # RMS = √(E_t[x²] + ε) captures both level and variability.
        self.stdev = torch.sqrt(
            torch.mean(x * x, dim=calc_dims, keepdim=True) + self.eps
        ).detach()

        # Zero mean for Darts API compatibility (inverse expects self.mean).
        self.mean = torch.zeros_like(self.stdev)

        # Scale-only normalization — level information preserved in z.
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _rms_inverse(self, x: torch.Tensor):
        # x shape: (batch, output_chunk_length, n_targets, nr_params)
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )

        sigma = self.stdev.view(self.stdev.shape + (1,))

        # Rescale to original magnitude. No mean addition — ẑ=0 maps to 0.
        x = x * sigma

        return x

    RINorm.forward = _rms_forward
    RINorm.inverse = _rms_inverse
    RINorm._raw_space_patched = True
    logger.info(
        "🐨 Patched RINorm: temporal RMS instance normalization v4 "
        "(scale-only, no mean shift, no over-stationarization)."
    )


# --- Initialize All Patches ---

def apply_all_patches():
    apply_torch_load_patch()
    apply_rinorm_compression_patch()
    apply_tide_mc_dropout_patch()
    # apply_nbeats_patch()


# --- 4. TiDE MC Dropout Patch ---
#
# TiDE architecture has two deterministic skip paths that dominate the output:
#
#   1. _ResidualBlock: out = dense(x) + skip(x)
#      dense = Linear→ReLU→Linear→MCDropout  (stochastic, ~10% of signal)
#      skip  = Linear                        (deterministic, ~90% of signal)
#
#   2. _TideModule: y = temporal_decoded + lookback_skip(x_lookback)
#      temporal_decoded = through all encoders/decoders   (stochastic after fix)
#      lookback_skip    = Linear(icl, ocl)                (deterministic, dominates)
#
# Without the fix, MC dropout produces negligible variation:
#   - The stochastic dense path contributes ~10% of each residual block output
#   - lookback_skip carries raw target history directly to output, no dropout
#   - Sum of deterministic skip paths overwhelms stochastic dense path
#   - Result: samples are effectively identical
#
# Fix: register MonteCarloDropout on BOTH skip paths as proper nn.Module instances.
# set_mc_dropout(active=True) — called by on_predict_start — iterates
# _get_mc_dropout_modules() which finds ALL MonteCarloDropout children recursively.
# Registered modules are picked up automatically; F.dropout with manual
# mc_active detection is NOT used (fragile, missed by set_mc_dropout).
#
# Dropout rates:
#   _ResidualBlock.skip_dropout     = p × 0.5  (half base rate — skip carries residual)
#   _TideModule.lookback_skip_dropout = p × 0.5 (half base rate — primary signal path)
#
# State dict safety: MonteCarloDropout (nn.Dropout subclass) has no parameters or
# buffers → contributes no keys to state_dict → loading existing checkpoints with
# strict=True is safe.


def _patched_residual_block_init(
    self,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    dropout: float,
    use_layer_norm: bool,
):
    """Patched _ResidualBlock.__init__: adds MonteCarloDropout to skip connection."""
    nn.Module.__init__(self)

    self.dense = nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
        MonteCarloDropout(dropout),
    )

    self.skip = nn.Linear(input_dim, output_dim)
    # Registered module — picked up by set_mc_dropout() automatically.
    self.skip_dropout = MonteCarloDropout(dropout * 0.5)

    if use_layer_norm:
        self.layer_norm = nn.LayerNorm(output_dim)
    else:
        self.layer_norm = None


def _patched_residual_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patched _ResidualBlock.forward: applies skip_dropout on skip connection."""
    x = self.dense(x) + self.skip_dropout(self.skip(x))
    if self.layer_norm is not None:
        x = self.layer_norm(x)
    return x


_original_tide_create_model = None  # set in apply_tide_mc_dropout_patch


def _patched_tide_create_model(self, train_sample):
    """Patched TiDEModel._create_model: registers lookback_skip_dropout on the module.

    Calls the original _create_model (which constructs _TideModule and triggers
    Lightning's save_hyperparameters with no wrapper in the stack), then adds
    lookback_skip_dropout as a registered nn.Module on the fully-built object.

    Avoids patching _TideModule.__init__ entirely, which previously caused
    Lightning's save_hyperparameters() to walk the wrapper frame and crash with
    KeyError: 'args' when trying to resolve named parameters from frame locals.
    """
    module = _original_tide_create_model(self, train_sample)
    # module.dropout is set by the original _TideModule.__init__
    module.lookback_skip_dropout = MonteCarloDropout(module.dropout * 0.5)
    logger.debug(
        f"[TiDE patch] _create_model: registered lookback_skip_dropout "
        f"(p={module.dropout * 0.5:.4f}), "
        f"{sum(1 for m in module.modules() if isinstance(m, MonteCarloDropout))} "
        f"MonteCarloDropout modules total."
    )
    return module


def _patched_tide_module_forward(
    self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
) -> torch.Tensor:
    """Patched _TideModule.forward: applies lookback_skip_dropout on skip path.

    lookback_skip_dropout is a registered MonteCarloDropout module — it is activated
    by set_mc_dropout(True) automatically, same as all other MCDropout modules.
    No manual _mc_dropout_enabled detection is needed.
    """
    x, x_future_covariates, x_static_covariates = x_in

    x_lookback = x[:, :, : self.output_dim]

    # future covariates
    if self.future_cov_dim:
        x_dynamic_future_covariates = torch.cat(
            [
                x[:, :, None if self.future_cov_dim == 0 else -self.future_cov_dim :],
                x_future_covariates,
            ],
            dim=1,
        )
        if self.temporal_width_future:
            x_dynamic_future_covariates = self.future_cov_projection(
                x_dynamic_future_covariates
            )
    else:
        x_dynamic_future_covariates = None

    # past covariates
    if self.past_cov_dim:
        x_dynamic_past_covariates = x[
            :, :, self.output_dim : self.output_dim + self.past_cov_dim
        ]
        if self.temporal_width_past:
            x_dynamic_past_covariates = self.past_cov_projection(
                x_dynamic_past_covariates
            )
    else:
        x_dynamic_past_covariates = None

    # encoder input
    encoded = [
        x_lookback,
        x_dynamic_past_covariates,
        x_dynamic_future_covariates,
        x_static_covariates,
    ]
    encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
    encoded = torch.cat(encoded, dim=1)

    # encode + decode
    encoded = self.encoders(encoded)
    decoded = self.decoders(encoded)
    decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

    # temporal decoder
    temporal_decoder_input = [
        decoded,
        (
            x_dynamic_future_covariates[:, -self.output_chunk_length :, :]
            if self.future_cov_dim > 0
            else None
        ),
    ]
    temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]
    temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
    temporal_decoded = self.temporal_decoder(temporal_decoder_input)

    # lookback skip — apply registered lookback_skip_dropout
    # Lazily register if missing (happens when model is loaded from checkpoint
    # rather than created via _create_model during fit())
    skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)
    if not hasattr(self, "lookback_skip_dropout"):
        self.lookback_skip_dropout = MonteCarloDropout(self.dropout * 0.5)
        self.lookback_skip_dropout.to(skip.device)
    skip = self.lookback_skip_dropout(skip)

    y = temporal_decoded + skip.reshape_as(temporal_decoded)
    y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
    return y


def apply_tide_mc_dropout_patch():
    """
    Patches TiDE to produce meaningful MC dropout sample variation.

    Root cause: TiDE skip connections (ResidualBlock.skip, lookback_skip) are
    deterministic and dominate output. The original MCDropout in dense paths
    contributes <10% of signal — insufficient variation across samples.

    Fix: register MonteCarloDropout on both skip paths as proper nn.Module
    instances so set_mc_dropout(True) activates them alongside dense dropout.

    Implementation note: lookback_skip_dropout is injected via a _create_model
    patch rather than a _TideModule.__init__ patch. Patching __init__ caused
    Lightning's save_hyperparameters() to walk the wrapper frame and crash with
    KeyError: 'args' because frame locals had *args/**kwargs, not named params.
    _create_model receives the fully-built module — no Lightning introspection
    occurs at that level.
    """
    global _original_tide_create_model

    from darts.models.forecasting.tide_model import _ResidualBlock, _TideModule, TiDEModel
    from darts.models.forecasting.pl_forecasting_module import io_processor

    if getattr(_ResidualBlock.forward, '_mc_patched', False):
        logger.debug("[TiDE patch] already applied, skipping.")
        return

    # --- Patch _ResidualBlock ---
    _ResidualBlock.__init__ = _patched_residual_block_init
    _patched_residual_block_forward._mc_patched = True
    _ResidualBlock.forward = _patched_residual_block_forward

    # --- Patch TiDEModel._create_model to register lookback_skip_dropout ---
    _original_tide_create_model = TiDEModel._create_model
    TiDEModel._create_model = _patched_tide_create_model

    # --- Patch _TideModule.forward to use lookback_skip_dropout ---
    _TideModule.forward = io_processor(_patched_tide_module_forward)

    logger.info(
        "[TiDE patch] ✅ installed: "
        "_ResidualBlock.skip_dropout (p×0.5), "
        "_TideModule.lookback_skip_dropout (p×0.5). "
        "Both registered as MonteCarloDropout modules — activated by "
        "set_mc_dropout(True) during on_predict_start."
    )