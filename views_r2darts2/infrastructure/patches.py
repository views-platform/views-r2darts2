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


# --- 3. RINorm Hybrid-Space Normalization Patch ---
#
# ═══════════════════════════════════════════════════════════════════════
# PROBLEM (v1: standard RevIN in asinh-space)
# ═══════════════════════════════════════════════════════════════════════
#
# Pipeline: raw counts → asinh → RevIN → model → RevIN⁻¹ → sinh → counts
#
# Standard RevIN: z = (x_asinh - μ_asinh) / σ_asinh.
# Inverse: ŷ_asinh = ẑ·σ_asinh + μ_asinh → counts = sinh(ŷ_asinh).
#
# Three pathologies:
#   1. Jensen amplification: E[sinh(μ + σ·Z)] = sinh(μ)·exp(σ²/2).
#   2. Sensitivity explosion: ∂counts/∂ẑ = σ·cosh(ẑ·σ+μ) → exponential.
#   3. Asymmetric errors in count space.
#
# ═══════════════════════════════════════════════════════════════════════
# PROBLEM (v2: pure raw-space RevIN — PREVIOUS IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════
#
# Pure raw-space centering: z = asinh((sinh(x) - μ_raw) / σ_raw)
# Inverse: ŷ = asinh(sinh(ẑ)·σ_raw + μ_raw)
#
# Gradient-safe (bounded ≈1 for large |ẑ|): ✓
# Jensen-safe for MC dropout (E[sinh(ẑ)]=0): ✓
#
# BUT: systematic positive mean bias.
#
# At ẑ=0 (model's "neutral" output):
#   ŷ_asinh = asinh(0·σ + μ_raw) = asinh(μ_raw)
#
# Meanwhile the true asinh-space mean is μ_asinh = E[x_asinh].
# Since sinh is convex for x>0:
#   μ_raw = E[sinh(x_asinh)] ≥ sinh(E[x_asinh]) = sinh(μ_asinh)
#   ⟹ asinh(μ_raw) ≥ μ_asinh
#
# The "neutral" prediction overshoots in asinh-space by:
#   bias = asinh(μ_raw) − μ_asinh = asinh(E[sinh(X)]) − E[X]
#
# For Gaussian X~N(μ,σ²): μ_raw = sinh(μ)·exp(σ²/2), so:
#   bias = asinh(sinh(μ)·exp(σ²/2)) − μ
#
#   | Series         | μ_asinh | σ_asinh | bias    |
#   |----------------|---------|---------|---------|
#   | Peaceful       | 0.10    | 0.30    | +0.00   |
#   | Low conflict   | 1.50    | 1.00    | +0.89   |
#   | Medium         | 3.00    | 2.00    | +1.71   |
#   | High           | 5.00    | 3.00    | +2.60   |
#   | Extreme (UKR)  | 5.00    | 4.00    | +7.70   |
#
# The level anchor partially compensates but requires the model to
# learn a per-series negative DC shift from context. Limited capacity
# → chronic overprediction across all series and all architectures.
#
# ═══════════════════════════════════════════════════════════════════════
# FIX: HYBRID-SPACE REVIN (v3)
# ═══════════════════════════════════════════════════════════════════════
#
# Key insight: center in asinh-space (bias-free), normalize variance in
# raw-space (gradient-safe). Best of both worlds.
#
# Forward:
#   μ_asinh = mean(x, dim=time)              — asinh-space mean
#   x_c = sinh(x − μ_asinh)                  — center in asinh, then to raw
#   σ_c = MeanAD(x_c) × √(π/2)              — centered-raw MeanAD scale
#   z = asinh(x_c / σ_c)                     — normalize, compress
#
# Inverse:
#   ŷ_asinh = asinh(sinh(ẑ) · σ_c) + μ_asinh — expand, shift back
#
# ═══════════════════════════════════════════════════════════════════════
# WHY THIS ELIMINATES THE MEAN BIAS
# ═══════════════════════════════════════════════════════════════════════
#
# At ẑ=0: ŷ = asinh(sinh(0)·σ_c) + μ_asinh = asinh(0) + μ_asinh = μ_asinh
#
# The model's neutral output maps EXACTLY to the asinh-space mean.
# No Jensen gap. No per-series DC correction needed.
#
# ═══════════════════════════════════════════════════════════════════════
# WHY GRADIENTS REMAIN BOUNDED
# ═══════════════════════════════════════════════════════════════════════
#
# ∂ŷ/∂ẑ = σ_c · cosh(ẑ) / √(1 + (sinh(ẑ)·σ_c)²)
#
# At ẑ=0: = σ_c (moderate — similar to σ_asinh)
# At ẑ→∞: cosh(ẑ) ≈ sinh(ẑ) → σ_c·sinh(ẑ) / (σ_c·sinh(ẑ)) = 1
#
# Bounded ∈ [1, σ_c]. No exponential explosion.
# The outer asinh cancels the inner sinh curvature.
#
# ═══════════════════════════════════════════════════════════════════════
# MC DROPOUT SAFETY
# ═══════════════════════════════════════════════════════════════════════
#
# The inverse is: asinh(sinh(ẑ)·σ_c) + μ_asinh.  The μ_asinh term is
# additive (constant, no amplification). The sinh(ẑ)·σ_c term:
#   E[sinh(ẑ)] = 0 when E[ẑ]=0 (odd symmetry)
#   Jensen factor for σ_ẑ≈0.3: exp(0.09/2)≈1.05× (negligible)
#
# ═══════════════════════════════════════════════════════════════════════
# Z-RANGE ANALYSIS (with MeanAD-based σ_c)
# ═══════════════════════════════════════════════════════════════════
#
# MeanAD linear influence bounds σ_c between std (quadratic, too large)
# and MAD (zero for >50% identical values, too small).
#
#   | Window type              | std σ_c | MAD σ_c | MeanAD σ_c |
#   |──────────────────────────|─────────|─────────|────────────|
#   | Peaceful                 | ~0.5    | ~0.5    | ~0.5       |
#   | 3 spikes / 36 months     | ~6      | ~0      | ~4         |
#   | 1 spike / 36 months      | ~5.6    | ~0      | ~2.2       |
#   | Syria (many high months) | ~34     | ~15     | ~20        |
#   | Extreme (Ukraine peak)   | ~96-138 | ~15     | ~44        |
#
# z-range is bounded by n: worst case |z| ≤ asinh(n) ≈ 4.3 for n=36.
# Gradient amplifier ∂ŷ/∂ẑ at ẑ=0 = σ_c: always within clip=50.
#
# For peaceful (μ_asinh≈0.1):
#   x − μ ≈ 0 for all t → z ≈ 0 (compact, no gradient contribution)
#
# ═══════════════════════════════════════════════════════════════════════
# ROUND-TRIP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════
#
# Forward: z = asinh(sinh(x − μ) / σ_c)
# Inverse: ŷ = asinh(sinh(z) · σ_c) + μ
#
# Compose: asinh(sinh(asinh(sinh(x−μ)/σ_c)) · σ_c) + μ
#         = asinh((sinh(x−μ)/σ_c) · σ_c) + μ    [sinh(asinh(a)) = a]
#         = asinh(sinh(x−μ)) + μ
#         = (x − μ) + μ
#         = x  ✓  (exact reconstruction)
#
# ═══════════════════════════════════════════════════════════════════════
# IMPLEMENTATION NOTES
# ═══════════════════════════════════════════════════════════════════════
#
# - No learnable parameters. Fully deterministic.
# - No checkpoint compatibility issue (no new state_dict keys).
# - self.mean stores μ_asinh. self.stdev stores σ_c (MeanAD-based scale).
# - Inverse patches shape broadcasting for Darts' 4D output tensor.
# - σ_c = mean(|x_c − mean(x_c)|) × √(π/2) computed from sinh(x − μ_asinh).
#   Linear influence function (|x| not x²) bounds outlier contribution.
#   Never collapses for zero-inflated data (unlike MAD/IQR which give 0
#   when >50% of values are identical).
#   Ref: Geary (1935), Pham-Gia & Hung (2001).
# - eps=1e-5 prevents division by zero for all-zero series.


def apply_rinorm_compression_patch():
    """
    Patches Darts RINorm with hybrid-space normalization (v3).

    Centers in asinh-space (zero mean bias), normalizes variance in
    raw-space (bounded gradients). Eliminates both the gradient
    explosion of standard RevIN AND the systematic positive bias of
    pure raw-space RevIN.

    Forward:  z = asinh(sinh(x − μ_asinh) / σ_c)
    Inverse:  ŷ = asinh(sinh(ẑ) · σ_c) + μ_asinh

    At ẑ=0: ŷ = μ_asinh exactly. No Jensen bias.
    Gradient: bounded ∈ [1, σ_c], no exponential explosion.
    σ_c estimated via MeanAD (robust to zero-inflated spike data).
    Round-trip: exact (forward ∘ inverse = identity).
    Zero learnable parameters. Backwards-compatible state_dict.
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm, '_raw_space_patched', False):
        return  # Already patched

    def _raw_space_forward(self, x: torch.Tensor):
        # x is in asinh-space: (batch, input_chunk_length, n_targets)
        calc_dims = tuple(range(1, x.ndim - 1))

        # Guard against any upstream numeric accident before sinh.
        x = torch.clamp(x, -88.0, 88.0)

        # Compute asinh-space mean (bias-free centering point)
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()

        # Center in asinh space, convert to raw
        x_centered_raw = torch.sinh(x - self.mean)

        # Mean Absolute Deviation (MeanAD) scale estimator (Geary, 1935).
        #
        # Why not std (L2): quadratic influence function → one spike month
        # at sinh(4.2)=33 contributes 33²=1089 to variance, inflating σ_c
        # to 30–138. Gradient amplifier ∂ŷ/∂ẑ(0)=σ_c starves the optimizer
        # (clip_val/grad_norm = 50/383 = 13% pass-through).
        #
        # Why not MAD (median of |x - median|): collapses for zero-inflated
        # data. When >50% of months are peace (identical values), the median
        # of absolute deviations = 0 → σ_c = eps → z-values explode to ±∞
        # for non-zero months → training diverges catastrophically.
        #
        # MeanAD uses the mean (not median) as center point:
        #   σ_c = mean(|x_c - mean(x_c)|) × √(π/2)
        # The mean gets pulled toward spikes, so even in a 90%-zero window,
        # every zero month has a positive deviation from the mean. Linear
        # penalty (|x| not x²) bounds the influence of extreme months while
        # still acknowledging them — outliers are signal, not noise.
        #
        # Properties:
        #   - Never collapses: any nonzero x_c gives σ_c > 0
        #   - Bounded influence: f(x) = sign(x-μ), versus std's f(x) = x²
        #   - 87.6% Gaussian efficiency (vs MAD's 37%, std's 100%)
        #   - √(π/2) ≈ 1.253: Gaussian consistency factor (Geary 1935)
        #   - z-range bounded: |z| ≤ asinh(n) ≈ 4.3 for n=36 (worst case)
        #   - Round-trip exact: σ_c cancels in forward∘inverse
        xc_mean = torch.mean(x_centered_raw, dim=calc_dims, keepdim=True)
        self.stdev = (
            torch.mean(
                torch.abs(x_centered_raw - xc_mean),
                dim=calc_dims, keepdim=True,
            ) * 1.2533141373155003  # √(π/2)
            + self.eps
        ).detach()

        # Normalize by centered-raw std, compress back via asinh
        x = torch.asinh(x_centered_raw / self.stdev)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _raw_space_inverse(self, x: torch.Tensor):
        # x shape: (batch, output_chunk_length, n_targets, nr_params)
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )

        sigma = self.stdev.view(self.stdev.shape + (1,))
        mu = self.mean.view(self.mean.shape + (1,))

        # Clamp before sinh to prevent float32 overflow.
        # ±50 is safe: sinh(50)≈2.59e21; MeanAD-based σ_c typically 2–44,
        # sinh(50)*44≈1.1e23 << 3.4e38.
        x = torch.clamp(x, -50.0, 50.0)

        # Expand: sinh(ẑ) · σ_c gives centered-raw prediction
        # Compress: asinh wraps it back into asinh-space
        # Shift: + μ_asinh re-centers (additive, no amplification)
        x = torch.asinh(torch.sinh(x) * sigma) + mu

        return x

    RINorm.forward = _raw_space_forward
    RINorm.inverse = _raw_space_inverse
    RINorm._raw_space_patched = True
    logger.info(
        "🐨 Patched RINorm: hybrid-space normalization v3 "
        "(asinh centering + MeanAD raw-space σ, zero mean bias)."
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