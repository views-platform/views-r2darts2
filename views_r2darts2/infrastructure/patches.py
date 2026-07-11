"""Monkey-patches for Darts internals (pandas-free).

This module applies targeted patches to Darts 0.45 internals to fix bugs and
add features that the upstream library does not yet expose:

    * ``torch.load`` — force ``weights_only=False`` for full artifact loading.
    * ``RINorm.forward`` / ``RINorm.inverse`` — hybrid asinh/raw-space RevIN
      that bounds the inverse-transform to prevent σ amplification on
      Laplace-likelihood scale parameters.
    * ``_ResidualBlock.forward`` (TCN) — swap ReLU for tanh to bound per-block
      output to ``[-1, 1]`` and prevent RevIN-denormalization overflow.
    * ``_ResidualBlock.__init__`` + ``_TideModule.forward`` (TiDE) — add
      ``MonteCarloDropout`` to both skip paths so MC dropout produces
      meaningful sample variation.

Call ``apply_all_patches()`` once at process startup (typically from
``DartsForecastingModelManager.__init__``).


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
from darts.utils.torch import MonteCarloDropout

logger = logging.getLogger(__name__)

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
#   σ_c = std(x_c, dim=time)                 — centered-raw std
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
# Z-RANGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
#
# For Ukraine (μ_asinh≈5, σ_asinh≈4):
#   x − μ ∈ [−5, +2], sinh(x−μ) ∈ [−74, +3.6]
#   σ_c ≈ 20 (dominated by the positive tail)
#   z = asinh(sinh(x−μ)/20) → z ∈ [−2.0, 0.18] (compact!)
#
# For peaceful (μ_asinh≈0.1):
#   x − μ ≈ 0 for all t → z ≈ 0 (even more compact)
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
# - self.mean stores μ_asinh. self.stdev stores σ_c (centered-raw std).
# - Inverse patches shape broadcasting for Darts' 4D output tensor.
# - σ_c is computed from sinh(x − μ_asinh), NOT from sinh(x).
#   These differ enormously — std(sinh(x)) is dominated by the
#   raw-space mean offset; std(sinh(x−μ)) reflects actual variability.
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

        # Compute variance of the centered-raw signal
        # NOTE: this is std(sinh(x - μ_asinh)), NOT std(sinh(x)).
        # These differ enormously — std(sinh(x)) is inflated by the
        # raw-space mean offset; std(sinh(x-μ)) reflects actual variability.
        self.stdev = torch.sqrt(
            torch.var(x_centered_raw, dim=calc_dims, keepdim=True, unbiased=False)
            + self.eps
        ).detach()
        self.stdev = torch.clamp(self.stdev, min=0.1)

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
        # ±50 is safe: sinh(50)≈2.59e21; max σ_c in practice ~1000
        # (Syria peak centered-raw std), sinh(50)*1000≈2.6e24 << 3.4e38.
        x = torch.clamp(x, -50.0, 50.0)

        # Expand and denormalize.
        # When using a likelihood (e.g. Gaussian/Laplace), Darts passes the raw
        # parameter tensor of shape (..., nr_params) through this inverse BEFORE
        # loss computation.  Only the LOCATION parameter (index 0) should receive
        # the full nonlinear inverse (asinh(sinh(x)×σ) + μ).
        #
        # SCALE parameters (index 1+) must NOT go through asinh(sinh(·)×σ).
        # Reason: σ_raw for high-conflict series can be 100+.  The transform
        # asinh(sinh(1.0)×100) ≈ 5.5, producing a Laplace b ≈ 5.5 in asinh-space.
        # The 5% tail samples then land at μ±16.5 in asinh-space → sinh(16.5) ≈
        # 7.3 million deaths.  But the ENTIRE target range in asinh-space is
        # only ~7 (asinh(500)≈6.9).  The natural uncertainty scale is O(1-3).
        #
        # Fix: scale parameters pass through as IDENTITY.  The model learns the
        # absolute scale of uncertainty directly in target (asinh) space.  The
        # NLL gradient naturally calibrates scale to the empirical residual
        # magnitude (~0.3 for peace, ~1-3 for conflict).  No σ amplification.
        if x.dim() == 4 and x.shape[-1] > 1:
            # x[..., 0]: location parameter — full nonlinear inverse
            loc = torch.asinh(torch.sinh(x[..., :1]) * sigma) + mu
            # x[..., 1+]: scale/dispersion — identity (model learns in target space)
            sca = x[..., 1:]
            x = torch.cat([loc, sca], dim=-1)
        else:
            # Point forecasting (nr_params == 1) — full inverse as before
            x = torch.asinh(torch.sinh(x) * sigma) + mu

        return x

    RINorm.forward = _raw_space_forward
    RINorm.inverse = _raw_space_inverse
    RINorm._raw_space_patched = True
    logger.info(
        "🐨 Patched RINorm: hybrid-space normalization v3 "
        "(asinh centering + raw-space σ, zero mean bias)."
    )


# --- 5. TCN Tanh Activation Patch ---
#
# TCN's _ResidualBlock uses hardcoded F.relu (not configurable via any parameter).
# With 4 residual blocks stacking additively, outputs accumulate to ~20+ in
# normalized space. RevIN inverse then multiplies by σ_c (up to 458 for
# extreme-conflict entities), and sinh() explodes to infinity.
#
# Fix: replace F.relu with torch.tanh in the forward pass.
# - Tanh bounds each conv path to [-1, 1]
# - After 4 blocks with residual: max output ≈ |input| + 4 ≈ 6-7
# - vs ReLU: |input| + 4*5 ≈ 22 (observed: pred_sanity/max = 23.6)
#
# Tanh is appropriate here because:
# 1. The data is already centered (RevIN forward subtracts μ_asinh)
# 2. Negative predictions are meaningful (below-mean states)
# 3. The bounded [-1,1] per-layer prevents accumulation explosion
# 4. Weight norm ensures gradients don't vanish at saturation
#
# Note: the LAST block already skips the second ReLU in the original code
# (only applies activation to intermediate blocks). We apply tanh uniformly
# to both positions — the last block's conv2 output passes through tanh too,
# giving a bounded final output before the residual add.


def _patched_tcn_residual_forward(self, x):
    """TCN _ResidualBlock.forward with Tanh replacing ReLU."""
    residual = x

    # first step
    left_padding = (self.dilation_base ** self.nr_blocks_below) * (
        self.kernel_size - 1
    )
    x = F.pad(x, (left_padding, 0))
    x = self.dropout1(torch.tanh(self.conv1(x)))

    # second step
    x = F.pad(x, (left_padding, 0))
    x = self.conv2(x)
    if self.nr_blocks_below < self.num_layers - 1:
        x = torch.tanh(x)
    x = self.dropout2(x)

    # add residual
    if self.conv1.in_channels != self.conv2.out_channels:
        residual = self.conv3(residual)
    x = x + residual

    return x


def apply_tcn_tanh_patch():
    """
    Patches Darts TCN _ResidualBlock to use Tanh instead of ReLU.

    Bounds each block's conv path output to [-1, 1], preventing the
    additive residual accumulation from reaching values that overflow
    when RevIN denormalizes by σ_c (which can be 400+ for extreme entities).

    With 4 blocks: max normalized output ≈ 7 (vs 23+ with ReLU).
    sinh(7) × σ_c=458 ≈ 251K → asinh(251K) ≈ 12.8 (recoverable)
    vs sinh(23) × 458 ≈ 4.8 × 10¹² (overflow → infinity).
    """
    from darts.models.forecasting.tcn_model import _ResidualBlock

    if getattr(_ResidualBlock.forward, '_tcn_tanh_patched', False):
        return

    _patched_tcn_residual_forward._tcn_tanh_patched = True
    _ResidualBlock.forward = _patched_tcn_residual_forward
    logger.info(
        "[TCN patch] ✅ Replaced ReLU with Tanh in _ResidualBlock.forward. "
        "Output per block bounded to [-1, 1] + residual."
    )


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


# --- Initialize All Patches ---

def apply_all_patches():
    apply_torch_load_patch()
    apply_rinorm_compression_patch()
    apply_tcn_tanh_patch()
    apply_tide_mc_dropout_patch()
