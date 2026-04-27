import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
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


# --- 3. SC-RevIN: Statistics-Conditioned Reversible Instance Normalization ---
#
# Combines three ideas to address both over-stationarization (canonical shapes)
# and the sinh blowup (uncontrolled magnitude amplification):
#
#   A) log(1+σ) compression — sublinearly compresses σ in the normalization
#      denominator, reducing how much the inverse denorm can amplify.
#        σ_raw=0.5  →  0.41  (peaceful: minimal change)
#        σ_raw=4.0  →  1.61  (Ukraine: 2.5× reduction)
#
#   B) FiLM conditioning (forward) — per-target Feature-wise Linear Modulation
#      from (μ, σ_raw). Breaks canonical shapes: the backbone's nonlinearities
#      (GELU/ReLU) see entity-specific magnitudes, producing different temporal
#      dynamics instead of identical patterns for all countries. Symmetric FiLM
#      cancels for linear models, but N-HiTS has O(9) GELU layers — γ scaling
#      changes which neurons saturate, genuinely breaking shape uniformity.
#
#      z_film = (1 + γ(μ,σ)) · z + β(μ,σ)
#
#      Zero-initialized: γ=0→scale=1, β=0→shift=0 (identity at init).
#
#   C) Learned denorm scale τ (inverse) — Dish-TS-inspired (Fan et al. AAAI 2023)
#      asymmetric denormalization. Instead of mechanically using σ_compressed,
#      the inverse uses:
#
#        σ_denorm = σ_compressed × softplus(τ(μ, σ_raw)) / ln(2)
#
#      where τ is a small learned projection, zero-initialized so that
#      softplus(0)/ln(2) = 1.0 → identity at init.
#
#      Key insight from Dish-TS: the right denormalization scale need not
#      equal the normalization scale. The Non-stationary Transformers paper
#      (Liu et al. NeurIPS 2022) similarly injects statistics asymmetrically
#      into computation (via τ/δ on attention scores). Here, τ modulates the
#      denorm scale — NOT undone in the forward pass.
#
#      During training, SpotlightLoss gradient flows through τ. For entities
#      where the model overshoots (Ukraine), the gradient pushes τ negative
#      → softplus(τ)/ln(2) < 1 → dampened denorm → reduced ŷ in asinh space
#      → bounded sinh output. This is a data-driven, per-entity dampener:
#
#        τ=-2  →  softplus(-2)/ln(2) ≈ 0.18  (strong dampening)
#        τ= 0  →  softplus(0)/ln(2)  = 1.00  (identity)
#        τ=+2  →  softplus(+2)/ln(2) ≈ 3.00  (mild amplification)
#
# Integration: patches RINorm.__init__, .forward, .inverse.
# Works with ALL Darts models via the io_processor decorator.
# Adds 3×Linear(2,1) per target = 9 parameters per target (negligible).


def apply_conditioned_rinorm_patch():
    """
    Patches RINorm with:
      - log(1+σ) compression (forward)
      - FiLM conditioning from (μ, σ_raw) (forward)
      - Learned denorm scale τ from (μ, σ_raw) (inverse, asymmetric)
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm.forward, '_sc_revin', False):
        return  # Already patched

    _orig_init = RINorm.__init__

    def _sc_init(self, input_dim: int, eps=1e-5, affine=True):
        _orig_init(self, input_dim, eps, affine)

        # Per-target FiLM projections: (μ_i, σ_i) → (γ_i, β_i).
        # Independent per target to avoid cross-target leakage.
        self.film_scale = nn.ModuleList([
            nn.Linear(2, 1, bias=True) for _ in range(input_dim)
        ])
        self.film_shift = nn.ModuleList([
            nn.Linear(2, 1, bias=True) for _ in range(input_dim)
        ])

        # Per-target denorm scale projection: (μ_i, σ_i) → τ_i.
        # τ modulates denorm σ via softplus(τ)/ln(2). Asymmetric: not undone.
        self.denorm_tau = nn.ModuleList([
            nn.Linear(2, 1, bias=True) for _ in range(input_dim)
        ])

        # Zero-init all projections → identity at init.
        for proj in list(self.film_scale) + list(self.film_shift) + list(self.denorm_tau):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _sc_forward(self, x: torch.Tensor):
        # x: (B, T, n_targets)
        calc_dims = tuple(range(1, x.ndim - 1))

        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        raw_stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        # Part A: log(1+σ) compression.
        self.stdev = torch.log1p(raw_stdev)

        # Store raw statistics for FiLM/τ projections.
        self._raw_stdev = raw_stdev  # (B, 1, n_targets)

        z = (x - self.mean) / self.stdev

        if self.affine:
            z = z * self.affine_weight
            z = z + self.affine_bias

        # Part B: FiLM conditioning.
        mu_sq = self.mean.squeeze(1)      # (B, n_targets)
        sig_sq = raw_stdev.squeeze(1)     # (B, n_targets)
        n_targets = mu_sq.shape[-1]

        # Build (γ, β) per target from (μ_i, σ_i).
        gamma = torch.cat([
            self.film_scale[i](torch.stack([mu_sq[:, i], sig_sq[:, i]], dim=-1))
            for i in range(n_targets)
        ], dim=-1)  # (B, n_targets)

        beta = torch.cat([
            self.film_shift[i](torch.stack([mu_sq[:, i], sig_sq[:, i]], dim=-1))
            for i in range(n_targets)
        ], dim=-1)  # (B, n_targets)

        # Store for symmetric FiLM inverse.
        self._film_gamma = gamma
        self._film_beta = beta

        # Apply FiLM: γ centered at 1 (zero-init → 1+0=1), β centered at 0.
        # Clamp (1+γ) to prevent degenerate zero-scaling.
        scale = (1.0 + gamma).clamp(min=0.1)
        z = scale.unsqueeze(1) * z + beta.unsqueeze(1)

        return z

    def _sc_inverse(self, x: torch.Tensor):
        # x: (B, T_out, n_targets, nr_params)

        # Step 1: Undo FiLM (symmetric — same scale/shift as forward).
        scale = (1.0 + self._film_gamma).clamp(min=0.1)
        g = scale.unsqueeze(1).unsqueeze(-1)   # (B, 1, n_targets, 1)
        b = self._film_beta.unsqueeze(1).unsqueeze(-1)
        x = (x - b) / g

        # Step 2: Undo affine.
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )

        # Step 3: Learned denorm scale (asymmetric — NOT undone in forward).
        # σ_denorm = σ_compressed × softplus(τ(μ, σ_raw)) / ln(2)
        mu_sq = self.mean.squeeze(1)       # (B, n_targets)
        sig_sq = self._raw_stdev.squeeze(1)
        n_targets = mu_sq.shape[-1]

        tau = torch.cat([
            self.denorm_tau[i](torch.stack([mu_sq[:, i], sig_sq[:, i]], dim=-1))
            for i in range(n_targets)
        ], dim=-1)  # (B, n_targets)

        # softplus(0)/ln(2) = 1.0 → identity at init.
        denorm_factor = F.softplus(tau) / 0.6931471805599453  # ln(2)
        # (B, n_targets) → (B, 1, n_targets, 1)
        denorm_factor = denorm_factor.unsqueeze(1).unsqueeze(-1)

        stdev_view = self.stdev.view(self.stdev.shape + (1,))
        mean_view = self.mean.view(self.mean.shape + (1,))

        x = x * (stdev_view * denorm_factor) + mean_view

        return x

    _sc_forward._sc_revin = True
    RINorm.__init__ = _sc_init
    RINorm.forward = _sc_forward
    RINorm.inverse = _sc_inverse
    logger.info(
        "🐨 SC-RevIN: log(1+σ) compression + FiLM conditioning "
        "+ learned denorm τ (softplus). Model-agnostic, asymmetric inverse."
    )


# --- Initialize All Patches ---

def apply_all_patches():
    apply_torch_load_patch()
    apply_conditioned_rinorm_patch()
    apply_nhits_theta_squash_patch()
    apply_tide_mc_dropout_patch()
    # apply_nbeats_patch()


# --- 3b. N-HiTS Expansion Coefficient Squash ---
#
# Belt-and-suspenders complement to SC-RevIN. The N-HiTS architecture
# produces forecast via interpolated expansion coefficients:
#
#   θ_forecast = Linear(hidden)          ← UNBOUNDED
#   y_hat = F.interpolate(θ_forecast)    ← linearly spreads θ to ocl steps
#
# A single extreme θ coefficient gets interpolated across multiple steps
# (especially in coarse stacks with n_freq_downsample > 1), amplifying
# its effect. After RevIN inverse + sinh, this causes the blowup.
#
# Fix: apply C·tanh(θ/C) to θ_forecast before interpolation.
# This is identity-like for small values (tanh(x)≈x for |x|<<C) and
# smoothly saturates for large values. It's not a hard clamp — it's
# an architectural constraint saying "basis coefficients are bounded."
#
# C=6 is chosen so that:
#   - Normal-range θ (|θ|<3) passes through nearly unchanged
#     (tanh(3/6)×6 = 2.91 vs 3.0 — 3% compression)
#   - Extreme θ (|θ|=10) gets compressed to ±5.97
#   - The hard bound is ±C=±6, which after RevIN denorm + sinh
#     produces reasonable values even for worst-case (μ, σ)


def apply_nhits_theta_squash_patch():
    """
    Patches N-HiTS _Block.forward to apply tanh squashing on θ_forecast
    before interpolation. Bounds expansion coefficients architecturally.
    """
    from darts.models.forecasting.nhits import _Block as NHiTSBlock

    if getattr(NHiTSBlock.forward, '_theta_squashed', False):
        return  # Already patched

    _THETA_BOUND = 6.0

    def _squashed_block_forward(self, x):
        batch_size = x.shape[0]

        # pooling
        x = x.unsqueeze(1)
        x = self.pooling_layer(x)
        x = x.squeeze(1)

        # fully connected layer stack
        x = self.layers(x)

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # Squash forecast expansion coefficients: C·tanh(θ/C).
        # Identity-like for small θ, smoothly saturates for large θ.
        theta_forecast = _THETA_BOUND * torch.tanh(theta_forecast / _THETA_BOUND)

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)

        # interpolate function expects (batch, "channels", time)
        theta_backcast = theta_backcast.unsqueeze(1)

        # interpolate both backcast and forecast from the thetas
        x_hat = F.interpolate(
            theta_backcast, size=self.input_chunk_length, mode="linear"
        )
        y_hat = F.interpolate(
            theta_forecast, size=self.output_chunk_length, mode="linear"
        )

        x_hat = x_hat.squeeze(1)

        # Set the distribution parameters as the last dimension
        y_hat = y_hat.reshape(x.shape[0], self.output_chunk_length, self.nr_params)

        return x_hat, y_hat

    _squashed_block_forward._theta_squashed = True
    NHiTSBlock.forward = _squashed_block_forward
    logger.info(
        "🐨 N-HiTS θ squash: C·tanh(θ/C) with C=%.1f on forecast expansion "
        "coefficients. Architectural bound on interpolation basis.",
        _THETA_BOUND,
    )


# --- 4. TiDE MC Dropout Patch ---
#
# TiDE's architecture has two layers of skip connections that bypass dropout:
#
#   1. _ResidualBlock: out = dense(x) + skip(x)
#      where dense = Linear→ReLU→Linear→MCDropout, skip = Linear (no dropout)
#
#   2. _TideModule.forward: y = temporal_decoded + lookback_skip(x_lookback)
#      where lookback_skip = Linear (no dropout, norm=48, dominates output)
#
# With dropout=0.15 and skip paths carrying >90% of signal, MC dropout
# produces negligible (or zero) variation across samples because:
#   - The stochastic path (encoder→decoder) contributes ~10% of output
#   - Its dropout-induced variation is dwarfed by the deterministic skip
#   - After lookback_skip addition, variation is below float32 precision
#
# Fix: add MonteCarloDropout to both skip paths so the ENTIRE output
# is stochastic under MC dropout, not just a small residual correction.
# The skip dropout uses a reduced rate (dropout * 0.5) to avoid excessive
# noise on the primary signal path. The lookback skip uses (dropout * 0.25)
# since it carries the majority of the signal.


def _patched_residual_block_init(
    self,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    dropout: float,
    use_layer_norm: bool,
):
    """Patched _ResidualBlock.__init__ with MC dropout on skip connection."""
    nn.Module.__init__(self)

    self.dense = nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
        MonteCarloDropout(dropout),
    )

    self.skip = nn.Linear(input_dim, output_dim)
    self.skip_dropout = MonteCarloDropout(dropout * 0.5)

    if use_layer_norm:
        self.layer_norm = nn.LayerNorm(output_dim)
    else:
        self.layer_norm = None


def _patched_residual_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patched _ResidualBlock.forward with MC dropout on skip connection."""
    x = self.dense(x) + self.skip_dropout(self.skip(x))

    if self.layer_norm is not None:
        x = self.layer_norm(x)

    return x


def _patched_tide_module_forward(
    self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
) -> torch.Tensor:
    """Patched _TideModule.forward with MC dropout on lookback skip path.

    Uses F.dropout directly instead of a MonteCarloDropout module to avoid
    the lazy-init timing bug: on_predict_start() activates MC dropout on all
    existing MonteCarloDropout modules BEFORE the first forward() call, so a
    lazily created module would never be activated. We detect MC dropout
    activation by checking _mc_dropout_enabled on any child MonteCarloDropout
    (set by set_mc_dropout() in on_predict_start).
    """
    import torch.nn.functional as F

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

    # temporal decoder with future covariates
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

    # lookback skip WITH MC dropout (direct F.dropout to avoid lazy-init timing bug)
    skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)
    # Detect MC dropout from child MonteCarloDropout modules (set by set_mc_dropout)
    mc_active = any(
        getattr(m, '_mc_dropout_enabled', False)
        for m in self.modules()
        if isinstance(m, MonteCarloDropout)
    )
    drop_active = self.training or mc_active
    skip = F.dropout(skip, self.dropout * 0.25, drop_active)

    y = temporal_decoded + skip.reshape_as(temporal_decoded)
    y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
    return y


def apply_tide_mc_dropout_patch():
    """
    Patches TiDE's _ResidualBlock and _TideModule to inject MonteCarloDropout
    into skip connections, making MC dropout produce meaningful sample variation.
    """
    from darts.models.forecasting.tide_model import _ResidualBlock, _TideModule
    from darts.models.forecasting.pl_forecasting_module import io_processor

    if getattr(_ResidualBlock.forward, '_mc_patched', False):
        return  # Already patched

    # Patch _ResidualBlock
    _ResidualBlock.__init__ = _patched_residual_block_init
    _patched_residual_block_forward._mc_patched = True
    _ResidualBlock.forward = _patched_residual_block_forward

    # Patch _TideModule.forward (replace with dropout-injected version)
    # Uses direct F.dropout on lookback_skip, detecting MC dropout activation
    # from child MonteCarloDropout modules (set by set_mc_dropout in on_predict_start)
    # instead of relying on a lazily-init'd module that would miss activation.
    # Must re-apply @io_processor decorator for RevIN support.
    _TideModule.forward = io_processor(_patched_tide_module_forward)

    logger.info(
        "Successfully patched TiDE for MC dropout: "
        "skip dropout (p×0.5) on _ResidualBlock, "
        "lookback skip dropout (p×0.25) on _TideModule."
    )
