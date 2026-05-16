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


# --- 3. RINorm v9: Asinh-Space RevIN + Dish-TS HORICONET (linear inverse) ---
#
# ═══════════════════════════════════════════════════════════════════════
# EVOLUTION: v3 → v5 → v7 → v8 → v9
# ═══════════════════════════════════════════════════════════════════════
#
# Pipeline: raw counts → asinh → RevIN → model → RevIN⁻¹ → sinh → counts
#
# v3: σ = std(sinh(x−μ)). Sudan/Niger σ ≈ 80–290, crushing all quiet
#     months to z ≈ 0. Model can't distinguish temporal patterns.
#
# v5: Forward z = sinh(x−μ)/σ, Inverse ŷ = asinh(ẑ·σ)+μ.
#     Matched nonlinearities: exact round-trip. Same σ explosion.
#
# v7: v5 + CONET on inverse. σ_out bounded by exp(tanh). Can't fix
#     forward σ crush. 3 features too sparse.
#
# v8: Decoupled: linear forward z=(x−μ)/σ, asinh inverse. Fixed
#     forward visibility (Sudan z_quiet=−0.17 vs 0.001). BUT:
#     mismatched forward/inverse forces CONET to learn σ_out≈300,
#     re-creating the ẑ-space crush. ẑ=0.0071 for moderate conflict
#     vs ẑ=0 for peace. Gradient amplified 300× by inverse Jacobian.
#
# v9 (current): Linear forward AND linear inverse.
#   Forward: z = (x − μ) / σ          — standard RevIN in asinh-space
#   Inverse: ŷ = ẑ · σ_out + μ_out   — linear denormalization
#
#   Exact round-trip at init: (x−μ)/σ · σ + μ = x.
#   σ_out ≈ σ_asinh ≈ 1.38 for Sudan — no inflation needed.
#   CONET learns small inter-space corrections, not 300× rescaling.
#   Uniform Jacobian: ∂ŷ/∂ẑ = σ_out ≈ 1.38 everywhere.
#
# ═══════════════════════════════════════════════════════════════════════
# WHY LINEAR INVERSE IS SAFE (with SpotlightLossAsinh)
# ═══════════════════════════════════════════════════════════════════════
#
# Previous versions used asinh in the inverse for three safety reasons.
# SpotlightLossAsinh makes all three obsolete:
#
# 1. Gradient explosion: Asinh-integral loss bounds ∂L/∂ŷ to asinh(e).
#    DC/AC split prevents mean accumulation. Linear inverse Jacobian
#    is σ_out ≈ 1–5, so ∂L/∂ẑ ≈ 1–5 × asinh(e). Bounded.
#
# 2. MC Dropout Jensen: Average predictions in asinh space, sinh once.
#    E[ẑ·σ+μ] = E[ẑ]·σ+μ = linear. No Jensen amplification.
#
# 3. OOD runaway: ẑ=10 → ŷ=10×1.38+1.5=15.3 → sinh(15.3)=2.2M.
#    Soft output clamp ŷ ∈ [-20, 20] prevents downstream Inf/NaN.
#    Never binds on real data (asinh(10000)≈9.9 < 20).
#
# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC RANGE COMPARISON
# ═══════════════════════════════════════════════════════════════════════
#
# Sudan (μ=1.5, σ=1.38), required ẑ to predict various targets:
#
#   Target ŷ     | v8 asinh inv (σ_out=300) | v9 linear inv (σ_out=1.38)
#   peace  1.5   | ẑ = 0.0000               | ẑ = 0.00
#   low    2.0   | ẑ = 0.0024               | ẑ = 0.36
#   mod    3.0   | ẑ = 0.0071               | ẑ = 1.09
#   high   5.0   | ẑ = 0.0235               | ẑ = 2.54
#   spike  9.9   | ẑ = 0.0588               | ẑ = 6.09
#
# v9: model distinguishes peace from moderate conflict by outputting
# ẑ=1.09 vs ẑ=0. Clear signal. No sub-percent dynamic range.
#
# ═══════════════════════════════════════════════════════════════════════
# DISH-TS ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════
#
# Paper Eq. 6: ŷ = ξ_h · F_Θ((x − φ_b) / ξ_b) + φ_h
# v9:          ŷ = σ_out · ẑ + μ_out
#
# This IS the Dish-TS denormalization (Eq. 6, right side), with:
#   φ_h = μ_out = μ + Δμ       (HORICONET level coefficient)
#   ξ_h = σ_out = exp(log(σ) + Δlog_σ)  (HORICONET scaling coefficient)
#
# Exact match to the paper's linear denormalization structure.
#
# ═══════════════════════════════════════════════════════════════════════
# HORICONET FEATURES (7 per target)
# ═══════════════════════════════════════════════════════════════════════
#
# 1. μ            — current asinh-space level
# 2. log(σ)       — variability scale (log-compressed)
# 3. trend        — (x[-1] − x[0]) / (T−1)
# 4. zero_frac    — fraction of near-zero timesteps (< 0.88 asinh)
# 5. x_max        — peak asinh value in window (spike magnitude)
# 6. x_last       — last observation (most recent level)
# 7. recent_trend — trend of last 25% of window (acceleration)
#
# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
#
# HORICONET: Linear(7·n_targets, 16) → GELU → Linear(16, 2·n_targets)
#   Output: [Δμ, Δlog_σ] per target — zero-init (identity at start)
#   ~162 params for n_targets=1
#
# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINT COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════
#
# _load_from_state_dict patched to silently ignore missing CONET keys.
# Old checkpoints load normally — CONET initializes to identity.


def apply_rinorm_compression_patch():
    """
    Patches Darts RINorm with v9: asinh-space RevIN + Dish-TS HORICONET
    with linear inverse.

    Forward: z = (x − μ) / σ  (standard RevIN in asinh-space, bounded σ)
    Inverse: ŷ = ẑ · σ_out + μ_out  (linear denormalization, exact Dish-TS Eq. 6)

    HORICONET maps 7 lookback features to [Δμ, Δlog_σ] per target:
      - μ_out = μ + Δμ (learned level shift)
      - σ_out = exp(log(σ) + Δlog_σ) (learned scaling correction)

    Exact round-trip at init: (x−μ)/σ · σ + μ = x.
    Uniform Jacobian: ∂ŷ/∂ẑ = σ_out ≈ 1–5 (no amplification).
    OOD safety: soft clamp on ŷ ∈ [-20, 20] (sinh(20)=242M, never binds).
    ~162 learnable parameters for n_targets=1.
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm, '_dish_ts_patched', False):
        return  # Already patched

    _original_init = RINorm.__init__

    def _dish_ts_init(self, input_dim, eps=1e-5, affine=True):
        _original_init(self, input_dim, eps=eps, affine=affine)

        # HORICONET: 7 features per target → [Δμ, Δlog_σ] per target
        self._output_conet = nn.Sequential(
            nn.Linear(7 * input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2 * input_dim),
        )
        # Zero-init final layer → identity at start (Δμ=0, Δlog_σ=0)
        nn.init.zeros_(self._output_conet[2].weight)
        nn.init.zeros_(self._output_conet[2].bias)
        self._n_targets = input_dim
        self._dish_step = 0

    def _dish_ts_forward(self, x: torch.Tensor):
        # x: (batch, input_chunk_length, n_targets) in asinh-space
        calc_dims = tuple(range(1, x.ndim - 1))

        # Standard RevIN in asinh-space — σ naturally bounded ∈ [~0.3, ~5]
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        x_centered = x - self.mean
        self.stdev = torch.sqrt(
            torch.var(x_centered, dim=calc_dims, keepdim=True, unbiased=False)
            + self.eps
        ).detach()
        z = x_centered / self.stdev

        if self.affine:
            z = z * self.affine_weight + self.affine_bias

        # ── HORICONET features (7 per target, all detached) ──
        T = x.shape[1]
        # 1. μ — current level
        f_mu = self.mean.squeeze(1)                                   # (B, n_targets)
        # 2. log(σ) — variability scale
        f_log_sigma = torch.log(self.stdev.squeeze(1) + self.eps)     # (B, n_targets)
        # 3. trend — overall slope
        f_trend = ((x[:, -1, :] - x[:, 0, :]) / max(T - 1, 1)).detach()
        # 4. zero_frac — fraction of near-zero timesteps (asinh < 0.88 ≈ asinh(1))
        f_zero_frac = (x < 0.88).float().mean(dim=1).detach()        # (B, n_targets)
        # 5. x_max — peak value in window
        f_x_max = x.amax(dim=1).detach()                             # (B, n_targets)
        # 6. x_last — most recent observation
        f_x_last = x[:, -1, :].detach()                              # (B, n_targets)
        # 7. recent_trend — trend of last 25% (acceleration)
        q = max(T // 4, 1)
        f_recent_trend = ((x[:, -1, :] - x[:, -q, :]) / max(q - 1, 1)).detach()

        conet_in = torch.cat([
            f_mu, f_log_sigma, f_trend, f_zero_frac,
            f_x_max, f_x_last, f_recent_trend,
        ], dim=-1)                                                    # (B, 7*n_targets)
        conet_out = self._output_conet(conet_in)                      # (B, 2*n_targets)

        n = self._n_targets
        self._delta_mu = conet_out[:, :n].unsqueeze(1)                # (B, 1, n_targets)
        self._delta_log_sigma = conet_out[:, n:].unsqueeze(1)         # (B, 1, n_targets)

        # Diagnostics every 500 forward passes
        self._dish_step += 1
        if self._dish_step % 500 == 1:
            _dm = self._delta_mu.detach()
            _dls = self._delta_log_sigma.detach()
            logger.info(
                f"[v9 step {self._dish_step}] "
                f"μ={f_mu.mean():.3f}±{f_mu.std():.3f}  "
                f"log_σ={f_log_sigma.mean():.3f}±{f_log_sigma.std():.3f}  "
                f"zero_frac={f_zero_frac.mean():.3f}  "
                f"x_max={f_x_max.mean():.2f}  "
                f"Δμ={_dm.mean():.4f}±{_dm.std():.4f}  "
                f"Δlog_σ={_dls.mean():.4f}±{_dls.std():.4f}"
            )

        return z

    def _dish_ts_inverse(self, x: torch.Tensor):
        # x: (batch, output_chunk_length, n_targets, nr_params)
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )

        # Residual-connected HORICONET corrections
        mu_out = self.mean + self._delta_mu                           # (B, 1, n_targets)
        log_sigma = torch.log(self.stdev + self.eps)
        sigma_out = torch.exp(log_sigma + self._delta_log_sigma)      # unconstrained positive

        # Broadcast over nr_params dimension
        mu_out = mu_out.unsqueeze(-1)                                 # (B, 1, n_targets, 1)
        sigma_out = sigma_out.unsqueeze(-1)                           # (B, 1, n_targets, 1)

        # Linear denormalization: ŷ = ẑ · σ_out + μ_out
        # Exact Dish-TS Eq. 6: uniform Jacobian ∂ŷ/∂ẑ = σ_out
        x = x * sigma_out + mu_out

        # OOD safety clamp — sinh(20) = 242M, never binds on real data
        x = x.clamp(-20.0, 20.0)

        if hasattr(self, '_dish_step') and self._dish_step % 500 == 1:
            logger.info(
                f"[v9 inv step {self._dish_step}] "
                f"μ_out={mu_out.mean():.3f}  "
                f"σ_out={sigma_out.mean():.4f}  "
                f"ŷ_range=[{x.min():.3f}, {x.max():.3f}]"
            )

        return x

    _original_load = RINorm._load_from_state_dict

    def _dish_ts_load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        _original_load(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        # Silently ignore missing CONET keys for pre-v8 checkpoints
        conet_keys = {
            prefix + k for k in self.state_dict() if '_output_conet' in k
        }
        missing_keys[:] = [k for k in missing_keys if k not in conet_keys]

    RINorm.__init__ = _dish_ts_init
    RINorm.forward = _dish_ts_forward
    RINorm.inverse = _dish_ts_inverse
    RINorm._load_from_state_dict = _dish_ts_load_from_state_dict
    RINorm._dish_ts_patched = True
    logger.info(
        "🐨 Patched RINorm: v9 — asinh-space RevIN + Dish-TS HORICONET "
        "(linear inverse, uniform Jacobian, soft output clamp)"
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