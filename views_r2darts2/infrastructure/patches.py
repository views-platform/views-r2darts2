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


# --- 3. RINorm Dish-TS Patch (Learned Denormalization) ---
#
# ═══════════════════════════════════════════════════════════════════════
# BACKGROUND
# ═══════════════════════════════════════════════════════════════════════
#
# Pipeline: raw counts → asinh → RevIN → model → RevIN⁻¹ → sinh → counts
#
# ALL fixed RevIN variants (v1/v3/v5) share a fatal flaw: the sinh()
# evaluation step exponentially amplifies any bias in asinh-space.
#
# Proof: sinh(asinh(ẑ·σ) + μ) ≈ ẑ·σ·exp(μ) for large ẑ.
# Each +0.1 unit of ẑ-bias at μ=5.69 → +656 raw fatalities.
# No fixed normalization inverse can prevent this — the amplification
# happens AFTER the inverse, in the sinh conversion to raw counts.
#
# ═══════════════════════════════════════════════════════════════════════
# SOLUTION: DISH-TS LEARNED DENORMALIZATION (v6)
# ═══════════════════════════════════════════════════════════════════════
#
# Inspired by Dish-TS (Fan et al., 2023): learned Output CONET replaces
# the fixed inverse. The CONET adjusts μ and σ for denormalization,
# compensating for inter-space distribution shift and Jensen amplification.
#
# Forward (standard linear RevIN — clean z-distribution):
#   μ = mean(x, dim=time)              — asinh-space mean (detached)
#   σ = std(x, dim=time)               — asinh-space std (detached)
#   z = (x − μ) / σ                    — standard z-scores
#   CONET conditioning: [μ, σ, trend_slope] → [Δμ, raw_scale]
#
# Inverse (learned denormalization):
#   μ_out = μ + Δμ                     — learned shift
#   σ_out = σ × (0.5 + sigmoid(s))     — bounded ∈ [0.5σ, 1.5σ]
#   ŷ = ẑ · σ_out + μ_out              — linear denorm with learned stats
#
# ═══════════════════════════════════════════════════════════════════════
# WHY THIS WORKS
# ═══════════════════════════════════════════════════════════════════════
#
# 1. CONET can learn to SHRINK σ_out → directly reduces the Jensen
#    amplification factor exp(σ²/2) at the sinh eval step.
#
# 2. CONET can learn a NEGATIVE Δμ → counteracts the systematic
#    positive bias from E[sinh(X)] > sinh(E[X]).
#
# 3. Identity warm start: at init, Δμ=0, sigmoid(0)=0.5 → σ_out=σ,
#    μ_out=μ. Exact standard RevIN. No risk of degradation.
#
# 4. Bounded corrections: σ_out ∈ [0.5σ, 1.5σ], Δμ starts at 0.
#    The CONET can dampen (fighting Jensen) but never collapse.
#
# 5. ~100 learnable parameters (for n_targets=1). Minimal overhead.
#
# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
#
# Output CONET: Linear(3·n_targets, 16) → GELU → Linear(16, 2·n_targets)
#   Input:  [μ_asinh, σ_asinh, trend_slope] per target (detached)
#   Output: [Δμ, raw_scale] per target
#   Init:   Final layer weights=0, bias=0 → identity at start
#
# Registered as self._output_conet on RINorm → discovered by
# model.parameters() automatically. Backprop through inverse provides
# gradients naturally. No training loop changes needed.
#
# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINT COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════
#
# _load_from_state_dict patched to silently ignore missing CONET keys.
# Old checkpoints load normally — CONET initializes to identity and
# learns from scratch during fine-tuning.


def apply_rinorm_compression_patch():
    """
    Patches Darts RINorm with Dish-TS-inspired learned denormalization (v6).

    Forward: standard linear RevIN in asinh-space (clean z-distribution).
    Inverse: learned Output CONET adjusts μ and σ for denormalization.

    The CONET maps 3 lookback statistics [μ, σ, trend_slope] to 2
    correction coefficients [Δμ, raw_scale] per target:
      - μ_out = μ + Δμ (learned shift)
      - σ_out = σ × (0.5 + sigmoid(raw_scale)) (bounded ∈ [0.5σ, 1.5σ])
      - ŷ = ẑ · σ_out + μ_out

    At initialization the CONET outputs zeros → exact standard RevIN.
    ~100 learnable parameters for n_targets=1.
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm, '_dish_ts_patched', False):
        return  # Already patched

    _original_init = RINorm.__init__

    def _dish_ts_init(self, input_dim, eps=1e-5, affine=True):
        _original_init(self, input_dim, eps=eps, affine=affine)

        # Output CONET: [μ, σ, trend_slope] → [Δμ, raw_scale] per target
        self._output_conet = nn.Sequential(
            nn.Linear(3 * input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2 * input_dim),
        )
        # Identity init: final layer zeros → Δμ=0, sigmoid(0)=0.5 → σ_out=σ
        nn.init.zeros_(self._output_conet[2].weight)
        nn.init.zeros_(self._output_conet[2].bias)
        self._n_targets = input_dim
        self._dish_step = 0

    def _dish_ts_forward(self, x: torch.Tensor):
        # x: (batch, input_chunk_length, n_targets) in asinh-space
        calc_dims = tuple(range(1, x.ndim - 1))

        # Standard linear RevIN normalization (detached statistics)
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

        z = (x - self.mean) / self.stdev

        if self.affine:
            z = z * self.affine_weight + self.affine_bias

        # CONET conditioning from detached lookback statistics
        T = x.shape[1]
        mu_sq = self.mean.squeeze(1)                    # (B, n_targets)
        sigma_sq = self.stdev.squeeze(1)                # (B, n_targets)
        trend = ((x[:, -1, :] - x[:, 0, :]) / max(T - 1, 1)).detach()  # (B, n_targets)

        conet_in = torch.cat([mu_sq, sigma_sq, trend], dim=-1)  # (B, 3*n_targets)
        conet_out = self._output_conet(conet_in)                # (B, 2*n_targets)

        n = self._n_targets
        self._delta_mu = conet_out[:, :n].unsqueeze(1)    # (B, 1, n_targets)
        self._raw_scale = conet_out[:, n:].unsqueeze(1)   # (B, 1, n_targets)

        # Log CONET diagnostics every 500 forward passes
        self._dish_step += 1
        if self._dish_step % 500 == 1:
            _dm = self._delta_mu.detach()
            _rs = self._raw_scale.detach()
            _mult = (0.5 + torch.sigmoid(_rs)).detach()
            logger.info(
                f"[Dish-TS step {self._dish_step}] "
                f"μ_asinh={mu_sq.mean():.3f}±{mu_sq.std():.3f}  "
                f"σ_asinh={sigma_sq.mean():.3f}±{sigma_sq.std():.3f}  "
                f"Δμ={_dm.mean():.4f}±{_dm.std():.4f}  "
                f"σ_mult={_mult.mean():.4f}±{_mult.std():.4f}  "
                f"trend={trend.mean():.4f}"
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

        # Learned denormalization via CONET corrections
        mu_out = self.mean + self._delta_mu          # (B, 1, n_targets)
        sigma_out = self.stdev * (
            0.5 + torch.sigmoid(self._raw_scale)     # ∈ [0.5σ, 1.5σ]
        )

        # Broadcast over nr_params dimension
        mu_out = mu_out.unsqueeze(-1)                # (B, 1, n_targets, 1)
        sigma_out = sigma_out.unsqueeze(-1)          # (B, 1, n_targets, 1)

        x = x * sigma_out + mu_out

        # Log inverse diagnostics at same cadence as forward
        if hasattr(self, '_dish_step') and self._dish_step % 500 == 1:
            logger.info(
                f"[Dish-TS inv  step {self._dish_step}] "
                f"μ_out={mu_out.mean():.3f}  "
                f"σ_out={sigma_out.mean():.4f}  "
                f"ŷ_range=[{x.min():.3f}, {x.max():.3f}]  "
                f"sinh(ŷ)_range=[{torch.sinh(x.clamp(-20,20)).min():.1f}, "
                f"{torch.sinh(x.clamp(-20,20)).max():.1f}]"
            )

        return x

    _original_load = RINorm._load_from_state_dict

    def _dish_ts_load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Let standard loading proceed
        _original_load(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        # Remove CONET keys from missing_keys so strict loading doesn't fail
        # when loading pre-CONET checkpoints (CONET inits to identity anyway)
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
        "🐨 Patched RINorm: Dish-TS learned denormalization v6 "
        "(standard linear forward + Output CONET inverse)"
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