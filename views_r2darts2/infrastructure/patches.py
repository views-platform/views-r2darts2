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


# --- 3. RINorm log(1+log(1+σ)) Double-Log Compression Patch ---
#
# Bounds scale amplification in RevIN's inverse pass.
#
# Standard RevIN denormalizes as ŷ = ẑ·σ + μ. When σ is large (high-variance
# entities like Ukraine with σ≈4 in asinh space) and the inverse transform is
# convex (sinh), small normalized-space outputs get exponentially amplified:
#
#   Raw σ:    ŷ = 2.5 × 4.0 + 5 = 15.0  →  sinh(15) ≈ 1,634,508 deaths
#   log1p:    ŷ = 2.5 × 1.61 + 5 = 9.025 →  sinh(9)  ≈ 4,143 deaths
#   log²:     ŷ = 2.5 × 0.96 + 5 = 7.40  →  sinh(7.4) ≈ 816 deaths
#
# Double-log compresses σ more aggressively than single log1p, nearly
# eliminating σ as an amplifier for high-variance entities while still
# providing meaningful variance normalization for low-σ ones:
#
#   σ_raw=0.5  →  σ_compressed=0.34    (peaceful: mild change)
#   σ_raw=1.0  →  σ_compressed=0.52    (low-conflict: moderate)
#   σ_raw=4.0  →  σ_compressed=0.96    (Ukraine-level: 4.2× reduction)
#   σ_raw=10   →  σ_compressed=1.22    (extreme: 8.2× reduction)
#
# For the blowup case (ẑ=6.3, Ukraine μ=5):
#   log1p:   ŷ = 6.3 × 1.61 + 5 = 15.14  →  sinh(15.1) ≈ 1,900,000
#   log²:    ŷ = 6.3 × 0.96 + 5 = 11.05  →  sinh(11.0) ≈ 30,000
#
# Tradeoff: mid-range entities (σ≈1.5) see ~30% less amplification than
# with single log1p. The model compensates by learning larger ẑ values
# (training targets also scale up), but this widens the z-spread and
# slightly reduces responsiveness to genuine mid-range spikes.


def apply_rinorm_compression_patch():
    """
    Patches Darts RINorm with log(1+log(1+σ)) double-log compressed stdev.
    Only modifies forward(). The inverse is unchanged.
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm.forward, '_compressed', False):
        return  # Already patched

    def _compressed_forward(self, x: torch.Tensor):
        calc_dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        raw_stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        self.stdev = torch.log1p(torch.log1p(raw_stdev))
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    _compressed_forward._compressed = True
    RINorm.forward = _compressed_forward
    logger.info(
        "🐨 Patched RINorm: log(1+log(1+σ)) double-log compression. "
        "Bounds scale amplification while preserving normalized z-space."
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
