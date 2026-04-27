import torch
import torch.nn as nn
import logging
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


# --- 3. RINorm log(1+σ) Compression Patch ---
#
# Bounds scale amplification in RevIN's inverse pass.
#
# Standard RevIN denormalizes as ŷ = ẑ·σ + μ. When σ is large (high-variance
# entities like Ukraine with σ≈4 in asinh space) and the inverse transform is
# convex (sinh), small normalized-space outputs get exponentially amplified:
#
#   Raw σ:    ŷ = 2.5 × 4.0 + 5 = 15.0  →  sinh(15) ≈ 1,634,508 deaths
#   log1p:    ŷ = 2.5 × 1.61 + 5 = 9.025 →  sinh(9)  ≈ 4,143 deaths
#
# log(1+σ) compresses σ sublinearly, bounding amplification while preserving
# the normalization that makes the learning task tractable:
#
#   σ_raw=0.5  →  σ_compressed=0.41    (peaceful: minimal change)
#   σ_raw=1.0  →  σ_compressed=0.69    (low-conflict: mild compression)
#   σ_raw=4.0  →  σ_compressed=1.61    (Ukraine-level: 2.5× reduction)
#   σ_raw=10   →  σ_compressed=2.40    (extreme: 4.2× reduction)
#
# Why not σ≡1 (mean-only)? Without ANY σ normalization, the model sees
# multi-scale z-targets (Ukraine ±4, peaceful ±0.1). Many HP configurations
# produce outlier predictions (ẑ=6+) that explode through sinh. log(1+σ)
# keeps z-targets normalized across entities for stable training.
#
# Note on the canonical shape problem: RevIN strips per-series identity,
# causing all entities to produce identical temporal patterns ("over-
# stationarization", Liu et al. NeurIPS 2022). This CANNOT be fixed at
# the normalization layer because:
#
#   - Symmetric conditioning (undo in inverse): the model's task is
#     equivalent regardless of conditioning — it's just a coordinate
#     transform that cancels out.
#   - Asymmetric conditioning (don't undo): shared backbone weights
#     can't compensate for per-series modulation → muted magnitudes.
#
# The canonical shape problem must be addressed through richer covariates,
# static covariates (country-level features), or architectural changes
# that inject entity identity into the backbone itself.


def apply_rinorm_compression_patch():
    """
    Patches Darts RINorm with log(1+σ) compressed standard deviation.
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
        self.stdev = torch.log1p(raw_stdev)
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    _compressed_forward._compressed = True
    RINorm.forward = _compressed_forward
    logger.info(
        "🐨 Patched RINorm: log(1+σ) compression. "
        "Bounds scale amplification while preserving normalized z-space."
    )


# --- Initialize All Patches ---

def apply_all_patches():
    apply_torch_load_patch()
    apply_rinorm_compression_patch()
    # apply_nbeats_patch()
