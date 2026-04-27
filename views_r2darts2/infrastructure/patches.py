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


# --- 3. Statistics-Conditioned RevIN Patch ---
#
# Two-part patch for RINorm:
#
#   A) log(1+σ) compression — bounds scale amplification in the inverse pass.
#      Without this, high-variance entities (Ukraine σ≈4 in asinh space) produce
#      explosive predictions through the convex sinh inverse:
#        Raw σ:  ŷ = 2.5 × 4.0 + 5 = 15.0  →  sinh(15) ≈ 1,634,508 deaths
#        log1p:  ŷ = 2.5 × 1.61 + 5 = 9.025 →  sinh(9)  ≈ 4,143 deaths
#
#   B) Statistics conditioning — re-injects (μ, σ_raw) as a learnable additive
#      signal into the normalized z-space. Standard RevIN strips per-series
#      identity, causing "over-stationarization" (Liu et al. NeurIPS 2022):
#      all entities produce an identical canonical temporal pattern, with only
#      level/scale differing after the mechanical ẑ·σ+μ inverse. The model
#      cannot learn entity-specific dynamics.
#
#      By projecting (μ, σ_raw) through a zero-initialized Linear layer and
#      adding the result to z, the model can learn to differentiate series
#      while still benefiting from normalized z-targets for stable training.
#
#      Zero-init ensures the model starts as standard RevIN (conditioning = 0)
#      and gradually learns to use the statistics via gradient flow.
#
#   For n_targets=1 (VIEWS fatalities), this adds 1× Linear(2, 1) = 3 parameters.
#   For n_targets=K, this adds K× Linear(2, 1) = 3K parameters.
#
#   Works with ALL Darts models (N-BEATS, Transformer, TFT, TiDE, TSMixer,
#   BlockRNN, TCN, NHiTS) because input/output dimensions are unchanged —
#   the conditioning is additive within existing target channels.
#
#   Per-target projections: each target i has its own Linear(2, 1) mapping
#   (μ_i, σ_i) → scalar conditioning. This prevents cross-target leakage
#   when targets have heterogeneous scales (e.g., fatalities + covariates).
#   AdamW's per-parameter adaptive learning rate handles (μ, σ) scale
#   differences, so no manual normalization of inputs is needed.


def apply_conditioned_rinorm_patch():
    """
    Patches Darts RINorm with log(1+σ) compression and per-target statistics conditioning.
    Patches both __init__ (to add stat_projs) and forward (to apply conditioning).
    The inverse method is unchanged — output dimensions are identical.
    """
    from darts.models.components.layer_norm_variants import RINorm

    if getattr(RINorm.forward, '_stat_conditioned', False):
        return  # Already patched

    _orig_init = RINorm.__init__

    def _conditioned_init(self, input_dim: int, eps=1e-5, affine=True):
        _orig_init(self, input_dim, eps, affine)
        # Per-target learnable projection: (μ_i, σ_i) → scalar conditioning offset.
        # Independent per target — no cross-target leakage.
        # Zero-initialized so the model starts as standard RevIN.
        self.stat_projs = nn.ModuleList([
            nn.Linear(2, 1, bias=True) for _ in range(input_dim)
        ])
        for proj in self.stat_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _conditioned_forward(self, x: torch.Tensor):
        # x: (B, T, n_targets)
        calc_dims = tuple(range(1, x.ndim - 1))

        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        raw_stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        # Part A: log(1+σ) compression for the normalization denominator.
        self.stdev = torch.log1p(raw_stdev)
        z = (x - self.mean) / self.stdev

        if self.affine:
            z = z * self.affine_weight
            z = z + self.affine_bias

        # Part B: per-target statistics conditioning.
        # Each target independently maps its own (μ_i, σ_i) to an additive
        # offset so the model can distinguish series identity in z-space.
        mu = self.mean.squeeze(1)        # (B, n_targets)
        sigma = raw_stdev.squeeze(1)     # (B, n_targets)
        conditioning = torch.cat([
            proj(torch.stack([mu[:, i], sigma[:, i]], dim=-1))  # (B, 1)
            for i, proj in enumerate(self.stat_projs)
        ], dim=-1)                       # (B, n_targets)
        z = z + conditioning.unsqueeze(1)  # broadcast → (B, T, n_targets)

        return z

    _conditioned_forward._stat_conditioned = True
    RINorm.__init__ = _conditioned_init
    RINorm.forward = _conditioned_forward
    logger.info(
        "🐨 Patched RINorm: log(1+σ) compression + per-target statistics conditioning. "
        "Series identity (μ, σ) re-injected via independent zero-init projections."
    )


# --- Initialize All Patches ---

def apply_all_patches():
    apply_torch_load_patch()
    apply_conditioned_rinorm_patch()
    # apply_nbeats_patch()
