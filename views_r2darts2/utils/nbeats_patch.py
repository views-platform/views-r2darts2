import torch
import torch.nn as nn
from darts.logging import raise_if_not, raise_log, get_logger
from darts.models.forecasting.nbeats import _GType, _TrendGenerator, _SeasonalityGenerator, ACTIVATIONS, _Block
from darts.utils.torch import MonteCarloDropout
from darts.models import NBEATSModel
import numpy as np
from darts import TimeSeries

logger = get_logger(__name__)

def _patched_block_init(self, num_layers: int, layer_width: int, nr_params: int, expansion_coefficient_dim: int, input_chunk_length: int, target_length: int, g_type: _GType, batch_norm: bool, dropout: float, activation: str):
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
        self.bn_stack.extend([nn.BatchNorm1d(num_features=layer_width) for _ in range(num_layers)])
    if self.dropout_val > 0:
        self.dropout_stack.extend([MonteCarloDropout(p=self.dropout_val) for _ in range(num_layers)])
    if g_type == _GType.SEASONALITY:
        self.backcast_linear_layer = nn.Linear(layer_width, 2 * int(input_chunk_length / 2 - 1) + 1)
        self.forecast_linear_layer = nn.Linear(layer_width, nr_params * (2 * int(target_length / 2 - 1) + 1))
    else:
        self.backcast_linear_layer = nn.Linear(layer_width, expansion_coefficient_dim)
        self.forecast_linear_layer = nn.Linear(layer_width, nr_params * expansion_coefficient_dim)
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
        raise_log(ValueError("g_type not supported"), logger)

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

def patch_nbeats_dropout_issue():

    try:

        _Block.__init__ = _patched_block_init

        _Block.forward = _patched_block_forward

        logger.info("Successfully patched Darts NBEATSModel.")

    except Exception as e:

        logger.error(f"An unexpected error occurred during N-BEATS patching: {e}")



# Apply the monkey-patch on import

patch_nbeats_dropout_issue()



if __name__ == '__main__':

    # --- Sanity Check Script ---



    # 2. Define model parameters

    input_chunk_length = 20

    output_chunk_length = 5

    # Use a fixed random_state to ensure deterministic initialization

    random_seed = 42



    # Create a dummy TimeSeries to trigger model creation

    dummy_data = np.random.randn(input_chunk_length + output_chunk_length)

    dummy_series = TimeSeries.from_values(dummy_data)





    # 3. Create a model with layer_width=64

    model64 = NBEATSModel(

        input_chunk_length=input_chunk_length,

        output_chunk_length=output_chunk_length,

        layer_widths=64,

        n_epochs=1, # not training, just instantiating

        random_state=random_seed

    )

    model64.fit(dummy_series, verbose=False)





    # 4. Create a model with layer_width=128

    model128 = NBEATSModel(

        input_chunk_length=input_chunk_length,

        output_chunk_length=output_chunk_length,

        layer_widths=128,

        n_epochs=1, # not training, just instantiating

        random_state=random_seed

    )

    model128.fit(dummy_series, verbose=False)



    # 5. Get the total number of parameters for each model

    params_total64 = sum(p.numel() for p in model64.model.parameters())

    params_total128 = sum(p.numel() for p in model128.model.parameters())



    print(f"Total parameters for model with layer_width=64: {params_total64}")

    print(f"Total parameters for model with layer_width=128: {params_total128}")



    if params_total64 == params_total128:

        print("\nCRITICAL ERROR: Models have the exact same number of parameters!")

        print("This confirms the `layer_widths` parameter is not being used correctly during model creation.")

    else:

        print("\nSUCCESS: Models have a different number of parameters, as expected.")

        print("This suggests the problem may lie in your training/evaluation script, not the model architecture itself.")


