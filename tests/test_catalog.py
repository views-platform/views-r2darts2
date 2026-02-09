import pytest
from unittest.mock import patch, MagicMock
from views_r2darts2.model.catalog import ModelCatalog
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tsmixer_model import TSMixerModel
from darts.models.forecasting.nlinear import NLinearModel
from darts.models.forecasting.tide_model import TiDEModel
from darts.models.forecasting.dlinear import DLinearModel


@pytest.fixture
def basic_config():
    """Fixture providing a basic configuration dictionary."""
    return {
        "steps": [1, 2, 3],
        "input_chunk_length": 48,
        "output_chunk_shift": 0,
        "name": "test_model",
        "random_state": 42,
        "batch_size": 64,
        "n_epochs": 2,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "dropout": 0.1,
        "activation": "ReLU",
        "gradient_clip_val": 0.8,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.001,
        "loss_function": "WeightedPenaltyHuberLoss",
        "zero_threshold": 0.01,
        "delta": 0.5,
        "non_zero_weight": 5.0,
        "false_negative_weight": 15.0,
        "false_positive_weight": 10.0,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 3,
        "lr_scheduler_min_lr": 1e-6,
        "use_reversible_instance_norm": False,
        "use_static_covariates": True,
        # Model-specific params with sensible defaults
        "generic_architecture": True,
        "num_stacks": 4,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_width": 128,
        "layer_widths": 128,
        "force_reset": True,
        # Loss params
        "p": 1.5,
        "eps": 1e-8,
        "tau": 0.75,
        # TFT params
        "feed_forward": "GatedResidualNetwork",
        "add_relative_index": True,
        "full_attention": False,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "hidden_size": 64,
        "skip_interpolation": False,
        "hidden_continuous_size": 16,
        # TCN params
        "kernel_size": 3,
        "num_filters": 64,
        "dilation_base": 2,
        # RNN params
        "rnn_type": "LSTM",
        "hidden_dim": 5,
        "n_rnn_layers": 2,
        # Transformer params
        "d_model": 128,
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 512,
        "norm_type": "LayerNorm",
        "detect_anomaly": False,
        # TSMixer params
        "ff_size": 64,
        "normalize_before": False,
        # NLinear/DLinear params
        "shared_weights": False,
        "const_init": True,
        "normalize": False,
        # NHiTS params
        "pooling_kernel_sizes": None,
        "n_freq_downsample": None,
        "max_pool_1d": True,
        # TiDE params
        "decoder_output_dim": 16,
        "temporal_width_past": 4,
        "temporal_width_future": 4,
        "temporal_hidden_size_past": None,
        "temporal_hidden_size_future": None,
        "temporal_decoder_hidden": 32,
        "use_layer_norm": False,
    }


@pytest.fixture
def full_config():
    """Fixture providing a full configuration with all parameters."""
    return {
        "steps": [1, 2, 3, 6, 12],
        "input_chunk_length": 48,
        "output_chunk_shift": 0,
        "name": "full_test_model",
        "random_state": 42,
        "batch_size": 128,
        "n_epochs": 10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "activation": "ReLU",
        "gradient_clip_val": 0.8,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.001,
        "loss_function": "WeightedPenaltyHuberLoss",
        "zero_threshold": 0.05,
        "delta": 0.5,
        "non_zero_weight": 5.0,
        "false_negative_weight": 15.0,
        "false_positive_weight": 10.0,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 3,
        "lr_scheduler_min_lr": 1e-6,
        "use_reversible_instance_norm": False,
        "use_static_covariates": True,
        # Model-specific params
        "generic_architecture": True,
        "num_stacks": 4,
        "num_blocks": 2,
        "num_layers": 2,
        "layer_width": 128,
        "layer_widths": 128,
        "force_reset": True,
        # Loss params
        "p": 1.5,
        "eps": 1e-8,
        "tau": 0.75,
        # TFT params
        "feed_forward": "GatedResidualNetwork",
        "add_relative_index": True,
        "full_attention": False,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "hidden_size": 64,
        "skip_interpolation": False,
        "hidden_continuous_size": 16,
        # TCN params
        "kernel_size": 3,
        "num_filters": 64,
        "dilation_base": 2,
        # RNN params
        "rnn_type": "LSTM",
        "hidden_dim": 5,
        "n_rnn_layers": 2,
        # Transformer params
        "d_model": 128,
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 512,
        "norm_type": "LayerNorm",
        "detect_anomaly": False,
        # TSMixer params
        "ff_size": 64,
        "normalize_before": False,
        # NLinear/DLinear params
        "shared_weights": False,
        "const_init": True,
        "normalize": False,
        # NHiTS params
        "pooling_kernel_sizes": None,
        "n_freq_downsample": None,
        "max_pool_1d": True,
        # TiDE params
        "decoder_output_dim": 16,
        "temporal_width_past": 4,
        "temporal_width_future": 4,
        "temporal_hidden_size_past": None,
        "temporal_hidden_size_future": None,
        "temporal_decoder_hidden": 32,
        "use_layer_norm": False,
    }


@pytest.fixture
def mock_device():
    """Mock the device detection."""
    with patch("views_r2darts2.model.catalog.DartsForecaster.get_device") as mock:
        mock.return_value = "cpu"
        yield mock


@pytest.fixture
def mock_wandb():
    """Mock WandB logger to prevent actual logging."""
    with patch("views_r2darts2.model.catalog.WandbLogger") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture(autouse=True)
def mock_all_external_deps(mock_device, mock_wandb):
    """Automatically mock all external dependencies for all tests."""
    pass


class TestModelCatalogInitialization:
    """Test suite for ModelCatalog initialization."""

    def test_init_with_basic_config(self, basic_config):
        """Test initialization with minimal configuration."""
        catalog = ModelCatalog(basic_config)
        
        assert catalog.config == basic_config
        assert catalog.device == "cpu"
        assert catalog.loss_name == "WeightedPenaltyHuberLoss"
        assert len(catalog.models) == 10

    def test_init_with_valid_loss_functions(self, basic_config):
        """Test initialization with valid loss function names."""
        valid_losses = ["WeightedPenaltyHuberLoss", "WeightedHuberLoss"]
        
        for loss_name in valid_losses:
            config = {**basic_config, "loss_function": loss_name}
            catalog = ModelCatalog(config)
            assert catalog.loss_name == loss_name

    def test_init_with_invalid_loss_raises_error(self, basic_config):
        """Test that invalid loss function raises ValueError."""
        config = {**basic_config, "loss_function": "InvalidLoss"}
        
        with pytest.raises(ValueError, match="Unknown loss function"):
            ModelCatalog(config)

    def test_loss_args_from_config(self, full_config):
        """Test that loss arguments are correctly extracted from config."""
        catalog = ModelCatalog(full_config)
        
        assert catalog.loss_args["zero_threshold"] == 0.05
        assert catalog.loss_args["delta"] == 0.5
        assert catalog.loss_args["non_zero_weight"] == 5.0
        assert catalog.loss_args["false_negative_weight"] == 15.0
        assert catalog.loss_args["false_positive_weight"] == 10.0

    def test_lr_scheduler_args(self, full_config):
        """Test that learning rate scheduler arguments are set correctly."""
        catalog = ModelCatalog(full_config)
        
        assert catalog.lr_scheduler_args["mode"] == "min"
        assert catalog.lr_scheduler_args["factor"] == 0.1
        assert catalog.lr_scheduler_args["patience"] == 3
        assert catalog.lr_scheduler_args["min_lr"] == 1e-6
        assert catalog.lr_scheduler_args["monitor"] == "train_loss"


class TestModelCatalogMethods:
    """Test suite for ModelCatalog public methods."""

    def test_list_models(self, basic_config):
        """Test listing all available models."""
        catalog = ModelCatalog(basic_config)
        models = catalog.list_models()
        
        assert len(models) == 10
        assert "NBEATSModel" in models
        assert "NHiTSModel" in models
        assert "TFTModel" in models
        assert "TCNModel" in models
        assert "BlockRNNModel" in models
        assert "TransformerModel" in models
        assert "TSMixerModel" in models
        assert "NLinearModel" in models
        assert "TiDEModel" in models
        assert "DLinearModel" in models

    @patch("torch.serialization.add_safe_globals")
    def test_get_model_valid(self, mock_safe_globals, basic_config):
        """Test getting a valid model."""
        catalog = ModelCatalog(basic_config)
        model = catalog.get_model("NBEATSModel")
        
        assert isinstance(model, NBEATSModel)
        mock_safe_globals.assert_called()

    def test_get_model_invalid(self, basic_config):
        """Test getting an invalid model returns None."""
        catalog = ModelCatalog(basic_config)
        
        # The models dict uses get() which returns None for invalid keys
        # But get_model() tries to call it, which will raise TypeError
        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            catalog.get_model("InvalidModel")

    @patch("torch.serialization.add_safe_globals")
    def test_get_model_case_sensitive(self, mock_safe_globals, basic_config):
        """Test that model names are case-sensitive."""
        catalog = ModelCatalog(basic_config)
        
        # Correct case should work
        assert catalog.get_model("NBEATSModel") is not None
        
        # Wrong case should raise TypeError (because None() is called)
        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            catalog.get_model("nbeatsmodel")


class TestNBEATSModel:
    """Test suite for N-BEATS model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_nbeats_default_params(self, mock_safe_globals, basic_config):
        """Test N-BEATS model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_nbeats()
        
        assert isinstance(model, NBEATSModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_nbeats_custom_params(self, mock_safe_globals, basic_config):
        """Test N-BEATS model with custom parameters."""
        config = {
            **basic_config,
            "input_chunk_length": 24,
            "num_stacks": 5,
            "num_blocks": 3,
            "layer_width": 256,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_nbeats()
        
        assert model.input_chunk_length == 24

    @patch("torch.serialization.add_safe_globals")
    def test_nbeats_uses_config_loss(self, mock_safe_globals, basic_config):
        """Test that N-BEATS model is configured with loss function."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_nbeats()
        
        # Verify the model was created successfully
        assert isinstance(model, NBEATSModel)
        # Verify loss function was passed during initialization
        assert catalog.loss_fn is not None


class TestTFTModel:
    """Test suite for TFT model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_tft_default_params(self, mock_safe_globals, basic_config):
        """Test TFT model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_tft_model()
        
        assert isinstance(model, TFTModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_tft_custom_params(self, mock_safe_globals, basic_config):
        """Test TFT model with custom parameters."""
        config = {
            **basic_config,
            "input_chunk_length": 36,
            "hidden_size": 512,
            "num_attention_heads": 8,
            "lstm_layers": 2,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_tft_model()
        
        assert model.input_chunk_length == 36


class TestTCNModel:
    """Test suite for TCN model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_tcn_default_params(self, mock_safe_globals, basic_config):
        """Test TCN model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_tcn_model()
        
        assert isinstance(model, TCNModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_tcn_custom_kernel_size(self, mock_safe_globals, basic_config):
        """Test TCN model with custom kernel size."""
        config = {
            **basic_config,
            "kernel_size": 5,
            "num_filters": 128,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_tcn_model()
        
        assert isinstance(model, TCNModel)


class TestBlockRNNModel:
    """Test suite for BlockRNN model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_rnn_default_params(self, mock_safe_globals, basic_config):
        """Test RNN model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_rnn_model()
        
        assert isinstance(model, BlockRNNModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_rnn_custom_type(self, mock_safe_globals, basic_config):
        """Test RNN model with custom RNN type."""
        config = {
            **basic_config,
            "rnn_type": "GRU",
            "hidden_dim": 128,
            "n_rnn_layers": 3,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_rnn_model()
        
        assert isinstance(model, BlockRNNModel)


class TestTransformerModel:
    """Test suite for Transformer model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_transformer_default_params(self, mock_safe_globals, basic_config):
        """Test Transformer model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_transformer_model()
        
        assert isinstance(model, TransformerModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_transformer_custom_layers(self, mock_safe_globals, basic_config):
        """Test Transformer with custom encoder/decoder layers."""
        config = {
            **basic_config,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_model": 128,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_transformer_model()
        
        assert isinstance(model, TransformerModel)


class TestTSMixerModel:
    """Test suite for TSMixer model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_tsmixer_default_params(self, mock_safe_globals, basic_config):
        """Test TSMixer model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_tsmixer_model()
        
        assert isinstance(model, TSMixerModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_tsmixer_custom_blocks(self, mock_safe_globals, basic_config):
        """Test TSMixer with custom number of blocks."""
        config = {
            **basic_config,
            "num_blocks": 4,
            "hidden_size": 128,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_tsmixer_model()
        
        assert isinstance(model, TSMixerModel)


class TestNLinearModel:
    """Test suite for NLinear model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_nlinear_default_params(self, mock_safe_globals, basic_config):
        """Test NLinear model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_nlinear_model()
        
        assert isinstance(model, NLinearModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_nlinear_shared_weights(self, mock_safe_globals, basic_config):
        """Test NLinear with shared weights enabled."""
        config = {
            **basic_config,
            "shared_weights": True,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_nlinear_model()
        
        assert isinstance(model, NLinearModel)


class TestDLinearModel:
    """Test suite for DLinear model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_dlinear_default_params(self, mock_safe_globals, basic_config):
        """Test DLinear model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_dlinear_model()
        
        assert isinstance(model, DLinearModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_dlinear_custom_kernel(self, mock_safe_globals, basic_config):
        """Test DLinear with custom kernel size."""
        config = {
            **basic_config,
            "kernel_size": 50,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_dlinear_model()
        
        assert isinstance(model, DLinearModel)


class TestTiDEModel:
    """Test suite for TiDE model creation."""

    @patch("torch.serialization.add_safe_globals")
    def test_tide_default_params(self, mock_safe_globals, basic_config):
        """Test TiDE model creation with default parameters."""
        catalog = ModelCatalog(basic_config)
        model = catalog._get_tide_model()
        
        assert isinstance(model, TiDEModel)
        assert model.output_chunk_length == len(basic_config["steps"])

    @patch("torch.serialization.add_safe_globals")
    def test_tide_custom_params(self, mock_safe_globals, basic_config):
        """Test TiDE model with custom parameters."""
        config = {
            **basic_config,
            "input_chunk_length": 48,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "hidden_size": 256,
        }
        catalog = ModelCatalog(config)
        model = catalog._get_tide_model()
        
        assert isinstance(model, TiDEModel)


class TestConfigurationHandling:
    """Test suite for configuration parameter handling."""

    @patch("torch.serialization.add_safe_globals")
    def test_output_chunk_length_from_steps(self, mock_safe_globals, basic_config):
        """Test that output_chunk_length is correctly derived from steps."""
        config = {**basic_config, "steps": [1, 2, 3, 6, 12, 24]}
        catalog = ModelCatalog(config)
        model = catalog._get_nbeats()
        
        assert model.output_chunk_length == 6

    def test_default_values_applied(self, basic_config):
        """Test that configured values are correctly applied."""
        catalog = ModelCatalog(basic_config)
        
        # Check loss function parameters from basic_config
        assert catalog.loss_args["zero_threshold"] == 0.01
        assert catalog.loss_args["delta"] == 0.5
        assert catalog.loss_args["non_zero_weight"] == 5.0

    def test_custom_values_override_defaults(self, basic_config):
        """Test that custom values are correctly applied."""
        config = {
            **basic_config,
            "zero_threshold": 0.5,
            "delta": 1.0,
            "non_zero_weight": 10.0,
        }
        catalog = ModelCatalog(config)
        
        assert catalog.loss_args["zero_threshold"] == 0.5
        assert catalog.loss_args["delta"] == 1.0
        assert catalog.loss_args["non_zero_weight"] == 10.0