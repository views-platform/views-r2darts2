# file: tests/test_tweedie_loss.py

import torch
import pytest
from darts import TimeSeries
from darts.models import (
    NBEATSModel,
    TCNModel,
    BlockRNNModel,
    TransformerModel,
    NLinearModel,
    TiDEModel,
)
from views_r2darts2.utils.loss.loss_catalog import LossCatalog
from views_r2darts2.utils.loss.tweedie_loss import TweedieLoss

# --- Unit Tests for the new TweedieLoss (with softplus link) ---


def test_tweedie_loss_init():
    """Tests the initialization of the corrected TweedieLoss."""
    # Test valid initialization
    loss_fn = TweedieLoss(
        p=1.5,
        non_zero_weight=5.0,
        zero_threshold=0.01,
        false_positive_weight=1.0,
        false_negative_weight=1.0,
        eps=1e-7,
    )
    assert loss_fn.p == 1.5
    assert loss_fn.eps == 1e-7

    # Test invalid power parameter
    with pytest.raises(ValueError, match="Tweedie power parameter p must be in"):
        TweedieLoss(
            p=0.9,
            non_zero_weight=5.0,
            zero_threshold=0.01,
            false_positive_weight=1.0,
            false_negative_weight=1.0,
            eps=1e-8,
        )
    with pytest.raises(ValueError, match="Tweedie power parameter p must be in"):
        TweedieLoss(
            p=2.0,
            non_zero_weight=5.0,
            zero_threshold=0.01,
            false_positive_weight=1.0,
            false_negative_weight=1.0,
            eps=1e-8,
        )


@pytest.mark.parametrize(
    "preds, targets, p, expected_loss",
    [
        # Case 1: Zero target, very negative preds. mu -> eps.
        # mu = softplus(-20.0) + 1e-8 approx 2.06e-9 + 1e-8 approx 1.206e-8
        # loss = 2 * sqrt(1.206e-8) = 0.0002196
        (torch.tensor([-20.0]), torch.tensor([0.0]), 1.5, 0.00020),
        # Case 2: Non-zero target, perfect prediction (mu=target)
        # y=2.0, preds=log(exp(2.0)-1)=1.9560475, mu=2.0
        # loss = 4*sqrt(2) = 5.65685425
        (torch.tensor([1.9560475]), torch.tensor([2.0]), 1.5, 5.65820),
        # Case 3: Non-zero target, under-prediction
        # y=2.0, preds=0.0, mu=softplus(0.0)+1e-8 = 0.69314718 + 1e-8
        # loss = 2*sqrt(0.69314719) + 4/sqrt(0.69314719) = 1.664741 + 4.805548 = 6.47029
        (torch.tensor([0.0]), torch.tensor([2.0]), 1.5, 6.46960),
        # Case 4: Non-zero target, over-prediction
        # y=2.0, preds=3.0, mu=softplus(3.0)+1e-8 = 3.0485076 + 1e-8
        # loss = 2*sqrt(3.04850761) + 4/sqrt(3.04850761) = 3.492003 + 2.291000 = 5.783003
        (torch.tensor([3.0]), torch.tensor([2.0]), 1.5, 5.78300),
        # Case 5: Different p value (p=1.2), perfect prediction (mu=target)
        # y=2.0, preds=log(exp(2.0)-1)=1.9560475, mu=2.0, p=1.2
        # loss = 1.25 * (2.00000001**0.8) + 5 * (2 * 2.00000001**-0.2) = 2.176376 + 8.70550 = 10.881876
        (torch.tensor([1.9560475]), torch.tensor([2.0]), 1.2, 10.88350),
    ],
)
def test_tweedie_loss_golden_values(preds, targets, p, expected_loss):
    """Tests the forward pass of the new TweedieLoss against manually calculated golden values."""
    loss_fn = TweedieLoss(
        p=p,
        non_zero_weight=1.0,
        zero_threshold=0.01,
        false_positive_weight=1.0,
        false_negative_weight=1.0,
        eps=1e-8,
    )
    loss = loss_fn(preds, targets)
    assert torch.isclose(
        loss, torch.tensor(expected_loss), atol=1e-3
    )  # Increased precision for atol


def test_tweedie_loss_invariant_minimization():
    """Tests that for a given target, the loss is minimized when mu equals the target (using softplus)."""
    loss_fn = TweedieLoss(
        p=1.5,
        non_zero_weight=1.0,
        zero_threshold=0.01,
        false_positive_weight=1.0,
        false_negative_weight=1.0,
        eps=1e-8,
    )
    target = torch.tensor([2.0])

    # For softplus link, pred_val_at_target should make softplus(pred_val_at_target) == target
    # softplus(x) = target => log(1 + exp(x)) = target => 1 + exp(x) = exp(target) => exp(x) = exp(target) - 1
    # x = log(exp(target) - 1)
    pred_val_at_target = torch.log(torch.exp(target) - 1)

    loss_at_min = loss_fn(pred_val_at_target, target)
    loss_below_min = loss_fn(pred_val_at_target - 0.5, target)
    loss_above_min = loss_fn(pred_val_at_target + 0.5, target)

    assert loss_at_min < loss_below_min
    assert loss_at_min < loss_above_min


def test_tweedie_loss_gradient_check():
    """Performs a gradient check for the new TweedieLoss."""
    from torch.autograd import gradcheck

    loss_fn = TweedieLoss(
        p=1.5,
        non_zero_weight=1.0,
        zero_threshold=0.01,
        false_positive_weight=1.0,
        false_negative_weight=1.0,
        eps=1e-6,
    )  # Use larger eps for stability in gradcheck

    # Use double precision for gradcheck
    preds = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    # Targets must be non-negative for Tweedie
    targets = torch.rand(2, 2, dtype=torch.double)

    assert gradcheck(loss_fn, (preds, targets), eps=1e-6, atol=1e-4)


# --- Integration Tests with Darts Models ---
ts_train = TimeSeries.from_values(
    torch.arange(0, 100, dtype=torch.float32).unsqueeze(1)
)
ts_val = TimeSeries.from_values(
    torch.arange(100, 150, dtype=torch.float32).unsqueeze(1)
)

MODEL_CONFIGS = {
    "N-BEATS": (NBEATSModel, {"input_chunk_length": 12, "output_chunk_length": 6}),
    "TCN": (
        TCNModel,
        {
            "input_chunk_length": 24,
            "output_chunk_length": 6,
            "kernel_size": 3,
            "num_filters": 4,
        },
    ),
    "BlockRNN (LSTM)": (
        BlockRNNModel,
        {
            "model": "LSTM",
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "hidden_dim": 16,
        },
    ),
    "Transformer": (
        TransformerModel,
        {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
        },
    ),
    "N-Linear": (NLinearModel, {"input_chunk_length": 12, "output_chunk_length": 6}),
    "TiDE": (
        TiDEModel,
        {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "decoder_output_dim": 8,
            "hidden_size": 16,
        },
    ),
}


@pytest.mark.parametrize("seed", [42, 1337])
@pytest.mark.parametrize("model_name, model_tuple", MODEL_CONFIGS.items())
def test_tweedie_loss_with_darts_models(model_name, model_tuple, seed):
    """
    Tests that the new TweedieLoss can be successfully used for training with various Darts models.
    """
    model_cls, model_kwargs = model_tuple
    torch.manual_seed(seed)

    # Using LossCatalog to ensure it correctly passes the parameters
    config = {
        "loss_function": "TweedieLoss",
        "p": 1.5,
        "non_zero_weight": 5.0,
        "zero_threshold": 0.01,
        "false_positive_weight": 1.0,
        "false_negative_weight": 1.0,
        "eps": 1e-8
    }
    loss_fn = LossCatalog(config).get_loss()

    model = model_cls(
        **model_kwargs,
        n_epochs=2,
        random_state=seed,
        loss_fn=loss_fn,
        pl_trainer_kwargs={"accelerator": "cpu", "devices": 1},
        force_reset=True,
    )

    try:
        model.fit(series=ts_train, val_series=ts_val, verbose=False)
    except Exception as e:
        pytest.fail(
            f"Integration test for {model_name} with new TweedieLoss and seed {seed} failed with error: {e}"
        )

    try:
        preds = model.predict(n=5)
        assert len(preds) == 5
    except Exception as e:
        pytest.fail(
            f"Prediction for {model_name} with new TweedieLoss and seed {seed} failed after training: {e}"
        )
