import pytest
import torch
from views_r2darts2.utils.loss import (
    LossSelector,
    WeightedHuberLoss,
    TimeAwareWeightedHuberLoss,
    SpikeFocalLoss,
    WeightedPenaltyHuberLoss,
)


class TestLossSelector:
    def test_get_weighted_huber_loss(self):
        loss = LossSelector.get_loss_function(
            "WeightedHuberLoss", zero_threshold=0.02, delta=1.0, non_zero_weight=10.0
        )
        assert isinstance(loss, WeightedHuberLoss)
        assert loss.threshold == 0.02
        assert loss.delta == 1.0
        assert loss.non_zero_weight == 10.0

    def test_get_time_aware_weighted_huber_loss(self):
        loss = LossSelector.get_loss_function(
            "TimeAwareWeightedHuberLoss",
            zero_weight=0.5,
            non_zero_weight=2.0,
            decay_factor=0.9,
            delta=1.0,
        )
        assert isinstance(loss, TimeAwareWeightedHuberLoss)
        assert loss.zero_weight == 0.5
        assert loss.non_zero_weight == 2.0

    def test_get_spike_focal_loss(self):
        loss = LossSelector.get_loss_function(
            "SpikeFocalLoss", alpha=0.7, gamma=3.0, spike_threshold=5.0
        )
        assert isinstance(loss, SpikeFocalLoss)
        assert loss.alpha == 0.7
        assert loss.gamma == 3.0

    def test_get_weighted_penalty_huber_loss(self):
        loss = LossSelector.get_loss_function(
            "WeightedPenaltyHuberLoss",
            zero_threshold=0.05,
            delta=0.8,
            non_zero_weight=6.0,
            false_positive_weight=12.0,
            false_negative_weight=18.0,
        )
        assert isinstance(loss, WeightedPenaltyHuberLoss)
        assert loss.threshold == 0.05

    def test_unknown_loss_function(self):
        with pytest.raises(ValueError, match="Unknown loss function"):
            LossSelector.get_loss_function("NonExistentLoss")

    def test_filters_invalid_kwargs(self):
        loss = LossSelector.get_loss_function(
            "WeightedHuberLoss", delta=1.0, invalid_param=999
        )
        assert isinstance(loss, WeightedHuberLoss)
        assert loss.delta == 1.0


class TestWeightedHuberLoss:
    @pytest.fixture
    def loss_fn(self):
        return WeightedHuberLoss(zero_threshold=0.01, delta=0.5, non_zero_weight=5.0)

    def test_initialization(self):
        loss = WeightedHuberLoss(zero_threshold=0.02, delta=1.0, non_zero_weight=10.0)
        assert loss.threshold == 0.02
        assert loss.delta == 1.0
        assert loss.non_zero_weight == 10.0

    def test_forward_all_zeros(self, loss_fn):
        preds = torch.zeros(10)
        targets = torch.zeros(10)
        loss = loss_fn(preds, targets)
        assert loss.item() == 0.0

    def test_forward_non_zero_targets(self, loss_fn):
        preds = torch.zeros(10)
        targets = torch.ones(10) * 0.5
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_weighting_applied(self):
        loss_fn = WeightedHuberLoss(
            zero_threshold=0.01, delta=0.5, non_zero_weight=10.0
        )
        preds = torch.zeros(2)
        targets = torch.tensor([0.005, 0.5])  # Below and above threshold
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_quadratic_region(self, loss_fn):
        # Error less than delta should use quadratic form
        preds = torch.tensor([0.0])
        targets = torch.tensor([0.2])  # Error = 0.2 < delta=0.5
        loss = loss_fn(preds, targets)
        # Target 0.2 is above threshold (0.01), so weight is non_zero_weight (5.0)
        # Quadratic form: 0.5 * error^2 * weight = 0.5 * 0.2^2 * 5.0
        expected = 0.5 * 0.2**2 * 5.0
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_linear_region(self, loss_fn):
        # Error greater than delta should use linear form
        preds = torch.tensor([0.0])
        targets = torch.tensor([1.0])  # Error = 1.0 > delta=0.5
        loss = loss_fn(preds, targets)
        # Linear form: delta * (|error| - 0.5 * delta) * non_zero_weight
        expected = 0.5 * (1.0 - 0.5 * 0.5) * 5.0
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_output_shape(self, loss_fn):
        preds = torch.randn(32, 10)
        targets = torch.randn(32, 10)
        loss = loss_fn(preds, targets)
        assert loss.shape == torch.Size([])


class TestTimeAwareWeightedHuberLoss:
    @pytest.fixture
    def loss_fn(self):
        return TimeAwareWeightedHuberLoss(
            zero_weight=1.0, non_zero_weight=5.0, decay_factor=0.95, delta=0.5
        )

    def test_initialization(self):
        loss = TimeAwareWeightedHuberLoss(
            zero_weight=0.5, non_zero_weight=2.0, decay_factor=0.9, delta=1.0
        )
        assert loss.zero_weight == 0.5
        assert loss.non_zero_weight == 2.0
        assert loss.decay_factor == 0.9
        assert loss.delta == 1.0

    def test_forward_shape(self, loss_fn):
        preds = torch.randn(16, 10, 1)
        targets = torch.randn(16, 10, 1)
        loss = loss_fn(preds, targets)
        assert loss.shape == torch.Size([])

    def test_temporal_decay(self):
        loss_fn = TimeAwareWeightedHuberLoss(
            zero_weight=1.0, non_zero_weight=1.0, decay_factor=0.5, delta=1.0
        )
        # Recent errors should matter more than older ones
        batch_size = 4
        seq_len = 5
        preds = torch.zeros(batch_size, seq_len)
        targets = torch.ones(batch_size, seq_len)

        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_event_weighting(self, loss_fn):
        batch_size = 4
        seq_len = 5
        preds = torch.zeros(batch_size, seq_len)
        # Mix of zero and non-zero targets
        targets = torch.tensor([[0.0, 0.1, 0.0, 0.5, 0.0]] * batch_size)
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_all_zeros(self, loss_fn):
        preds = torch.zeros(8, 10)
        targets = torch.zeros(8, 10)
        loss = loss_fn(preds, targets)
        assert loss.item() == 0.0


class TestSpikeFocalLoss:
    @pytest.fixture
    def loss_fn(self):
        return SpikeFocalLoss(alpha=0.8, gamma=2.0, spike_threshold=3.0445)

    def test_initialization(self):
        loss = SpikeFocalLoss(alpha=0.7, gamma=3.0, spike_threshold=5.0)
        assert loss.alpha == 0.7
        assert loss.gamma == 3.0
        assert loss.spike_threshold == 5.0

    def test_forward_all_zeros(self, loss_fn):
        preds = torch.zeros(10)
        targets = torch.zeros(10)
        loss = loss_fn(preds, targets)
        assert loss.item() == 0.0

    def test_forward_with_spikes(self, loss_fn):
        preds = torch.zeros(10)
        targets = torch.tensor([0.0, 0.0, 5.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 0.0])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_spike_detection(self, loss_fn):
        preds = torch.zeros(2)
        targets = torch.tensor([2.0, 5.0])  # Below and above threshold
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_focal_weighting(self):
        loss_fn = SpikeFocalLoss(alpha=0.9, gamma=2.0, spike_threshold=3.0)
        preds = torch.zeros(2)
        targets = torch.tensor([1.0, 5.0])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_output_shape(self, loss_fn):
        preds = torch.randn(32, 10)
        targets = torch.abs(torch.randn(32, 10))
        loss = loss_fn(preds, targets)
        assert loss.shape == torch.Size([])


class TestWeightedPenaltyHuberLoss:
    @pytest.fixture
    def loss_fn(self):
        return WeightedPenaltyHuberLoss(
            zero_threshold=0.01,
            delta=0.5,
            non_zero_weight=5.0,
            false_positive_weight=10.0,
            false_negative_weight=15.0,
        )

    def test_initialization(self):
        loss = WeightedPenaltyHuberLoss(
            zero_threshold=0.02,
            delta=1.0,
            non_zero_weight=6.0,
            false_positive_weight=12.0,
            false_negative_weight=18.0,
        )
        assert loss.threshold == 0.02
        assert loss.delta == 1.0
        assert loss.non_zero_weight == 6.0
        assert loss.false_positive_weight == 12.0
        assert loss.false_negative_weight == 18.0

    def test_forward_all_zeros(self, loss_fn):
        preds = torch.zeros(10)
        targets = torch.zeros(10)
        loss = loss_fn(preds, targets)
        assert loss.item() == 0.0

    def test_false_positive_penalty(self, loss_fn):
        # Predict non-zero when target is zero
        preds = torch.tensor([0.5, 0.5, 0.5])
        targets = torch.tensor([0.0, 0.0, 0.0])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_false_negative_penalty(self, loss_fn):
        # Predict zero when target is non-zero
        preds = torch.tensor([0.0, 0.0, 0.0])
        targets = torch.tensor([0.5, 0.5, 0.5])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_mixed_errors(self, loss_fn):
        # Mix of correct, false positive, and false negative
        preds = torch.tensor([0.0, 0.5, 0.0, 0.5])
        targets = torch.tensor([0.0, 0.0, 0.5, 0.5])
        loss = loss_fn(preds, targets)
        assert loss.item() > 0.0

    def test_penalty_weights_differ(self):
        loss_fn = WeightedPenaltyHuberLoss(
            zero_threshold=0.01,
            delta=0.5,
            non_zero_weight=5.0,
            false_positive_weight=10.0,
            false_negative_weight=20.0,  # Higher than FP
        )

        # False negative
        preds_fn = torch.tensor([0.0])
        targets_fn = torch.tensor([0.5])
        loss_fn_val = loss_fn(preds_fn, targets_fn)

        # False positive
        preds_fp = torch.tensor([0.5])
        targets_fp = torch.tensor([0.0])
        loss_fp_val = loss_fn(preds_fp, targets_fp)

        # False negative should have higher loss due to higher weight
        assert loss_fn_val.item() > loss_fp_val.item()

    def test_output_shape(self, loss_fn):
        preds = torch.randn(32, 10)
        targets = torch.randn(32, 10)
        loss = loss_fn(preds, targets)
        assert loss.shape == torch.Size([])

    def test_gradient_flow(self, loss_fn):
        preds = torch.randn(10, requires_grad=True)
        targets = torch.randn(10)
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None
        assert preds.grad.shape == preds.shape


# Integration tests
class TestLossIntegration:
    def test_all_losses_with_same_input(self):
        preds = torch.randn(16, 10)
        targets = torch.randn(16, 10)

        weighted_huber = WeightedHuberLoss()
        time_aware = TimeAwareWeightedHuberLoss(
            zero_weight=1.0, non_zero_weight=5.0, decay_factor=0.95, delta=0.5
        )
        spike_focal = SpikeFocalLoss()
        weighted_penalty = WeightedPenaltyHuberLoss()

        loss1 = weighted_huber(preds, targets)
        # TimeAware expects shape (batch, seq_len, ...)
        loss2 = time_aware(preds.unsqueeze(1), targets.unsqueeze(1))
        loss3 = spike_focal(preds, targets)
        loss4 = weighted_penalty(preds, targets)

        assert all(loss.item() >= 0 for loss in [loss1, loss2, loss3, loss4])

    def test_loss_decreases_with_better_predictions(self):
        targets = torch.randn(32)
        loss_fn = WeightedHuberLoss()

        # Bad predictions
        bad_preds = torch.randn(32) * 10
        bad_loss = loss_fn(bad_preds, targets)

        # Good predictions (closer to targets)
        good_preds = targets + torch.randn(32) * 0.1
        good_loss = loss_fn(good_preds, targets)

        assert good_loss.item() < bad_loss.item()