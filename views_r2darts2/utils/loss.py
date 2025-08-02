import torch
import torch.nn.functional as F
import inspect
import logging

logger = logging.getLogger(__name__)


class LossSelector:
    @staticmethod
    def get_loss_function(loss_name, **kwargs):
        # Map loss names to their classes
        loss_classes = {
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
        }

        if loss_name not in loss_classes:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Get the class constructor parameters
        cls = loss_classes[loss_name]
        params = inspect.signature(cls).parameters

        # Filter kwargs to only include valid parameters for this loss
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}

        return cls(**valid_kwargs)


class WeightedHuberLoss(torch.nn.Module):
    def __init__(self, zero_threshold=0.01, delta=0.5, non_zero_weight=5.0):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight

    def forward(self, preds, targets):
        # Create sample weights
        weights = torch.where(
            torch.abs(targets) > self.threshold, self.non_zero_weight, 1.0
        )

        # Calculate base Huber loss
        errors = targets - preds
        huber_loss = torch.where(
            torch.abs(errors) <= self.delta,
            0.5 * errors**2,
            self.delta * (torch.abs(errors) - 0.5 * self.delta),
        )

        # Apply sample weighting
        weighted_loss = weights * huber_loss
        return torch.mean(weighted_loss)


class TimeAwareWeightedHuberLoss(torch.nn.Module):
    def __init__(self, zero_weight, non_zero_weight, decay_factor, delta):
        super().__init__()
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight
        self.decay_factor = decay_factor
        self.delta = delta
        self.huber = torch.nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, preds, targets):
        # Temporal decay weights
        seq_len = targets.size(1)
        time_weights = torch.tensor(
            [self.decay_factor ** (seq_len - i) for i in range(seq_len)],
            device=targets.device,
        )

        # Event weights
        event_weights = torch.where(
            targets.abs() > 1e-4, self.non_zero_weight, self.zero_weight
        )

        # Combined weighting
        weights = time_weights * event_weights
        losses = self.huber(preds, targets)
        return (weights * losses).mean()


class SpikeFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, spike_threshold=3.0445):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spike_threshold = spike_threshold

    def forward(self, preds, targets):
        errors = (preds - targets).abs()
        is_spike = targets > self.spike_threshold

        # Base loss
        loss = errors**2

        # Focal component
        focal_weights = torch.where(
            is_spike,
            self.alpha * (1 - torch.exp(-errors)) ** self.gamma,
            (1 - self.alpha) * (torch.exp(-errors)) ** self.gamma,
        )

        return torch.mean(focal_weights * loss)


class WeightedPenaltyHuberLoss(torch.nn.Module):
    def __init__(
        self,
        zero_threshold=0.01,
        delta=0.5,
        non_zero_weight=5.0,
        false_positive_weight=10.0,
        false_negative_weight=15.0,
    ):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        logger.info(
            "\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}\n{:<25} {:<10}".format(
            "zero_threshold", zero_threshold,
            "delta", delta,
            "non_zero_weight", non_zero_weight,
            "false_positive_weight", false_positive_weight,
            "false_negative_weight", false_negative_weight
            )
        )

    def forward(self, preds, targets):
        # Identify non-zero targets and predictions (detach masks to prevent gradient flow)
        is_target_nonzero = torch.abs(targets) > self.threshold
        is_pred_nonzero = (torch.abs(preds) > self.threshold).detach()

        # Base weights: prioritize non-zero targets
        base_weights = torch.where(is_target_nonzero, self.non_zero_weight, 1.0)

        # Identify error types
        false_positive_mask = ~is_target_nonzero & is_pred_nonzero
        false_negative_mask = is_target_nonzero & ~is_pred_nonzero

        # Apply penalties to error types while preserving base weights
        weights = torch.where(
            false_positive_mask,
            self.false_positive_weight,
            torch.where(false_negative_mask, self.false_negative_weight, base_weights),
        )

        # Calculate Huber loss
        errors = targets - preds
        huber_loss = torch.where(
            torch.abs(errors) <= self.delta,
            0.5 * errors**2,
            self.delta * (torch.abs(errors) - 0.5 * self.delta),
        )

        # Apply computed weights
        weighted_loss = weights * huber_loss
        return torch.mean(weighted_loss)
