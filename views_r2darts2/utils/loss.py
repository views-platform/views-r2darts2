import torch
import torch.nn.functional as F
import inspect

class LossSelector:
    @staticmethod
    def get_loss_function(loss_name, **kwargs):
        # Map loss names to their classes
        loss_classes = {
            "WeightedSmoothL1Loss": WeightedSmoothL1Loss,
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "AsymmetricSpikeLoss": AsymmetricSpikeLoss,
            "LogSpaceLoss": LogSpaceLoss,
            "ZeroInflatedTweedieLoss": ZeroInflatedTweedieLoss,
            "HybridSpikeLoss": HybridSpikeLoss,
        }
        
        if loss_name not in loss_classes:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        # Get the class constructor parameters
        cls = loss_classes[loss_name]
        params = inspect.signature(cls).parameters
        
        # Filter kwargs to only include valid parameters for this loss
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}
        
        return cls(**valid_kwargs)
class WeightedSmoothL1Loss(torch.nn.Module):
    def __init__(self, beta=0.2, zero_weight=0.3, non_zero_weight=1.5):
        super().__init__()
        self.beta = beta
        self.weights = {0: zero_weight, 1: non_zero_weight}
        
    def forward(self, input, target):
        loss = torch.nn.functional.smooth_l1_loss(
            input, target, reduction='none', beta=self.beta
        )
        weights = torch.where(target == 0, 
                            torch.full_like(target, self.weights[0]),
                            torch.full_like(target, self.weights[1]))
        return (loss * weights).mean()
    
class WeightedHuberLoss(torch.nn.Module):
    def __init__(self, zero_threshold=0.01, delta=0.5, non_zero_weight=5.0):
        super().__init__()
        self.threshold = zero_threshold
        self.delta = delta
        self.non_zero_weight = non_zero_weight
        
    def forward(self, preds, targets):
        # Create sample weights
        weights = torch.where(torch.abs(targets) > self.threshold, 
                            self.non_zero_weight, 1.0)
        
        # Calculate base Huber loss
        errors = targets - preds
        huber_loss = torch.where(
            torch.abs(errors) <= self.delta,
            0.5 * errors ** 2,
            self.delta * (torch.abs(errors) - 0.5 * self.delta)
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
        self.huber = torch.nn.HuberLoss(delta=delta, reduction='none')
        
    def forward(self, preds, targets):
        # Temporal decay weights
        seq_len = targets.size(1)
        time_weights = torch.tensor(
            [self.decay_factor ** (seq_len - i) for i in range(seq_len)],
            device=targets.device
        )
        
        # Event weights
        event_weights = torch.where(
            targets.abs() > 1e-4,
            self.non_zero_weight,
            self.zero_weight
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
        loss = errors ** 2
        
        # Focal component
        focal_weights = torch.where(
            is_spike,
            self.alpha * (1 - torch.exp(-errors)) ** self.gamma,
            (1 - self.alpha) * (torch.exp(-errors)) ** self.gamma
        )
        
        return torch.mean(focal_weights * loss)
    
class AsymmetricSpikeLoss(torch.nn.Module):
    def __init__(self, under_pred_penalty=4.0, over_pred_penalty=1.0, threshold=0.05):
        super().__init__()
        self.under_pred_penalty = under_pred_penalty
        self.over_pred_penalty = over_pred_penalty
        self.threshold = threshold
        
    def forward(self, preds, targets):
        errors = targets - preds
        is_spike = targets > self.threshold
        
        penalties = torch.where(
            errors > 0,  # Under-prediction
            self.under_pred_penalty,
            torch.where(
                errors < 0,  # Over-prediction
                self.over_pred_penalty,
                1.0  # Perfect prediction
            )
        )
        
        # Apply higher penalty for missing spikes
        spike_penalties = torch.where(
            is_spike,
            penalties * 2.0,
            penalties
        )
        
        return torch.mean(spike_penalties * errors ** 2)
    
class LogSpaceLoss(torch.nn.Module):
    def __init__(self, base_loss=torch.nn.HuberLoss(), epsilon=1e-8):
        super().__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        
    def forward(self, preds, targets):
        # Transform to log space with smoothing
        log_preds = torch.log(preds + self.epsilon)
        log_targets = torch.log(targets + self.epsilon)
        
        # Calculate loss in log space
        return self.base_loss(log_preds, log_targets)
    
class ZeroInflatedTweedieLoss(torch.nn.Module):
    def __init__(self, p=1.5, zero_weight=0.3, eps=1e-8):
        super().__init__()
        self.p = p  # 1 < p < 2 (1.5 is Poisson-gamma compound)
        self.zero_weight = zero_weight
        self.eps = eps
        
    def forward(self, preds, targets):
        # Add epsilon to avoid log(0)
        preds = torch.clamp(preds, min=self.eps)
        
        # Zero-inflation weight
        weights = torch.where(
            targets == 0,
            self.zero_weight,
            1.0
        )
        
        # Tweedie loss for p between 1-2
        loss = -(
            targets * torch.pow(preds, 1 - self.p) / (1 - self.p) - 
            torch.pow(preds, 2 - self.p) / (2 - self.p)
        )
        
        return torch.mean(weights * loss)
    
class HybridSpikeLoss(torch.nn.Module):
    def __init__(self, spike_threshold=0.1, alpha=0.7):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.alpha = alpha  # Weighting between MSE and spike detection
        
    def forward(self, preds, targets):
        # Standard MSE component
        mse_loss = F.mse_loss(preds, targets, reduction='none')
        
        # Spike detection component (penalize missing spikes)
        spike_mask = targets > self.spike_threshold
        spike_loss = F.binary_cross_entropy_with_logits(
            preds, 
            (targets > self.spike_threshold).float(),
            reduction='none'
        )
        
        # Combine losses
        hybrid_loss = self.alpha * mse_loss + (1 - self.alpha) * spike_loss
        return torch.mean(hybrid_loss)