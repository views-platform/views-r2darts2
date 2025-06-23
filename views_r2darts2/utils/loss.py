import torch

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
    
# class WeightedHuberLoss(torch.nn.Module):
#     def __init__(self, weights, zero_threshold=0.01, delta=0.5):
#         super().__init__()
#         self.weights = weights
#         self.threshold = zero_threshold
#         self.huber = torch.nn.HuberLoss(delta=delta)
        
#     def forward(self, preds, targets):
#         # Create weight matrix
#         weights = torch.where(torch.abs(targets) > self.threshold, 
#                             self.weights[1], self.weights[0])
#         return torch.mean(weights * self.huber(preds, targets))
    
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