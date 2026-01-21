# file: tweedie_stability_analysis.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Define the two versions of the Tweedie loss for comparison

class TweedieLossExp(torch.nn.Module):
    """Tweedie NLL with the canonical exp link function."""
    def __init__(self, p=1.5, eps=1e-8):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, preds, targets):
        mu = torch.clamp(torch.exp(preds), min=self.eps)
        loss = (torch.pow(mu, 2 - self.p) / (2 - self.p)) - \
               (targets * torch.pow(mu, 1 - self.p) / (1 - self.p))
        return loss

class TweedieLossSoftplus(torch.nn.Module):
    """Tweedie NLL with the stable softplus link function."""
    def __init__(self, p=1.5, eps=1e-8):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, preds, targets):
        mu = F.softplus(preds) + self.eps
        loss = (torch.pow(mu, 2 - self.p) / (2 - self.p)) - \
               (targets * torch.pow(mu, 1 - self.p) / (1 - self.p))
        return loss

def analyze_stability():
    """
    Analyzes and plots the loss and gradient landscapes for the two Tweedie loss implementations.
    """
    # Setup
    p = 1.5
    target_val = torch.tensor([2.0]) # A constant, non-zero target
    eta_range = torch.linspace(-10, 10, 400, requires_grad=True)

    loss_exp_fn = TweedieLossExp(p=p)
    loss_softplus_fn = TweedieLossSoftplus(p=p)

    # Calculate losses
    losses_exp = loss_exp_fn(eta_range, target_val)
    losses_softplus = loss_softplus_fn(eta_range, target_val)

    # Calculate gradients
    losses_exp.sum().backward()
    grads_exp = eta_range.grad.clone()
    eta_range.grad.zero_() # Reset gradients

    losses_softplus.sum().backward()
    grads_softplus = eta_range.grad.clone()

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

    # --- Plot 1: Loss Landscape ---
    ax1.plot(eta_range.detach().numpy(), losses_exp.detach().numpy(), 
             label='Loss with exp() link', color='r', linestyle='--')
    ax1.plot(eta_range.detach().numpy(), losses_softplus.detach().numpy(), 
             label='Loss with softplus() link (Current)', color='b', linewidth=2)
    ax1.set_title('Loss Landscape Comparison', fontsize=16)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_ylim(-10, 50) # Zoom in to a reasonable range to see the behavior
    ax1.legend()
    ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=0.5)

    # --- Plot 2: Gradient Landscape ---
    ax2.plot(eta_range.detach().numpy(), grads_exp.detach().numpy(), 
             label='Gradient with exp() link', color='r', linestyle='--')
    ax2.plot(eta_range.detach().numpy(), grads_softplus.detach().numpy(), 
             label='Gradient with softplus() link (Current)', color='b', linewidth=2)
    ax2.set_title('Gradient Landscape Comparison', fontsize=16)
    ax2.set_xlabel('Raw Model Output (eta)', fontsize=12)
    ax2.set_ylabel('Gradient (dL/d_eta)', fontsize=12)
    ax2.set_ylim(-10, 10) # Zoom in to a reasonable range
    ax2.legend()
    ax2.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)

    plt.suptitle('Proof of Numerical Stability: Tweedie Loss Link Functions', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save the plot
    output_path = 'tweedie_stability_analysis.png'
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")

if __name__ == '__main__':
    analyze_stability()
