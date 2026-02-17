import torch
import matplotlib.pyplot as plt
import numpy as np

# Assuming the loss functions are in the correct path
from views_r2darts2.math import ShrinkageLoss, TweedieLoss


def generate_synthetic_data(
    n_samples=1000, zero_inflation_prob=0.9, shape=1.5, scale=2.0
):
    """
    Generates synthetic data mimicking conflict data properties.

    Args:
        n_samples (int): Number of samples to generate.
        zero_inflation_prob (float): Probability of a sample being zero.
        shape (float): Shape parameter for the Gamma distribution (controls skewness).
        scale (float): Scale parameter for the Gamma distribution (controls spread).

    Returns:
        torch.Tensor: A tensor of synthetic data.
    """
    is_zero = np.random.rand(n_samples) < zero_inflation_prob
    non_zero_values = np.random.gamma(shape, scale, n_samples)

    raw_data = np.where(is_zero, 0, non_zero_values)
    return torch.from_numpy(raw_data.astype(np.float32))


def analyze_loss_functions():
    """
    Generates plots for the loss and gradient landscapes of TweedieLoss and ShrinkageLoss
    across different data transformations.
    """
    # --- Setup ---
    targets_raw = generate_synthetic_data()
    targets_log1p = torch.log1p(targets_raw)
    targets_asinh = torch.asinh(targets_raw)

    transformations = {
        "Raw": (targets_raw, torch.linspace(0, 15, 400)),
        "Log1p": (targets_log1p, torch.linspace(0, 4, 400)),
        "Asinh": (targets_asinh, torch.linspace(0, 4, 400)),
    }

    fig, axs = plt.subplots(len(transformations), 2, figsize=(15, 18))
    fig.suptitle(
        "Loss and Gradient Landscape Analysis Across Transformations", fontsize=16
    )

    for i, (trans_name, (targets, preds_range)) in enumerate(transformations.items()):
        # --- TweedieLoss Analysis ---
        tweedie_loss_fn = TweedieLoss(p=1.5)
        tweedie_losses = []

        for pred_val in preds_range:
            # For Tweedie, preds are in 'eta' space (before softplus)
            pred = torch.tensor([pred_val], requires_grad=True)
            loss = tweedie_loss_fn(pred.expand_as(targets), targets)
            tweedie_losses.append(loss.item())

        # Plot Tweedie Loss
        ax_loss = axs[i, 0]
        ax_loss.plot(preds_range.numpy(), tweedie_losses, label="TweedieLoss (p=1.5)")
        ax_loss.set_title(f"Tweedie Loss vs. Prediction ({trans_name} Targets)")
        ax_loss.set_xlabel("Prediction Value (eta)")
        ax_loss.set_ylabel("Mean Loss")
        ax_loss.legend()
        ax_loss.grid(True)
        ax_loss.set_ylim(bottom=0)

        # --- ShrinkageLoss Analysis ---
        shrinkage_loss_fn = ShrinkageLoss(
            a=5.0, c=0.5
        )  # Adjusted params for broader view
        shrinkage_losses = []

        for pred_val in preds_range:
            pred = torch.tensor([pred_val])
            # ShrinkageLoss expects preds and targets on the same scale
            loss = shrinkage_loss_fn(pred.expand_as(targets), targets)
            shrinkage_losses.append(loss.item())

        # Plot Shrinkage Loss
        ax_grad = axs[i, 1]
        ax_grad.plot(
            preds_range.numpy(),
            shrinkage_losses,
            label="ShrinkageLoss (a=5.0, c=0.5)",
            color="orange",
        )
        ax_grad.set_title(f"Shrinkage Loss vs. Prediction ({trans_name} Targets)")
        ax_grad.set_xlabel("Prediction Value")
        ax_grad.set_ylabel("Mean Loss")
        ax_grad.legend()
        ax_grad.grid(True)
        ax_grad.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("loss_landscape_analysis_with_transforms.png")
    print(
        "Analysis complete. Plot saved to loss_landscape_analysis_with_transforms.png"
    )


if __name__ == "__main__":
    analyze_loss_functions()
