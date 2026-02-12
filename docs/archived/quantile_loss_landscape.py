import torch
import matplotlib.pyplot as plt
import sys

# Ensure the path is correct to import from the parent directory's module
sys.path.append("..")
from views_r2darts2.utils.loss import AsymmetricQuantileLoss


def plot_quantile_loss_landscape():
    """
    Generates and saves plots of the loss and gradient landscapes for
    AsymmetricQuantileLoss with various tau values.
    """
    # --- Setup ---
    preds_range = torch.linspace(-2.0, 4.0, 600)
    target = torch.tensor([1.0])
    tau_values = [0.25, 0.50, 0.75, 0.95]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Behavioral Analysis of AsymmetricQuantileLoss", fontsize=16)

    # --- Loss Landscape Plot ---
    ax_loss = axs[0]
    ax_loss.set_title("Loss vs. Prediction Error")
    ax_loss.set_xlabel("Prediction Error (pred - target)")
    ax_loss.set_ylabel("Loss Value")
    ax_loss.grid(True)
    ax_loss.axvline(
        x=0, color="black", linestyle="--", label="Perfect Prediction (Error = 0)"
    )

    # --- Gradient Landscape Plot ---
    ax_grad = axs[1]
    ax_grad.set_title("Gradient vs. Prediction Error")
    ax_grad.set_xlabel("Prediction Error (pred - target)")
    ax_grad.set_ylabel("Gradient w.r.t. Prediction")
    ax_grad.grid(True)
    ax_grad.axvline(x=0, color="black", linestyle="--")
    ax_grad.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    # --- Analysis Loop ---
    for tau in tau_values:
        loss_fn = AsymmetricQuantileLoss(
            tau=tau, non_zero_weight=1.0
        )  # Disable weighting for clarity
        losses = []
        grads = []

        for pred_val in preds_range:
            pred = pred_val.clone().requires_grad_()
            loss = loss_fn(pred, target)
            loss.backward()
            losses.append(loss.item())
            # Gradient is d(loss)/d(pred), and error = target - pred, so d(loss)/d(error) = -d(loss)/d(pred)
            # The canonical gradient is d(loss)/d(error), so we flip the sign.
            grads.append(-pred.grad.item())

        prediction_errors = (preds_range - target).numpy()
        ax_loss.plot(prediction_errors, losses, label=f"tau = {tau}")
        ax_grad.plot(prediction_errors, grads, label=f"tau = {tau}", linestyle="-")

    ax_loss.legend()
    ax_grad.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("quantile_loss_landscape.png")
    print("\nBehavioral analysis complete. Plot saved to quantile_loss_landscape.png")


if __name__ == "__main__":
    plot_quantile_loss_landscape()
