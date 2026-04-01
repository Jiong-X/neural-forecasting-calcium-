# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
evaluate.py
-----------
Evaluation metrics and visualisation for probabilistic forecasts.
"""

import os
import torch
import matplotlib.pyplot as plt


def compute_metrics(model, loader, device, n_samples=100):
    """
    Returns a dict of evaluation metrics:
      - MAE   : mean absolute error of the predicted mean
      - RMSE  : root mean squared error of the predicted mean
      - CRPS  : continuous ranked probability score (lower = better)
      - Coverage90: empirical coverage of the 90% predictive interval
    """
    all_mean, all_logvar, all_y = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mean, logvar = model(x)
            all_mean.append(mean.cpu())
            all_logvar.append(logvar.cpu())
            all_y.append(y.cpu())

    mean   = torch.cat(all_mean)
    logvar = torch.cat(all_logvar)
    y      = torch.cat(all_y)
    std    = (0.5 * logvar).exp()

    mae  = (mean - y).abs().mean().item()
    rmse = ((mean - y) ** 2).mean().sqrt().item()

    # 90% interval coverage
    z90 = 1.645
    lower = mean - z90 * std
    upper = mean + z90 * std
    coverage = ((y >= lower) & (y <= upper)).float().mean().item()

    # CRPS for Gaussian: std * (1/sqrt(pi) - 2*phi(z) - z*(2*Phi(z)-1))
    # Approximate via sample-based CRPS
    samples = model.sample if hasattr(model, "sample") else None
    crps    = float("nan")  # placeholder — fill in with properscoring if available

    return {"MAE": mae, "RMSE": rmse, "Coverage90": coverage, "CRPS": crps}


def plot_predictions(model, loader, device, save_dir="results/figures", n_examples=4):
    """Plot predicted mean ± 2 std against ground truth for n_examples."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    with torch.no_grad():
        mean, logvar = model(x_batch)
    std  = (0.5 * logvar).exp().cpu()
    mean = mean.cpu()
    y    = y_batch.cpu()
    x    = x_batch.cpu()

    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3 * n_examples))
    for i, ax in enumerate(axes):
        t_ctx  = range(x.shape[1])
        t_pred = range(x.shape[1], x.shape[1] + y.shape[1])

        ax.plot(t_ctx,  x[i],      color="steelblue", label="Context")
        ax.plot(t_pred, y[i],      color="black",     label="Ground truth", linestyle="--")
        ax.plot(t_pred, mean[i],   color="tomato",    label="Predicted mean")
        ax.fill_between(t_pred,
                        mean[i] - 2 * std[i],
                        mean[i] + 2 * std[i],
                        alpha=0.3, color="tomato",   label="95% interval")
        ax.legend(fontsize=8)
        ax.set_title(f"Example {i+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=150)
    plt.close()
