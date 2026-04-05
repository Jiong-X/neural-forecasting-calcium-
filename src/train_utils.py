# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
train_utils.py
--------------
Training and validation step functions for probabilistic neural activity forecasting.

Loss: Gaussian NLL — jointly trains mean prediction and aleatoric uncertainty.
  NLL = 0.5 * (logvar + (target - mean)^2 / exp(logvar))

Reference:
  Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
  NeurIPS 2017.
"""

import torch


def gaussian_nll_loss(mean: torch.Tensor,
                      logvar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian NLL averaged over all elements.

    Args:
        mean   : (B, pred_len, N)  predicted mean
        logvar : (B, pred_len, N)  predicted log-variance
        target : (B, pred_len, N)  ground truth

    Returns:
        scalar loss
    """
    var  = logvar.exp().clamp(min=1e-6)
    loss = 0.5 * (logvar + (target - mean).pow(2) / var)
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device: str) -> float:
    """
    One full pass over the training set.

    Returns:
        mean NLL loss over the epoch
    """
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)   # (B, context_len, N), (B, pred_len, N)
        optimizer.zero_grad()
        mean, logvar = model(x)
        loss = gaussian_nll_loss(mean, logvar, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device: str) -> float:
    """
    Evaluate on validation set.

    Returns:
        mean NLL loss
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mean, logvar = model(x)
            loss = gaussian_nll_loss(mean, logvar, y)
            total_loss += loss.item()
    return total_loss / len(loader)


def compute_mae(model, loader, device: str) -> float:
    """
    Compute mean absolute error using only the predicted mean.
    Useful for comparing against deterministic baselines.
    """
    model.eval()
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mean, _ = model(x)
            total_mae += (mean - y).abs().mean().item() * len(x)
            n += len(x)
    return total_mae / n
