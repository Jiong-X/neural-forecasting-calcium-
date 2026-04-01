# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
train_utils.py
--------------
Training and validation step functions.
Loss: Gaussian negative log-likelihood (NLL) — trains both mean and uncertainty.
"""

import torch


def gaussian_nll_loss(mean, logvar, target):
    """
    Gaussian NLL: 0.5 * (logvar + (target - mean)^2 / var)
    Trains the model to predict both the mean and the aleatoric uncertainty.
    """
    var  = logvar.exp()
    loss = 0.5 * (logvar + (target - mean) ** 2 / var)
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        mean, logvar = model(x)
        loss = gaussian_nll_loss(mean, logvar, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mean, logvar = model(x)
            loss = gaussian_nll_loss(mean, logvar, y)
            total_loss += loss.item()
    return total_loss / len(loader)
