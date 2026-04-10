"""
TSMixer — Time-Series Mixer baseline for neural activity forecasting.

Architecture:
  Alternating MLP blocks applied along two axes:
    - Temporal mixing  : MLP across the time axis  (shared across neurons)
    - Feature mixing   : MLP across the neuron axis (shared across time steps)
  Each block uses LayerNorm + residual connection.

Intuition for calcium imaging:
  - Temporal MLP captures dynamics within each neuron's trace.
  - Feature MLP captures cross-neuron (population) interactions.

Reference: Chen et al. (2023) "TSMixer: An All-MLP Architecture for
           Time Series Forecasting." arXiv:2303.06053.

Run with:
  /home/jiongx/micromamba/envs/comp0197-pt/bin/python3 TSMixer.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util import fetch_data_loaders

# ---------------------------------------------------------------------------
# Model — self-contained TSMixer (no external dependencies)
# ---------------------------------------------------------------------------

class _TimeMixBlock(nn.Module):
    """MLP along the time axis, shared across channels."""
    def __init__(self, seq_len: int, ff_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.fc1  = nn.Linear(seq_len, ff_dim)
        self.fc2  = nn.Linear(ff_dim, seq_len)

    def forward(self, x):           # x: (B, L, N)
        r = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)   # LN over time
        x = x.transpose(1, 2)                              # (B, N, L)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).transpose(1, 2)                    # (B, L, N)
        return x + r


class _FeatMixBlock(nn.Module):
    """MLP along the channel axis, shared across time steps."""
    def __init__(self, n_channels: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(n_channels)
        self.fc1     = nn.Linear(n_channels, ff_dim)
        self.fc2     = nn.Linear(ff_dim, n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):           # x: (B, L, N)
        r = x
        x = self.norm(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x + r

class TSMixer(nn.Module):
    """
    TSMixer (Chen et al. 2023) — alternating temporal and feature MLP mixing.
    Input/output: (L, B, N)
    """
    def __init__(self, context_len: int, pred_len: int, n_channels: int,
                 ff_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(_TimeMixBlock(context_len, ff_dim))
            self.blocks.append(_FeatMixBlock(n_channels,  ff_dim, dropout))
        self.head = nn.Linear(context_len, pred_len)

    def forward(self, x):           # x: (L, B, N)
        x = x.permute(1, 0, 2)     # → (B, L, N)
        for blk in self.blocks:
            x = blk(x)
        # project time axis: (B, N, L) → (B, N, pred_len) → (B, pred_len, N)
        out = self.head(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out.permute(1, 0, 2) # → (pred_len, B, N)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimiser.zero_grad()
        pred = model(X)                         # (pred_len, B, N)
        loss = criterion(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        total += loss.item() * Y.size(1)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        mse += criterion(pred, Y).item() * Y.size(1)
        mae += (pred - Y).abs().mean().item() * Y.size(1)
        n   += Y.size(1)
    return mse / n, mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_tsmixer.pt"
    RESULTS_PATH = "results/tsmixer_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    N_PCS      = 128
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16
    FF_DIM     = 64   # hidden dim of MLP mixing blocks (paper default)
    N_LAYERS   = 2    # number of mixer blocks (paper default)
    DROPOUT    = 0.1  # paper default
    BATCH_SIZE = 64   # paper default
    EPOCHS     = 50
    LR         = 3e-4        # paper default
    WEIGHT_DECAY = 1e-4      # paper default
    GRAD_CLIP    = 5.0       # paper default
    VAL_FRAC   = 0.2
    TRAIN_FRAC = 0.6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

     # --- Data ---
    CONTEXT_LEN  = SEQ_LEN - PRED_LEN

    train_loader, val_loader, N = fetch_data_loaders("TSMixer", SEQ_LEN, PRED_LEN, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE)
    
    # --- Model ---
    
    model = TSMixer(CONTEXT_LEN, PRED_LEN, N, ff_dim=FF_DIM, n_layers=N_LAYERS, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_mses, val_maes = [], [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss       = train_epoch(model, train_loader, optimiser, criterion, device)
        val_mse, val_mae = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_mse)
        train_losses.append(train_loss)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

        tag = " *" if val_mse < best_val else ""
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train={train_loss:.4f}  "
              f"val_mse={val_mse:.4f}  "
              f"val_mae={val_mae:.4f}{tag}")

    print(f"\nBest val MSE: {best_val:.4f}  — saved to {MODEL_PATH}")
    np.savez(RESULTS_PATH, train_losses=train_losses,
             val_mses=val_mses, val_maes=val_maes)
    print(f"Loss history saved to {RESULTS_PATH}")
