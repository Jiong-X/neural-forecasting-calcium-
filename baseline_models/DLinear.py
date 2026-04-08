"""
DLinear — Decomposition-Linear baseline for neural activity forecasting.

Architecture:
  1. Decompose each neuron's context window into:
       - Trend    : moving-average filter (kernel = context_len//4 * 2 + 1)
       - Seasonal : residual  (original - trend)
  2. Apply a separate linear layer (context_len → pred_len) to each component
  3. Sum the two outputs to produce the forecast

Compared to NLinear, DLinear explicitly separates slow drift (trend) from
fast oscillations (seasonal) — both present in calcium imaging traces.

Reference: Zeng et al. (2023) "Are Transformers Effective for Time Series
           Forecasting?" AAAI 2023.

Run with:
  /home/jiongx/micromamba/envs/comp0197-pt/bin/python3 DLinear.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from util import fetch_data_loaders

# ---------------------------------------------------------------------------
# Model — self-contained DLinear (no external dependencies)
# ---------------------------------------------------------------------------

class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):           # x: (B, L, N)
        pad = self.kernel_size // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end   = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """
    Decomposition-Linear (Zeng et al. 2023).
    Splits the context into trend (moving average) + seasonal (residual),
    applies separate linear projections, sums the outputs.
    Input/output: (L, B, N)
    """
    def __init__(self, context_len: int, pred_len: int, n_channels: int,
                 individual: bool = False):
        super().__init__()
        kernel = context_len // 4 * 2 + 1
        self.decomp     = _MovingAvg(kernel)
        self.pred_len   = pred_len
        self.individual = individual
        self.channels   = n_channels

        init_w = (1 / context_len) * torch.ones(pred_len, context_len)
        if individual:
            self.lin_s = nn.ModuleList([nn.Linear(context_len, pred_len)
                                        for _ in range(n_channels)])
            self.lin_t = nn.ModuleList([nn.Linear(context_len, pred_len)
                                        for _ in range(n_channels)])
            for l in self.lin_s + self.lin_t:
                l.weight = nn.Parameter(init_w.clone())
        else:
            self.lin_s = nn.Linear(context_len, pred_len)
            self.lin_t = nn.Linear(context_len, pred_len)
            self.lin_s.weight = nn.Parameter(init_w.clone())
            self.lin_t.weight = nn.Parameter(init_w.clone())

    def forward(self, x):           # x: (L, B, N)
        x = x.permute(1, 0, 2)     # → (B, L, N)
        trend    = self.decomp(x)
        seasonal = x - trend
        if self.individual:
            s_out = torch.stack([self.lin_s[i](seasonal[:, :, i])
                                 for i in range(self.channels)], dim=-1)
            t_out = torch.stack([self.lin_t[i](trend[:, :, i])
                                 for i in range(self.channels)], dim=-1)
        else:
            s_out = self.lin_s(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            t_out = self.lin_t(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out = s_out + t_out         # (B, pred_len, N)
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
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    MODEL_PATH   = "models/best_dlinear.pt"
    RESULTS_PATH = "results/dlinear_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    N_PCS      = 128
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16
    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-3
    VAL_FRAC   = 0.2
    TRAIN_FRAC = 0.6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    CONTEXT_LEN  = SEQ_LEN - PRED_LEN

    train_loader, val_loader, N = fetch_data_loaders("DLinear",SEQ_LEN, PRED_LEN, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE)
    
    # --- Model ---
    model = DLinear(CONTEXT_LEN, PRED_LEN, N, individual=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
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
