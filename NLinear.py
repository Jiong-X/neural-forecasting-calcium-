"""
NLinear — Normalisation-Linear baseline for neural activity forecasting.

Architecture:
  1. Subtract the last observed value (removes level shift / non-stationarity)
  2. Apply a single shared linear layer: context_len → pred_len  (per neuron)
  3. Add the last observed value back

This is the simplest possible learned forecaster — one scalar weight per
(context step, forecast step) pair, shared across all neurons.

Reference: Zeng et al. (2023) "Are Transformers Effective for Time Series
           Forecasting?" AAAI 2023.

Run with:
  /home/jiongx/micromamba/envs/comp0197-pt/bin/python3 NLinear.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model — self-contained NLinear (no external dependencies)
# ---------------------------------------------------------------------------

class NLinear(nn.Module):
    """
    Normalisation-Linear (Zeng et al. 2023).
    Subtracts the last observed value, applies a shared linear
    projection context_len → pred_len, then adds the last value back.
    Input/output: (L, B, N)
    """
    def __init__(self, context_len: int, pred_len: int, n_channels: int,
                 individual: bool = False):
        super().__init__()
        self.pred_len   = pred_len
        self.individual = individual
        self.channels   = n_channels
        if individual:
            self.linear = nn.ModuleList(
                [nn.Linear(context_len, pred_len) for _ in range(n_channels)])
        else:
            self.linear = nn.Linear(context_len, pred_len)

    def forward(self, x):           # x: (L, B, N)
        x = x.permute(1, 0, 2)     # → (B, L, N)
        last = x[:, -1:, :].detach()
        x = x - last
        if self.individual:
            out = torch.stack(
                [self.linear[i](x[:, :, i]) for i in range(self.channels)],
                dim=-1)             # (B, pred_len, N)
        else:
            out = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + last
        return out.permute(1, 0, 2)  # → (pred_len, B, N)


# ---------------------------------------------------------------------------
# Dataset  (batch-first: B, T, N)
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    def __init__(self, traces: np.ndarray, seq_len: int = 64, pred_len: int = 16):
        traces = traces.astype(np.float32)
        mu = traces.mean(0, keepdims=True)
        sd = traces.std(0,  keepdims=True) + 1e-8
        traces = (traces - mu) / sd

        context_len = seq_len - pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t            : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])

        self.X = torch.tensor(np.array(X))
        self.Y = torch.tensor(np.array(Y))

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def collate_lbd(batch):
    """Stack into (L, B, D) format expected by POCO single-session models."""
    X = torch.stack([b[0] for b in batch], dim=1)   # (context_len, B, N)
    Y = torch.stack([b[1] for b in batch], dim=1)   # (pred_len,    B, N)
    return X, Y


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
    MODEL_PATH   = "models/best_nlinear.pt"
    RESULTS_PATH = "results/nlinear_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    N_PCS      = 128
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16
    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-3
    VAL_FRAC   = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(DATA_PATH)
    raw  = data["PC"].astype(np.float32)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    traces = raw[:, :N_PCS] if N_PCS else raw
    T, N   = traces.shape
    print(f"Traces: {T} steps x {N} features")

    split    = int(T * (1 - VAL_FRAC))
    train_ds = CalciumDataset(traces[:split], SEQ_LEN, PRED_LEN)
    val_ds   = CalciumDataset(traces[split:], SEQ_LEN, PRED_LEN)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_lbd, num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=collate_lbd, num_workers=0)

    CONTEXT_LEN = SEQ_LEN - PRED_LEN
    model = NLinear(CONTEXT_LEN, PRED_LEN, N, individual=False).to(device)
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
