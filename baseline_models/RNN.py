"""
Vanilla RNN for predicting neural activity from calcium imaging data.
Uses nn.RNN (tanh activations, no gating) — not LSTM or GRU.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    def __init__(self, traces: np.ndarray, seq_len: int = 50, pred_steps: int = 1):
        traces = traces.astype(np.float32)
        mu = traces.mean(axis=0, keepdims=True)
        sd = traces.std(axis=0, keepdims=True) + 1e-8
        traces = (traces - mu) / sd

        X, y = [], []
        for t in range(len(traces) - seq_len - pred_steps + 1):
            X.append(traces[t : t + seq_len])
            y.append(traces[t + seq_len : t + seq_len + pred_steps])

        self.X = torch.tensor(np.array(X))
        self.y = torch.tensor(np.array(y))

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CalciumVanillaRNN(nn.Module):
    """
    Multi-layer vanilla RNN (nn.RNN, tanh) for calcium imaging forecasting.
    Horizon is dynamic — rolled forward autoregressively at inference time.
    """

    def __init__(
        self,
        n_neurons: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        default_pred_steps: int = 1,
    ):
        super().__init__()
        self.default_pred_steps = default_pred_steps
        self.n_neurons = n_neurons

        self.rnn = nn.RNN(
            input_size=n_neurons,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, n_neurons)

    def forward(self, x: torch.Tensor, pred_steps: int | None = None) -> torch.Tensor:
        """
        Args:
            x:          (B, T, N)
            pred_steps: forecast horizon; defaults to self.default_pred_steps
        Returns:
            (B, pred_steps, N)
        """
        if pred_steps is None:
            pred_steps = self.default_pred_steps

        _, h = self.rnn(x)                          # h: (num_layers, B, H)

        preds = []
        inp = x[:, -1:, :]                          # (B, 1, N)
        for _ in range(pred_steps):
            out, h = self.rnn(inp, h)               # (B, 1, H)
            step   = self.head(out)                 # (B, 1, N)
            preds.append(step)
            inp = step.detach()

        return torch.cat(preds, dim=1)              # (B, pred_steps, N)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X, pred_steps=y.size(1))
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X, pred_steps=y.size(1))
        total_mse += criterion(pred, y).item() * len(X)
        total_mae += (pred - y).abs().mean().item() * len(X)
    n = len(loader.dataset)
    return total_mse / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_vanilla_rnn.pt"
    RESULTS_PATH = "results/rnn_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    N_PCS      = None
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_STEPS = 16
    HIDDEN     = 256
    LAYERS     = 2
    DROPOUT    = 0.2
    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-3
    VAL_FRAC   = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_PATH)
    raw  = data["PC"].astype(np.float32)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    traces = raw[:, :N_PCS] if N_PCS is not None else raw
    T, N   = traces.shape
    print(f"Traces shape: {traces.shape}  (T={T}, features={N})")

    split    = int(T * (1 - VAL_FRAC))
    train_ds = CalciumDataset(traces[:split], seq_len=SEQ_LEN, pred_steps=PRED_STEPS)
    val_ds   = CalciumDataset(traces[split:], seq_len=SEQ_LEN, pred_steps=PRED_STEPS)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CalciumVanillaRNN(
        n_neurons=N, hidden_size=HIDDEN, num_layers=LAYERS,
        dropout=DROPOUT, default_pred_steps=PRED_STEPS,
    ).to(device)
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_mses, val_maes = [], [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss       = train(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_mse)
        train_losses.append(train_loss)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

        tag = " *" if val_mse < best_val else ""
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}{tag}")

    print(f"\nBest val MSE: {best_val:.4f}  — weights saved to {MODEL_PATH}")
    np.savez(RESULTS_PATH, train_losses=train_losses, val_mses=val_mses, val_maes=val_maes)
    print(f"Loss history saved to {RESULTS_PATH}")
