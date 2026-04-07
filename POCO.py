"""
POCO (Population-Conditioned Forecaster) for predicting neural activity
from calcium imaging data.

Architecture (from standalone_poco.py in ~/POCO):
  - POYO Perceiver-IO with rotary time embeddings encodes each neuron's
    token sequence into a shared latent bottleneck.
  - An MLP conditioned on the POYO embedding produces the final forecast.

Reference: https://github.com/poyo-brain/poyo

Run with the 'poco' conda environment:
  /home/jiongx/micromamba/envs/poco/bin/python3 POCO.py

Input data
----------
  Expects data/processed/0.npz with a "PC" key of shape (N_pcs, T).
  Input size is inferred automatically.  Set N_PCS to an int to cap
  the number of components (e.g. 128), or None to use all.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from poco_src.standalone_poco import POCO, NeuralPredictionConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    """
    Sliding-window dataset returning (context, target) pairs.

    POCO expects (L, B, D) tensors, so each sample is returned as
    (context_len, D) and (pred_len, D); the DataLoader stacks them
    into (context_len, B, D) after transposing.

    Args:
        traces:      np.ndarray (T, N) — T time steps, N features.
        seq_len:     total window length (context + prediction).
        pred_len:    number of future steps to predict.
    """

    def __init__(self, traces: np.ndarray, seq_len: int = 64, pred_len: int = 16):
        traces = traces.astype(np.float32)
        mu = traces.mean(axis=0, keepdims=True)
        sd = traces.std(axis=0,  keepdims=True) + 1e-8
        traces = (traces - mu) / sd

        context_len = seq_len - pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t            : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])

        self.X = torch.tensor(np.array(X))   # (S, context_len, N)
        self.Y = torch.tensor(np.array(Y))   # (S, pred_len,    N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collate_poco(batch):
    """
    Collate (context, target) pairs into POCO's expected (L, B, D) format.
    Returns:
        x_list: list with one tensor of shape (context_len, B, N)
        y:      tensor of shape (pred_len, B, N)
    """
    X = torch.stack([b[0] for b in batch], dim=1)   # (context_len, B, N)
    Y = torch.stack([b[1] for b in batch], dim=1)   # (pred_len, B, N)
    return [X], Y


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.train()
    total = 0.0
    for x_list, Y in loader:
        x_list = [x.to(device) for x in x_list]
        Y = Y.to(device)
        optimizer.zero_grad()
        preds = model(x_list)          # list of (pred_len, B, N)
        loss  = criterion(preds[0], Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)   # max norm 5 per paper
        optimizer.step()
        total += loss.item() * Y.size(1)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    for x_list, Y in loader:
        x_list = [x.to(device) for x in x_list]
        Y = Y.to(device)
        preds = model(x_list)
        total_mse += criterion(preds[0], Y).item() * Y.size(1)
        total_mae += (preds[0] - Y).abs().mean().item() * Y.size(1)
    n = len(loader.dataset)
    return total_mse / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Paths ---
    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_calcium_poco.pt"
    RESULTS_PATH = "results/poco_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Hyperparameters ---
    N_PCS      = 128    # None = use all PCs; int to cap (e.g. 128)
    SEQ_LEN    = 64     # total window: context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16     # steps to forecast — matches paper main experiments
    BATCH_SIZE = 64     # matches paper spec
    EPOCHS     = 50
    LR         = 3e-4
    # 3:1:1 train/val/test split — matches paper partitioning ratio
    TRAIN_FRAC = 0.6
    VAL_FRAC   = 0.2
    # TEST_FRAC  = 0.2  (remainder)

    # POCO-specific
    COMPRESSION = 16    # time steps per token (must divide context length)
    NUM_LATENTS = 8     # number of Perceiver latent codes
    HIDDEN_DIM  = 128   # POYO / MLP hidden dimension — matches paper
    COND_DIM    = 1024  # MLP conditioning dimension  — matches paper
    NUM_LAYERS  = 1     # Perceiver depth
    NUM_HEADS   = 16    # self-attention heads         — matches paper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {DATA_PATH} ...")
    data = np.load(DATA_PATH)
    if "PC" in data:
        raw = data["PC"].astype(np.float32)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
    elif "M" in data:
        raw = data["M"].astype(np.float32)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if "valid_indices" in data:
            raw = raw[:, data["valid_indices"]]
    else:
        raise ValueError(f"No recognised key in {DATA_PATH}. Found: {list(data.keys())}")

    traces = raw[:, :N_PCS] if N_PCS is not None else raw
    T, N   = traces.shape
    print(f"Traces shape: {traces.shape}  (T={T}, features={N})")

    context_len = SEQ_LEN - PRED_LEN
    assert context_len % COMPRESSION == 0, (
        f"context_len ({context_len}) must be divisible by "
        f"compression_factor ({COMPRESSION})"
    )

    # --- Train / val / test split  (3:1:1 — matches paper) ---
    train_end = int(T * TRAIN_FRAC)
    val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

    train_ds = CalciumDataset(traces[:train_end],          seq_len=SEQ_LEN, pred_len=PRED_LEN)
    val_ds   = CalciumDataset(traces[train_end:val_end],   seq_len=SEQ_LEN, pred_len=PRED_LEN)
    test_ds  = CalciumDataset(traces[val_end:],            seq_len=SEQ_LEN, pred_len=PRED_LEN)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}  |  Test windows: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_poco, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_poco, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_poco, num_workers=0)

    # --- Build POCO config ---
    config = NeuralPredictionConfig()
    config.seq_length         = SEQ_LEN
    config.pred_length        = PRED_LEN
    config.compression_factor = COMPRESSION
    config.poyo_num_latents   = NUM_LATENTS
    config.decoder_hidden_size = HIDDEN_DIM
    config.conditioning_dim   = COND_DIM
    config.decoder_num_layers = NUM_LAYERS
    config.decoder_num_heads  = NUM_HEADS
    config.decoder_context_length = None
    config.freeze_backbone    = False
    config.freeze_conditioned_net = False

    # Single session: input_size is [[N]] — one dataset, one session
    input_size = [[N]]
    model = POCO(config, input_size).to(device)

    # FiLM conditioning weights are already zero-initialised inside POCO.__init__
    # (standalone_poco.py lines 838-841) — no manual init needed.
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # --- Training ---
    PATIENCE = 10   # early stopping — stop if val MSE doesn't improve for 10 epochs
    train_losses, val_mses, val_maes = [], [], []
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss       = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_mse)
        train_losses.append(train_loss)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

        tag = " *" if val_mse < best_val else ""
        if val_mse < best_val:
            best_val   = val_mse
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}{tag}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest val MSE: {best_val:.4f}  — weights saved to {MODEL_PATH}")

    # --- Test evaluation (best checkpoint, never seen during training) ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    test_mse, test_mae = eval_epoch(model, test_loader, criterion, device)
    print(f"Test  MSE: {test_mse:.4f}  |  Test MAE: {test_mae:.4f}")

    np.savez(RESULTS_PATH, train_losses=train_losses, val_mses=val_mses, val_maes=val_maes,
             test_mse=test_mse, test_mae=test_mae)
    print(f"Loss history saved to {RESULTS_PATH}")
