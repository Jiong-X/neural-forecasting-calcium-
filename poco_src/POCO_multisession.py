"""
POCO Multi-Session — Population-Conditioned Forecaster trained across all
4 zebrafish subjects simultaneously.

Each subject is treated as a separate session. POCO learns:
  - Shared population dynamics via the Perceiver latent bottleneck
  - Subject-specific variation via per-session unit embeddings

Architecture difference from single-session POCO
-------------------------------------------------
  input_size = [[N], [N], [N], [N]]   — 4 sessions, same N PCs each
  forward(x_list)                      — x_list has 4 tensors, one per session

Data split
----------
  Subjects 0, 1, 2  →  training
  Subject  3         →  validation  (tests generalisation across animals)

Each training step yields one batch per session (4 batches total),
which are passed jointly to model.forward() as x_list.

Run with:
  /home/jiongx/micromamba/envs/poco/bin/python3 POCO_multisession.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from standalone_poco import POCO, NeuralPredictionConfig


# ---------------------------------------------------------------------------
# Dataset — one per session
# ---------------------------------------------------------------------------

class SessionDataset(Dataset):
    """
    Sliding-window dataset for a single session.

    Args:
        traces:      (T, N) float32 — z-scored PC traces
        context_len: number of input time steps
        pred_len:    number of future steps to predict
    """

    def __init__(self, traces: np.ndarray, context_len: int, pred_len: int):
        traces = traces.astype(np.float32)
        mu  = traces.mean(0, keepdims=True)
        sd  = traces.std(0,  keepdims=True) + 1e-8
        traces = (traces - mu) / sd

        X, Y = [], []
        seq_len = context_len + pred_len
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t            : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])

        self.X = torch.tensor(np.array(X))   # (S, context_len, N)
        self.Y = torch.tensor(np.array(Y))   # (S, pred_len,    N)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ---------------------------------------------------------------------------
# Multi-session loader — zips one batch per session each step
# ---------------------------------------------------------------------------

class MultiSessionLoader:
    """
    Wraps multiple DataLoaders (one per session) and yields
    (x_list, y_list) at each step where:
        x_list: list of (context_len, B, N) tensors — one per session
        y_list: list of (pred_len,    B, N) tensors — one per session
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for batches in zip(*self.loaders):
            x_list = [b[0].permute(1, 0, 2) for b in batches]  # (L, B, N)
            y_list = [b[1].permute(1, 0, 2) for b in batches]  # (P, B, N)
            yield x_list, y_list

    def __len__(self):
        return min(len(l) for l in self.loaders)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x_list, y_list in loader:
        x_list = [x.to(device) for x in x_list]
        y_list = [y.to(device) for y in y_list]
        optimiser.zero_grad()
        preds = model(x_list)                           # list of (P, B, N)
        loss  = sum(criterion(p, y) for p, y in zip(preds, y_list)) / len(preds)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        b = y_list[0].size(1)
        total += loss.item() * b
        n     += b
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    for x_list, y_list in loader:
        x_list = [x.to(device) for x in x_list]
        y_list = [y.to(device) for y in y_list]
        preds  = model(x_list)
        mse = sum(criterion(p, y) for p, y in zip(preds, y_list)) / len(preds)
        mae = sum((p - y).abs().mean() for p, y in zip(preds, y_list)) / len(preds)
        b   = y_list[0].size(1)
        total_mse += mse.item() * b
        total_mae += mae.item() * b
        n         += b
    return total_mse / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_DIR     = "data/processed"
    MODEL_PATH   = "models/best_poco_multisession.pt"
    RESULTS_PATH = "results/poco_multisession_losses.npz"
    LOG_PATH     = "results/poco_multisession_log.txt"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Hyperparameters ---
    N_PCS       = 128
    CONTEXT_LEN = 48    # matches paper main experiments (C=48)
    PRED_LEN    = 16    # matches paper main experiments (P=16)
    BATCH_SIZE  = 16
    EPOCHS      = 50
    LR          = 3e-4
    COMPRESSION = 16
    NUM_LATENTS = 8
    HIDDEN_DIM  = 64
    COND_DIM    = 128
    NUM_LAYERS  = 1
    NUM_HEADS   = 8

    # Train on subjects 0,1,2 — validate on subject 3
    TRAIN_SUBJECTS = [0, 1, 2]
    VAL_SUBJECTS   = [3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load all sessions ---
    def load_traces(subject_id):
        d   = np.load(os.path.join(DATA_DIR, f"{subject_id}.npz"))
        raw = d["PC"].astype(np.float32)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        return raw[:, :N_PCS]

    train_traces = [load_traces(i) for i in TRAIN_SUBJECTS]
    val_traces   = [load_traces(i) for i in VAL_SUBJECTS]

    for i, t in zip(TRAIN_SUBJECTS, train_traces):
        print(f"  Train subject {i}: {t.shape}")
    for i, t in zip(VAL_SUBJECTS, val_traces):
        print(f"  Val   subject {i}: {t.shape}")

    # --- Datasets ---
    train_datasets = [SessionDataset(t, CONTEXT_LEN, PRED_LEN) for t in train_traces]
    val_datasets   = [SessionDataset(t, CONTEXT_LEN, PRED_LEN) for t in val_traces]

    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,
                                drop_last=True) for ds in train_datasets]
    val_loaders   = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                     for ds in val_datasets]

    train_loader = MultiSessionLoader(train_loaders)
    val_loader   = MultiSessionLoader(val_loaders)

    print(f"Train steps/epoch: {len(train_loader)}")
    print(f"Val   steps/epoch: {len(val_loader)}")

    # --- POCO config ---
    N = N_PCS
    config = NeuralPredictionConfig()
    config.seq_length             = CONTEXT_LEN + PRED_LEN
    config.pred_length            = PRED_LEN
    config.compression_factor     = COMPRESSION
    config.poyo_num_latents       = NUM_LATENTS
    config.decoder_hidden_size    = HIDDEN_DIM
    config.conditioning_dim       = COND_DIM
    config.decoder_num_layers     = NUM_LAYERS
    config.decoder_num_heads      = NUM_HEADS
    config.decoder_context_length = None
    config.freeze_backbone        = False
    config.freeze_conditioned_net = False

    # Multi-session: 3 training sessions, each with N neurons
    input_size = [[N]] * len(TRAIN_SUBJECTS)
    model      = POCO(config, input_size).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # --- Training ---
    train_losses, val_mses, val_maes = [], [], []
    best_mse = float("inf")

    # Build a single-session val loader for the val subject
    val_input_size = [[N]] * len(VAL_SUBJECTS)
    val_model_cfg  = NeuralPredictionConfig()
    val_model_cfg.__dict__.update(config.__dict__)

    log_lines = []
    for epoch in range(1, EPOCHS + 1):
        train_mse        = train_epoch(model, train_loader, optimiser, criterion, device)
        val_mse, val_mae = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_mse)

        train_losses.append(train_mse)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

        tag = " *" if val_mse < best_mse else ""
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), MODEL_PATH)

        line = (f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train_mse={train_mse:.4f}  "
                f"val_mse={val_mse:.4f}  "
                f"val_mae={val_mae:.4f}{tag}")
        print(line)
        log_lines.append(line)

    summary = f"\nBest val MSE: {best_mse:.4f}  — saved to {MODEL_PATH}"
    print(summary)
    log_lines.append(summary)

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    np.savez(RESULTS_PATH,
             train_losses=train_losses,
             val_mses=val_mses,
             val_maes=val_maes)
    print(f"Results saved to {RESULTS_PATH}")
