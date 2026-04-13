# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
Probabilistic POCO — Multi-Session version.

Combines:
  - ProbabilisticPOCO  from POCO_prob.py      (Gaussian output head, NLL training)
  - MultiSessionLoader from POCO_multisession.py (3 train subjects, 1 val subject)

Data split
----------
  Subjects 0, 1, 2  →  training
  Subject  3         →  validation (held-out animal)

Each forward pass returns a list of torch.distributions.Normal, one per session.
NLL loss is averaged across sessions.

Run with:
  /home/jiongx/micromamba/envs/poco/bin/python3 POCO_prob_multisession.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

from src.poco_src.standalone_poco import POCO, NeuralPredictionConfig
from src.poco_src.prob import ProbabilisticPOCO, nll_loss


# ---------------------------------------------------------------------------
# Dataset (same as POCO_multisession)
# ---------------------------------------------------------------------------

class SessionDataset(Dataset):
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

        self.X = torch.tensor(np.array(X))
        self.Y = torch.tensor(np.array(Y))

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


class MultiSessionLoader:
    """Yields (x_list, y_list) — one (L, B, D) tensor per session per step."""

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for batches in zip(*self.loaders):
            x_list = [b[0].permute(1, 0, 2) for b in batches]
            y_list = [b[1].permute(1, 0, 2) for b in batches]
            yield x_list, y_list

    def __len__(self):
        return min(len(l) for l in self.loaders)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device):
    model.train()
    total, n = 0.0, 0
    for x_list, y_list in loader:
        x_list = [x.to(device) for x in x_list]
        y_list = [y.to(device) for y in y_list]
        optimiser.zero_grad()
        dists = model(x_list)
        loss  = nll_loss(dists, y_list)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        b = y_list[0].size(1)
        total += loss.item() * b
        n     += b
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_nll, total_mae, n = 0.0, 0.0, 0
    for x_list, y_list in loader:
        x_list = [x.to(device) for x in x_list]
        y_list = [y.to(device) for y in y_list]
        dists  = model(x_list)
        total_nll += nll_loss(dists, y_list).item() * y_list[0].size(1)
        total_mae += sum(
            (d.mean - y).abs().mean().item()
            for d, y in zip(dists, y_list)
        ) / len(dists) * y_list[0].size(1)
        n += y_list[0].size(1)
    return total_nll / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_DIR     = "data/processed"
    MODEL_PATH   = "models/best_poco_prob_multisession.pt"
    RESULTS_PATH = "results/poco_prob_multisession_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Hyperparameters ---
    N_PCS           = 128
    CONTEXT_LEN     = 48    # matches paper main experiments (C=48)
    PRED_LEN        = 16    # matches paper main experiments (P=16)
    BATCH_SIZE      = 16
    EPOCHS          = 50
    LR              = 3e-4
    COMPRESSION     = 16
    NUM_LATENTS     = 8
    HIDDEN_DIM      = 64
    COND_DIM        = 128
    NUM_LAYERS      = 1
    NUM_HEADS       = 8
    TRAIN_SUBJECTS  = [0, 1, 2]
    VAL_SUBJECTS    = [3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load traces ---
    def load_traces(sid):
        d   = np.load(os.path.join(DATA_DIR, f"{sid}.npz"))
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

    # --- Datasets & loaders ---
    train_loaders = [
        DataLoader(SessionDataset(t, CONTEXT_LEN, PRED_LEN),
                   batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
        for t in train_traces
    ]
    val_loaders = [
        DataLoader(SessionDataset(t, CONTEXT_LEN, PRED_LEN),
                   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for t in val_traces
    ]
    train_loader = MultiSessionLoader(train_loaders)
    val_loader   = MultiSessionLoader(val_loaders)
    print(f"Train steps/epoch: {len(train_loader)}  |  Val steps: {len(val_loader)}")

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

    input_size = [[N]] * len(TRAIN_SUBJECTS)
    model      = ProbabilisticPOCO(config, input_size).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    # --- Training ---
    train_losses, val_nlls, val_maes = [], [], []
    best_nll = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_nll        = train_epoch(model, train_loader, optimiser, device)
        val_nll, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step(val_nll)

        train_losses.append(train_nll)
        val_nlls.append(val_nll)
        val_maes.append(val_mae)

        tag = " *" if val_nll < best_nll else ""
        if val_nll < best_nll:
            best_nll = val_nll
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_nll={train_nll:.4f}  "
              f"val_nll={val_nll:.4f}  "
              f"val_mae={val_mae:.4f}{tag}")

    print(f"\nBest val NLL: {best_nll:.4f}  — saved to {MODEL_PATH}")
    np.savez(RESULTS_PATH,
             train_losses=train_losses,
             val_nlls=val_nlls,
             val_maes=val_maes)
    print(f"Results saved to {RESULTS_PATH}")
