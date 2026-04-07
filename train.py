# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
train.py
--------
End-to-end training pipeline for probabilistic neural activity forecasting.

Data is retrieved automatically:
  - loads data/processed/0.npz if available
  - preprocesses data/raw/subject_0/TimeSeries.h5 if available
  - downloads from Janelia figshare otherwise (~2 GB, first run only)

Run:
    python train.py

Outputs:
    models/saved/model.pt          best model checkpoint
    results/train_losses.npz       loss curves (train NLL, val NLL, val MAE)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from src.dataset    import get_dataset, get_test_dataset
from src.model      import ProbabilisticForecaster
from src.train_utils import train_one_epoch, validate, compute_mae

# ---------------------------------------------------------------------------
# Config — matching POCO paper defaults
# ---------------------------------------------------------------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH  = 64       # context (48) + horizon (16)
PRED_LENGTH = 16       # prediction horizon
N_CHANNELS  = 128      # top-128 principal components
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 3e-4     # AdamW learning rate (paper default)
WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
PATIENCE    = 10       # early stopping patience (epochs)
SAVE_PATH   = "models/saved/model.pt"
RESULTS_PATH= "results/train_losses.npz"

# ---------------------------------------------------------------------------
# Data — auto-retrieved
# ---------------------------------------------------------------------------
print(f"Device: {DEVICE}")
print("Loading data (downloads automatically if not present)...")

train_dataset, val_dataset = get_dataset(
    seq_length  = SEQ_LENGTH,
    pred_length = PRED_LENGTH,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = ProbabilisticForecaster(
    seq_length  = SEQ_LENGTH,
    pred_length = PRED_LENGTH,
    n_channels  = N_CHANNELS,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: Probabilistic POCO  |  Parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(SAVE_PATH),   exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_PATH) or "results", exist_ok=True)

best_val_loss  = float("inf")
no_improve     = 0
train_losses, val_losses, val_maes = [], [], []

print(f"\n{'Epoch':>6}  {'Train NLL':>10}  {'Val NLL':>10}  {'Val MAE':>9}")
print("-" * 44)

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss   = validate(model, val_loader, DEVICE)
    val_mae    = compute_mae(model, val_loader, DEVICE)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_maes.append(val_mae)

    tag = " *" if val_loss < best_val_loss else ""
    print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_mae:>9.4f}{tag}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), SAVE_PATH)
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

print(f"\nBest val NLL: {best_val_loss:.4f}  — saved to {SAVE_PATH}")

# ---------------------------------------------------------------------------
# Test evaluation — run once on held-out set using best checkpoint
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))

test_dataset = get_test_dataset(seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_loss = validate(model, test_loader, DEVICE)
test_mae  = compute_mae(model, test_loader, DEVICE)
print(f"Test  NLL : {test_loss:.4f}")
print(f"Test  MAE : {test_mae:.4f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
np.savez(RESULTS_PATH,
         train_losses = np.array(train_losses),
         val_losses   = np.array(val_losses),
         val_maes     = np.array(val_maes),
         test_nll     = test_loss,
         test_mae     = test_mae)
print(f"Loss curves saved to {RESULTS_PATH}")
print("Training complete.")
