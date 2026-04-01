# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
train.py
--------
Automates data retrieval, model training, and saving of final weights.
Run: python train.py
"""

import os
import torch
from torch.utils.data import DataLoader

from src.dataset import get_dataset
from src.model import ProbabilisticForecaster
from src.train_utils import train_one_epoch, validate

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH  = 64
PRED_LENGTH = 16
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SAVE_PATH   = "models/saved/model.pt"

# ── Data ──────────────────────────────────────────────────────────────────────
train_dataset, val_dataset = get_dataset(seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = ProbabilisticForecaster(seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss   = validate(model, val_loader, DEVICE)
    scheduler.step()
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  -> Saved best model to {SAVE_PATH}")

print("Training complete.")
