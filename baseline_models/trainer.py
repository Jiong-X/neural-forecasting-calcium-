# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
from metrics import Score, _MetricBase
from util import fetch_data_loaders, _DATASET_CACHE

def train_epoch(model, loader, optimiser, criterion:_MetricBase, device):
    model.train()
    score = Score(criterion)
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimiser.zero_grad()
        pred = model(X)                         # (pred_len, B, N)
        loss, cur_scores = criterion(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        score.update(cur_scores)
    return score.get_scores()

@torch.no_grad()
def eval_epoch(model, loader, criterion:_MetricBase, device):
    model.eval()
    score = Score(criterion)
    for X, Y in loader:
        X, Y  = X.to(device), Y.to(device)
        pred  = model([X])[0]
        _, cur_scores = criterion(pred, Y)
        score.update(cur_scores)
    return score.get_scores()

def train(model_type:str, seed:Union[int, None]=None):

    print(f"\n{"="*40}\nTraining model: {model_type}")

    if model_type not in _DATASET_CACHE.keys():
        print(f"invalid model type '{model_type}', must be one of '{_DATASET_CACHE.keys()}'")
        return

    if seed:
        if type(seed)!= int:
            raise TypeError(f"seed must either be 'None' or an 'int', got: {seed}")
        torch.manual_seed(42)
        np.random.seed(42)
    
    MODEL_PATH   = f"models/best_{model_type}.pt"
    RESULTS_PATH = f"results/{model_type}_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    N_PCS      = 128
    CONTEXT    = 48   # context (48) — matches paper (C=48, P=16)
    PRED_LEN   = 16   # Prediction (16), for RNN/ LSTM Change to 1
    # PRED_LEN = 1
    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-3
    VAL_FRAC   = 0.2
    TRAIN_FRAC = 0.6

    
    # --- Data ---

    train_loader, val_loader, N = fetch_data_loaders("DLinear",CONTEXT, PRED_LEN, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE)
    
    # --- Model ---
    model = DLinear(CONTEXT, PRED_LEN, N, individual=False).to(device)
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
