# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.uncertainty import RESULTS_PATH
from src.dataset import get_splits
from torch.utils.data import DataLoader

from train import SAVE_PATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
from metrics import Score, _MetricBase, ScoreTracker
from util import fetch_data_loaders, _DATASET_CACHE, trainingConfig

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

def test_eval(model, save_path:str, test_loader, device):
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    test_scores = eval_epoch(model, test_loader, device)
    print(test_scores)

def training_loop(model, train_loader, val_loader, optimiser, scheduler, criterion, save_path:str, patience:int, device):
    
    best_val_loss  = float("inf")
    no_improve     = 0
    train_scores = ScoreTracker(criterion)
    val_scores = ScoreTracker(criterion)
    print(f"\n{'Epoch':>6}  {'Train NLL':>10}  {'Val NLL':>10}  {'Val MAE':>9}")
    print("-" * 44)

    for epoch in range(1, EPOCHS + 1):
        cur_train_score = train_epoch(model, train_loader, optimiser, device)
        cur_val_scores   = eval_epoch(model, val_loader, device)
        train_scores.update(cur_train_score)
        val_scores.update(cur_val_scores)
        scheduler.step(val_loss)

        tag = " *" if val_loss < best_val_loss else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {val_mae:>9.4f}{tag}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nBest val NLL: {best_val_loss:.4f}  — saved to {save_path}")
    return train_scores, val_scores

def train(config:trainingConfig):
    print(f"\n{"="*40}\nTraining model: {config.model_name}")

    # ── Reproducibility ───────────────────────────────────────────────────────────
    if (seed := config.seed) is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # force deterministic CUDA kernels
        torch.backends.cudnn.benchmark = False  # disable auto-tuner (picks same algo each run)

    print(f"Device: {config.device}")

    # Data — auto-retrieved ─────────────────────────────────────────────────────────
    print("Loading data (downloads automatically if not present)...")

    train_dataset, val_dataset, test_dataset = get_splits(
    seq_length  = config.sequence_length, pred_length = config.pred_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size,shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size,shuffle=False, num_workers=0)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.results_path), exist_ok=True)


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
