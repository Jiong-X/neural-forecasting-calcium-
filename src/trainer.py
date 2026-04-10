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
        pred  = model(X)
        _, cur_scores = criterion(pred, Y)
        score.update(cur_scores)
    return score.get_scores()

def test_eval(model, save_path:str, test_loader, device):
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    test_scores = eval_epoch(model, test_loader, device)
    print(test_scores)

def training_loop(model, config:trainingConfig, train_loader, val_loader, optimiser, scheduler, criterion):
    
    best_val_loss  = float("inf")
    no_improve     = 0
    scores = ScoreTracker(criterion)
    scores.print_headline()
    print("-" * 44)

    for epoch in range(1, config.epochs + 1):
        cur_train_score = train_epoch(model, train_loader, optimiser, config.device)
        cur_val_scores = eval_epoch(model, val_loader, config.device)
        scores.update(cur_train_score, "train")
        scores.update(cur_val_scores, "val")
        val_loss = cur_val_scores.get_metric(criterion.monitor_name)
        scheduler.step(val_loss)

        tag = " *" if val_loss < best_val_loss else ""
        scores.print_latest(epoch, tag)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), config.save_path)
        else:
            no_improve += 1

        if no_improve >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.patience} epochs)")
            break

    print(f"\nBest val NLL: {best_val_loss:.4f}  — saved to {config.save_path}")
    return scores

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

