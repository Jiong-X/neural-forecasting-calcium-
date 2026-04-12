# GEN AI STATEMENT
# this script was fully written by a human, training was standardised to enable easy and guaranteed benchmarking to baseline models

# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.dataset import get_splits
from src.metrics import Score, _MetricBase, ScoreTracker
from src.util import trainingConfig

def train_epoch(model, loader, optimiser, criterion:_MetricBase, device):
    model.train()
    score = Score.create(criterion)
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
    score = Score.create(criterion)
    for X, Y in loader:
        X, Y  = X.to(device), Y.to(device)
        pred  = model(X)
        _, cur_scores = criterion(pred, Y)
        score.update(cur_scores)
    return score.get_scores()

def training_loop(model, config:trainingConfig, train_loader, val_loader, optimiser, scheduler, criterion):
    
    best_val_loss  = float("inf")
    no_improve = 0
    scores = ScoreTracker.create(criterion)
    scores.print_headline()

    for epoch in range(1, config.epochs + 1):
        cur_train_score = train_epoch(model, train_loader, optimiser, criterion,config.device)
        cur_val_scores = eval_epoch(model, val_loader, criterion, config.device)
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

    print(f"\nBest val {criterion.monitor_name}: {best_val_loss:.4f}  — saved to {config.save_path}")
    return scores

def train(model, config:trainingConfig, optimiser, criterion):
    print(f"\n{'='*40}\nTraining model: {config.model_name}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Reproducibility ───────────────────────────────────────────────────────────

    """
    the following has been moved to the util.py function. It has to be initialised in the config before the model is created, otherwise the initial weights of the model will not be deterministic across runs, 
    which breaks benchmarking. By moving it to the util.py function, we can guarantee that all models are initialised with the same seed and are therefore directly comparable.
    
    if (seed := config.seed) is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # force deterministic CUDA kernels
        torch.backends.cudnn.benchmark = False  # disable auto-tuner (picks same algo each run)
    """
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
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5
)
    scores = training_loop(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        criterion=criterion
    )

    # ---------------------------------------------------------------------------
    # Test evaluation — run once on held-out set using best checkpoint
    # ---------------------------------------------------------------------------
    model.load_state_dict(torch.load(config.save_path, map_location=config.device, weights_only=True))

    test_scores = eval_epoch(model, test_loader, criterion, config.device)
    print(f"Test  | {test_scores}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------

    scores.update(test_scores, "test")
    np.savez(config.results_path, **scores.to_save_dict())

    print(f"Loss curves saved to {config.results_path}")
    print("Training complete.")