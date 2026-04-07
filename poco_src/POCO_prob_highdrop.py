"""
Probabilistic POCO with higher attention dropout for better epistemic uncertainty.

Same architecture as POCO_prob.py but patches atn_dropout after model creation
to 0.3 (vs default 0.0), giving MC Dropout more variance across forward passes.
"""

import sys
import os

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader

from POCO_prob import ProbabilisticPOCO, CalciumDataset, nll_loss
from standalone_poco import NeuralPredictionConfig, RotaryCrossAttention, RotarySelfAttention


def set_attention_dropout(model: nn.Module, atn_dropout: float):
    """
    Patch the dropout float stored on every RotaryCrossAttention and
    RotarySelfAttention module in the model.  These classes store dropout
    as a plain float (self.dropout) used directly in the forward pass,
    so we can update it without rebuilding the model.
    """
    patched = 0
    for module in model.modules():
        if isinstance(module, (RotaryCrossAttention, RotarySelfAttention)):
            module.dropout = atn_dropout
            patched += 1
    print(f"Patched attention dropout → {atn_dropout} on {patched} modules")


def train_epoch(model, loader, optimiser, device):
    model.train()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        x_list = [X.permute(1, 0, 2)]
        y_list = [Y.permute(1, 0, 2)]
        optimiser.zero_grad()
        dists = model(x_list)
        loss  = nll_loss(dists, y_list)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_nll, total_mae, n = 0.0, 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        x_list = [X.permute(1, 0, 2)]
        y_list = [Y.permute(1, 0, 2)]
        dists = model(x_list)
        total_nll += nll_loss(dists, y_list).item() * len(X)
        pred_mu = dists[0].mean
        total_mae += (pred_mu - y_list[0]).abs().mean().item() * len(X)
        n += len(X)
    return total_nll / n, total_mae / n


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_poco_prob_highdrop.pt"
    RESULTS_PATH = "results/poco_prob_highdrop_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    N_PCS      = 128
    SEQ_LEN     = 64    # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN    = 16    # matches paper main experiments
    CONTEXT_LEN = SEQ_LEN - PRED_LEN
    BATCH_SIZE = 16
    EPOCHS     = 50
    LR         = 3e-4
    VAL_FRAC   = 0.2

    # Dropout settings — only lin_dropout is safely patchable
    # (atn_dropout requires disabling xformers which breaks seqlen masking)
    ATN_DROPOUT = 0.0   # leave untouched
    FFN_DROPOUT = 0.3   # was 0.2
    LIN_DROPOUT = 0.6   # was 0.4 — main lever for MC Dropout epistemic spread

    NUM_LATENTS = 8
    HIDDEN_DIM  = 64
    COND_DIM    = 128
    NUM_LAYERS  = 1
    NUM_HEADS   = 8
    COMPRESSION = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dropout — atn: {ATN_DROPOUT}  ffn: {FFN_DROPOUT}  lin: {LIN_DROPOUT}")

    # --- Data ---
    data = np.load(DATA_PATH)
    raw  = data["PC"].astype(np.float32)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    traces = raw[:, :N_PCS]
    T, N   = traces.shape

    split        = int(T * (1 - VAL_FRAC))
    train_ds     = CalciumDataset(traces[:split], context_len=CONTEXT_LEN, pred_len=PRED_LEN)
    val_ds       = CalciumDataset(traces[split:], context_len=CONTEXT_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # --- Model ---
    config = NeuralPredictionConfig()
    config.seq_length             = SEQ_LEN
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

    model = ProbabilisticPOCO(config, [[N]]).to(device)

    # Patch lin_dropout (nn.Dropout on perceiver outputs) to higher value.
    # atn_dropout is left at 0.0 — changing it requires disabling xformers
    # which breaks variable-length sequence masking.
    lin_patched = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if abs(module.p - 0.4) < 1e-6:
                module.p = LIN_DROPOUT
                lin_patched += 1
    print(f"Patched lin dropout → {LIN_DROPOUT} on {lin_patched} modules")

    # Patch FFN dropout (stored as nn.Dropout modules)
    ffn_patched = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if abs(module.p - 0.2) < 1e-6:   # only patch FFN dropout (0.2)
                module.p = FFN_DROPOUT
                ffn_patched += 1
    print(f"Patched FFN dropout → {FFN_DROPOUT} on {ffn_patched} modules")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    train_nlls, val_nlls, val_maes = [], [], []
    best_nll = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_nll        = train_epoch(model, train_loader, optimiser, device)
        val_nll, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step(val_nll)
        train_nlls.append(train_nll)
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
    np.savez(RESULTS_PATH, train_nlls=train_nlls, val_nlls=val_nlls, val_maes=val_maes)
    print(f"Saved to {RESULTS_PATH}")
