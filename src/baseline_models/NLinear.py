"""
NLinear — Normalisation-Linear baseline for neural activity forecasting.

Architecture:
  1. Subtract the last observed value (removes level shift / non-stationarity)
  2. Apply a single shared linear layer: context_len → pred_len  (per neuron)
  3. Add the last observed value back

This is the simplest possible learned forecaster — one scalar weight per
(context step, forecast step) pair, shared across all neurons.

Reference: Zeng et al. (2023) "Are Transformers Effective for Time Series
           Forecasting?" AAAI 2023.

Run with:
  /home/jiongx/micromamba/envs/comp0197-pt/bin/python3 NLinear.py
"""

import torch
import torch.nn as nn
from src.metrics import Prediction
# ---------------------------------------------------------------------------
# Model — self-contained NLinear (no external dependencies)
# ---------------------------------------------------------------------------

class NLinear(nn.Module):
    """
    Normalisation-Linear (Zeng et al. 2023).
    Subtracts the last observed value, applies a shared linear
    projection context_len → pred_len, then adds the last value back.
    Input/output: (L, B, N)
    """
    def __init__(self, context_len: int, pred_len: int, n_channels: int,
                 individual: bool = False):
        super().__init__()
        self.pred_len   = pred_len
        self.individual = individual
        self.channels   = n_channels
        if individual:
            self.linear = nn.ModuleList(
                [nn.Linear(context_len, pred_len) for _ in range(n_channels)])
        else:
            self.linear = nn.Linear(context_len, pred_len)

    def forward(self, x):            # x comes in as (B, context_len, N)
        last = x[:, -1:, :].detach()
        x = x - last
        if self.individual:
            out = torch.stack(
                [self.linear[i](x[:, :, i]) for i in range(self.channels)],
                dim=-1)             # (B, pred_len, N)
        else:
            out = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + last
        return Prediction(mean=out)  # → (pred_len, B, N)

"""
# ---------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------- 

    N_PCS      = 128
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16
    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-3
    VAL_FRAC   = 0.2
    TRAIN_FRAC = 0.6
    
    # --- Model ---
    model = NLinear(CONTEXT_LEN, PRED_LEN, N, individual=False).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()
"""