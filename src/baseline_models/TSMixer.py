"""
TSMixer — Time-Series Mixer baseline for neural activity forecasting.

Architecture:
  Alternating MLP blocks applied along two axes:
    - Temporal mixing  : MLP across the time axis  (shared across neurons)
    - Feature mixing   : MLP across the neuron axis (shared across time steps)
  Each block uses LayerNorm + residual connection.

Intuition for calcium imaging:
  - Temporal MLP captures dynamics within each neuron's trace.
  - Feature MLP captures cross-neuron (population) interactions.

Reference: Chen et al. (2023) "TSMixer: An All-MLP Architecture for
           Time Series Forecasting." arXiv:2303.06053.

Run with:
  /home/jiongx/micromamba/envs/comp0197-pt/bin/python3 TSMixer.py
"""

import torch.nn as nn
import torch.nn.functional as F

from src.metrics import Prediction

# ---------------------------------------------------------------------------
# Model — self-contained TSMixer (no external dependencies)
# ---------------------------------------------------------------------------

class _TimeMixBlock(nn.Module):
    """MLP along the time axis, shared across channels."""
    def __init__(self, seq_len: int, ff_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.fc1  = nn.Linear(seq_len, ff_dim)
        self.fc2  = nn.Linear(ff_dim, seq_len)

    def forward(self, x):           # x: (B, L, N)
        r = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)   # LN over time
        x = x.transpose(1, 2)                              # (B, N, L)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).transpose(1, 2)                    # (B, L, N)
        return x + r


class _FeatMixBlock(nn.Module):
    """MLP along the channel axis, shared across time steps."""
    def __init__(self, n_channels: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(n_channels)
        self.fc1     = nn.Linear(n_channels, ff_dim)
        self.fc2     = nn.Linear(ff_dim, n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):           # x: (B, L, N)
        r = x
        x = self.norm(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x + r

class TSMixer(nn.Module):
    """
    TSMixer (Chen et al. 2023) — alternating temporal and feature MLP mixing.
    Input/output: (L, B, N)
    """
    def __init__(self, context_len: int, pred_len: int, n_channels: int,
                 ff_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(_TimeMixBlock(context_len, ff_dim))
            self.blocks.append(_FeatMixBlock(n_channels,  ff_dim, dropout))
        self.head = nn.Linear(context_len, pred_len)

    def forward(self, x):           # x: (B, context_len, N)
        for blk in self.blocks:
            x = blk(x)
        # project time axis: (B, N, L) → (B, N, pred_len) → (B, pred_len, N)
        out = self.head(x.permute(0, 2, 1)).permute(0, 2, 1)
        return Prediction(mean=out)

"""
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

    N_PCS      = 128
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16
    FF_DIM     = 64   # hidden dim of MLP mixing blocks (paper default)
    N_LAYERS   = 2    # number of mixer blocks (paper default)
    DROPOUT    = 0.1  # paper default
    BATCH_SIZE = 64   # paper default
    EPOCHS     = 50
    LR         = 3e-4        # paper default
    WEIGHT_DECAY = 1e-4      # paper default
    GRAD_CLIP    = 5.0       # paper default
    VAL_FRAC   = 0.2
    TRAIN_FRAC = 0.6

    # --- Model ---    
    model = TSMixer(CONTEXT_LEN, PRED_LEN, N, ff_dim=FF_DIM, n_layers=N_LAYERS, dropout=DROPOUT).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()

"""