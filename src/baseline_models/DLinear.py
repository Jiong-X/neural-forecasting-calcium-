"""
DLinear — Decomposition-Linear baseline for neural activity forecasting.

Architecture:
  1. Decompose each neuron's context window into:
       - Trend    : moving-average filter (kernel = context_len//4 * 2 + 1)
       - Seasonal : residual  (original - trend)
  2. Apply a separate linear layer (context_len → pred_len) to each component
  3. Sum the two outputs to produce the forecast

Compared to NLinear, DLinear explicitly separates slow drift (trend) from
fast oscillations (seasonal) — both present in calcium imaging traces.

Reference: Zeng et al. (2023) "Are Transformers Effective for Time Series
           Forecasting?" AAAI 2023.

Run via run_benchmark.py

"""

import torch
import torch.nn as nn
from src.metrics import Prediction

# ---------------------------------------------------------------------------
# Model — self-contained DLinear (no external dependencies)
# ---------------------------------------------------------------------------

class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):           # x: (B, L, N)
        pad = self.kernel_size // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end   = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """
    Decomposition-Linear (Zeng et al. 2023).
    Splits the context into trend (moving average) + seasonal (residual),
    applies separate linear projections, sums the outputs.
    Input/output: (L, B, N)
    """
    def __init__(self, context_len: int, pred_len: int, n_channels: int,
                 individual: bool = True):
        super().__init__()
        kernel = context_len // 4 * 2 + 1
        self.decomp     = _MovingAvg(kernel)
        self.pred_len   = pred_len
        self.individual = individual
        self.channels   = n_channels

        init_w = (1 / context_len) * torch.ones(pred_len, context_len)
        if individual:
            self.lin_s = nn.ModuleList([nn.Linear(context_len, pred_len)
                                        for _ in range(n_channels)])
            self.lin_t = nn.ModuleList([nn.Linear(context_len, pred_len)
                                        for _ in range(n_channels)])
            for l in self.lin_s + self.lin_t:
                l.weight = nn.Parameter(init_w.clone())
        else:
            self.lin_s = nn.Linear(context_len, pred_len)
            self.lin_t = nn.Linear(context_len, pred_len)
            self.lin_s.weight = nn.Parameter(init_w.clone())
            self.lin_t.weight = nn.Parameter(init_w.clone())

    def forward(self, x):      # x comes in as (B, context_len, N)
        trend    = self.decomp(x)
        seasonal = x - trend
        if self.individual:
            s_out = torch.stack([self.lin_s[i](seasonal[:, :, i])
                                 for i in range(self.channels)], dim=-1)
            t_out = torch.stack([self.lin_t[i](trend[:, :, i])
                                 for i in range(self.channels)], dim=-1)
        else:
            s_out = self.lin_s(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            t_out = self.lin_t(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out = s_out + t_out         # (B, pred_len, N)
        return Prediction(mean=out)
        
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
    model = DLinear(CONTEXT_LEN, PRED_LEN, N, individual=False).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()
"""