"""
Vanilla LSTM for predicting neural activity from calcium imaging data.
Uses nn.LSTM with a single linear output head — no encoder-decoder,
no teacher forcing.
"""

import torch
import torch.nn as nn

from src.metrics import Prediction

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CalciumLSTM(nn.Module):
    """
    Vanilla multi-layer LSTM for calcium imaging forecasting.
    Horizon is dynamic — rolled forward autoregressively at inference time.
    """

    def __init__(
        self,
        n_neurons: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        default_pred_steps: int = 1,
    ):
        super().__init__()
        self.default_pred_steps = default_pred_steps
        self.n_neurons = n_neurons

        self.lstm = nn.LSTM(
            input_size=n_neurons,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, n_neurons)

    def forward(self, x: torch.Tensor, pred_steps: int | None = None) -> torch.Tensor:
        """
        Args:
            x:          (B, T, N) 
            pred_steps: forecast horizon; defaults to self.default_pred_steps
        Returns:
            (B, pred_steps, N)
        """
        if pred_steps is None:
            pred_steps = self.default_pred_steps

        _, (h, c) = self.lstm(x)

        preds = []
        inp = x[:, -1:, :]                          # (B, 1, N)
        for _ in range(pred_steps):
            out, (h, c) = self.lstm(inp, (h, c))    # (B, 1, H)
            step = self.head(out)                   # (B, 1, N)
            preds.append(step)
            inp = step.detach()

        mean_out = torch.cat(preds, dim=1)  # (B, pred_steps, N)

        return Prediction(mean=mean_out)      

"""
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

    N_PCS      = None
    SEQ_LEN    = 64   # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_STEPS = 16
    HIDDEN     = 256
    LAYERS     = 2
    DROPOUT    = 0.2
    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-3
    VAL_FRAC   = 0.2
    TRAIN_FRAC = 0.6
    
    # --- Model ---
    model = CalciumLSTM(n_neurons=N, hidden_size=HIDDEN, num_layers=LAYERS,dropout=DROPOUT, default_pred_steps=PRED_STEPS,).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
"""