"""
MLP-only ablation of probabilistic POCO for predicting neural activity from
calcium imaging data.

Architecture:
  - Per-neuron MLP: in_proj (Linear + ReLU) → mu_proj + log_sig_proj (Linear)
  - Identical to POCO_prob's conditioning head, but without the POYO encoder.
  - Without the Perceiver-IO embedding, the FiLM alpha/beta terms collapse to
    zero (their initialisation in POCO), leaving:
        h       = in_proj(x)
        mu      = mu_proj(h)
        log_sig = log_sig_proj(h)
  - Trained with Gaussian NLL, same as POCO_prob.

Dimensions match POCO_prob exactly (COND_DIM=1024, PRED_LEN=16, context_len=48)
so any performance gap is attributable solely to the POYO encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import Prediction

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Per-neuron probabilistic MLP — POCO_prob's conditioning head without the
    POYO encoder.

    Each neuron's context window is projected independently:
        x (B, context_len, N) → in_proj → weights (B, N, cond_dim)
            → mu_proj      → mu      (B, pred_len, N)
            → log_sig_proj → logvar  (B, pred_len, N)

    Note: no FiLM conditioning — weights = in_proj(inp) only, no alpha/beta.
    """

    LOG_SIG_MIN = -6.0   # clamp raw log_sig to avoid underflow
    LOG_SIG_MAX =  2.0   # clamp raw log_sig to avoid instability
    name:str = "MLP"

    def __init__(self, n_neurons: int, context_len: int, cond_dim: int, pred_len: int):
        super().__init__()
        self.in_proj      = nn.Sequential(nn.Linear(context_len, cond_dim), nn.ReLU())
        self.mu_proj      = nn.Linear(cond_dim, pred_len)
        self.log_sig_proj = nn.Linear(cond_dim, pred_len)

        # Initialise log_sig bias to predict ~0.5 std initially — matches POCO_prob
        nn.init.constant_(self.log_sig_proj.bias, -0.69)
        nn.init.zeros_(self.log_sig_proj.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, context_len, N)
        Returns:
            mu:     (B, pred_len, N)
            logvar: (B, pred_len, N)
        """
        inp     = x.transpose(1, 2)                                        # (B, N, context_len)
        weights = self.in_proj(inp)                                        # (B, N, cond_dim) — no FiLM conditioning
        mu      = self.mu_proj(weights)                                    # (B, N, pred_len)
        log_sig = self.log_sig_proj(weights).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        sigma   = F.softplus(log_sig) + 1e-4

        mu     = mu.transpose(1, 2)                                        # (B, pred_len, N)
        logvar = (2 * sigma.log()).transpose(1, 2)                         # (B, pred_len, N)
        return Prediction(mean=mu, logvar=logvar)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

"""
if __name__ == "__main__":

    # Hyperparameters — match POCO_prob exactly for a fair ablation
    N_PCS      = 128    # cap at 128 PCs, same as POCO_prob
    CONTEXT    = 48     # input steps — matches paper (C=48)
    PRED_LEN   = 16     # forecast horizon — matches paper (P=16)
    COND_DIM   = 1024   # MLP conditioning dimension — matches paper
    BATCH_SIZE = 64     # matches POCO_prob
    EPOCHS     = 50
    LR         = 3e-4
    TRAIN_FRAC = 0.6    # 3:1:1 split — matches POCO_prob / paper
    VAL_FRAC   = 0.2
    PATIENCE   = 10


    T, N   = traces.shape

    train_end = int(T * TRAIN_FRAC)
    val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

    model = MLPHead(
        n_neurons=N, context_len=CONTEXT, cond_dim=COND_DIM, pred_len=PRED_LEN,
    ).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
"""