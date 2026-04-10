"""
TexFilter — Frequency-domain filter baseline for neural activity forecasting.

Architecture (from the POCO paper / POCO repository):
  1. RevIN  : per-neuron instance normalisation (learnable affine rescale)
  2. Embed  : linear projection of the context window per neuron → embed_size
  3. LayerNorm
  4. FFT    : real FFT along the neuron axis → complex spectrum
  5. TexFilter : two-layer complex MLP applied in the frequency domain
                 (learnable weights w, w1 + biases rb/ib, sparse via softshrink)
  6. IFFT   : back to spatial domain
  7. LayerNorm + Dropout + FC
  8. Output : linear projection embed_size → pred_len per neuron
  9. RevIN denorm

Source: POCO repo — models/multi_session_models.py  (class TexFilter)
        models/layers/normalizer.py                 (class RevIN)

Run:
  python3 TexFilter.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.util import fetch_data_loaders

# ---------------------------------------------------------------------------
# RevIN  (copied verbatim from POCO/models/layers/normalizer.py)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalisation (Kim et al. 2022)."""

    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features  = num_features
        self.eps           = eps
        self.affine        = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unknown mode: {mode}")

    def _get_statistics(self, x):
        dims = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = x.mean(dim=dims, keepdim=True).detach()
        self.stdev = (x.var(dim=dims, keepdim=True, unbiased=False) + self.eps).sqrt().detach()

    def _normalize(self, x):
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


# ---------------------------------------------------------------------------
# TexFilter  (adapted from POCO/models/multi_session_models.py)
# ---------------------------------------------------------------------------

class TexFilter(nn.Module):
    """
    Frequency-domain filter baseline from the POCO paper.

    Input / output convention (matches all other models here):
        forward(x_list) where x_list = [(context_len, B, N)]
        returns         pred_list   = [(pred_len,    B, N)]
    """

    SPARSITY_THRESHOLD = 0.01
    SCALE              = 0.02

    def __init__(self, n_channels: int, context_len: int, pred_len: int,
                 embed_size: int = 64, hidden_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.context_len = context_len
        self.pred_len    = pred_len
        self.embed_size  = embed_size

        # Per-session instance normalisation
        self.revin = RevIN(n_channels, affine=True, subtract_last=False)

        # Time-axis embedding: (B, N, context_len) → (B, N, embed_size)
        self.embedding = nn.Linear(context_len, embed_size)

        # Complex filter weights  (two-layer, applied in frequency domain)
        s = self.SCALE
        self.w  = nn.Parameter(s * torch.randn(2, embed_size))
        self.w1 = nn.Parameter(s * torch.randn(2, embed_size))
        self.rb1 = nn.Parameter(s * torch.randn(embed_size))
        self.ib1 = nn.Parameter(s * torch.randn(embed_size))
        self.rb2 = nn.Parameter(s * torch.randn(embed_size))
        self.ib2 = nn.Parameter(s * torch.randn(embed_size))

        self.norm1   = nn.LayerNorm(embed_size)
        self.norm2   = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, embed_size),
        )
        self.output = nn.Linear(embed_size, pred_len)

    # ------------------------------------------------------------------
    # Complex frequency-domain filter  (texfilter in the POCO repo)
    # ------------------------------------------------------------------
    def _texfilter(self, x):
        """x: complex tensor (B, N//2+1, embed_size)"""
        o1_real = F.relu(
            torch.einsum("bid,d->bid", x.real, self.w[0]) -
            torch.einsum("bid,d->bid", x.imag, self.w[1]) + self.rb1
        )
        o1_imag = F.relu(
            torch.einsum("bid,d->bid", x.imag, self.w[0]) +
            torch.einsum("bid,d->bid", x.real, self.w[1]) + self.ib1
        )
        o2_real = (
            torch.einsum("bid,d->bid", o1_real, self.w1[0]) -
            torch.einsum("bid,d->bid", o1_imag, self.w1[1]) + self.rb2
        )
        o2_imag = (
            torch.einsum("bid,d->bid", o1_imag, self.w1[0]) +
            torch.einsum("bid,d->bid", o1_real, self.w1[1]) + self.ib2
        )
        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.SPARSITY_THRESHOLD)
        return torch.view_as_complex(y)

    # ------------------------------------------------------------------
    def forward(self, x_list):
        """
        x_list : list of one (context_len, B, N) tensor
        returns : list of one (pred_len,    B, N) tensor
        """
        x = x_list[0].permute(1, 0, 2)   # → (B, context_len, N)
        B, L, N = x.shape

        x = self.revin(x, "norm")         # instance normalise
        x = x.permute(0, 2, 1)           # → (B, N, context_len)
        x = self.norm1(self.embedding(x)) # → (B, N, embed_size)

        # Frequency domain
        xf     = torch.fft.rfft(x, dim=1, norm="ortho")   # (B, N//2+1, embed_size)
        weight = self._texfilter(xf)
        x      = torch.fft.irfft(xf * weight, n=N, dim=1, norm="ortho")

        x = self.norm2(x)
        x = self.dropout(x)
        x = self.fc(x)                    # (B, N, embed_size)
        x = self.output(x)                # (B, N, pred_len)
        x = x.permute(0, 2, 1)           # → (B, pred_len, N)
        x = self.revin(x, "denorm")
        x = x.permute(1, 0, 2)           # → (pred_len, B, N)
        return [x]




# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device):
    model.train()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimiser.zero_grad()
        pred = model([X])[0]             # (pred_len, B, N)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total += loss.item() * X.shape[1]
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    for X, Y in loader:
        X, Y  = X.to(device), Y.to(device)
        pred  = model([X])[0]
        total_mse += F.mse_loss(pred, Y).item() * X.shape[1]
        total_mae += (pred - Y).abs().mean().item() * X.shape[1]
        n += X.shape[1]
    return total_mse / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_texfilter.pt"
    RESULTS_PATH = "results/texfilter_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Hyperparameters (from POCO repo config) ---
    N_PCS      = 128
    SEQ_LEN    = 64     # context (48) + horizon (16) — matches paper (C=48, P=16)
    PRED_LEN   = 16
    EMBED_SIZE = 128    # filter_embed_size in POCO repo
    HIDDEN     = 512    # hidden_size in POCO repo
    DROPOUT    = 0.3    # dropout in POCO repo
    BATCH_SIZE = 64
    EPOCHS     = 50
    LR         = 3e-4
    TRAIN_FRAC = 0.6
    VAL_FRAC   = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    CONTEXT_LEN  = SEQ_LEN - PRED_LEN

    train_loader, val_loader, N = fetch_data_loaders("TexFilter",SEQ_LEN, PRED_LEN, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE)
    
    
    # --- Model ---
    model = TexFilter(N, CONTEXT_LEN, PRED_LEN,
                      embed_size=EMBED_SIZE, hidden_size=HIDDEN,
                      dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    # --- Training ---
    train_mses, val_mses, val_maes = [], [], []
    best_mse = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_mse        = train_epoch(model, train_loader, optimiser, device)
        val_mse, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step(val_mse)

        train_mses.append(train_mse)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

        tag = " *" if val_mse < best_mse else ""
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train={train_mse:.4f}  "
              f"val_mse={val_mse:.4f}  "
              f"val_mae={val_mae:.4f}{tag}")

    print(f"\nBest val MSE: {best_mse:.4f}  — saved to {MODEL_PATH}")
    np.savez(RESULTS_PATH,
             train_mses=train_mses, val_mses=val_mses, val_maes=val_maes)
    print(f"Loss history saved to {RESULTS_PATH}")
