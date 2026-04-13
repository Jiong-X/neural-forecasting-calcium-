# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

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

Run via run_benchmark.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import Prediction

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
    def forward(self, x):
        """
        x_list : one (B, context_len, N) tensor 
        returns : one (pred_len, B, N) tensor
        """
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
        return Prediction(mean=x)

"""
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

    # --- Model ---
    model = TexFilter(N, CONTEXT_LEN, PRED_LEN,embed_size=EMBED_SIZE, hidden_size=HIDDEN, dropout=DROPOUT).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
"""