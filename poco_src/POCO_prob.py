"""
Probabilistic POCO — Population-Conditioned Forecaster with distributional output.

Subclasses the standalone POCO from the POCO repo.  The only architectural
change is replacing the deterministic output projection with two heads that
parameterise a per-neuron, per-step Gaussian:

    mu      (B, D, pred_length)   — predicted mean
    log_sig (B, D, pred_length)   — log standard deviation (unconstrained)

The forward pass returns a torch.distributions.Normal object so callers can
draw samples, compute log-probabilities, or read off mu/sigma directly.
Training uses the closed-form Gaussian NLL, which the helper `nll_loss` below
provides.
"""

import sys
import os

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

from poco_src.standalone_poco import POCO, NeuralPredictionConfig


# ---------------------------------------------------------------------------
# Probabilistic head
# ---------------------------------------------------------------------------

class ProbabilisticPOCO(POCO):
    """
    POCO with a Gaussian output head.

    Replaces the single out_proj (→ pred_length values per neuron) with:
        mu_proj      → pred_length means
        log_sig_proj → pred_length log-standard-deviations

    forward() returns torch.distributions.Normal(mu, sigma) where
    sigma = softplus(log_sig) + 1e-4  (always positive, no numerical issues).
    """

    LOG_SIG_MIN = -6.0   # clamp raw log_sig to avoid underflow
    LOG_SIG_MAX =  2.0   # clamp raw log_sig to avoid instability

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, input_size)

        # Replace the deterministic head with two heads of identical shape.
        cond_dim       = config.conditioning_dim
        pred_length    = config.pred_length

        # Remove the original head so its parameters are not optimised for
        # point-estimate loss by mistake.
        del self.out_proj

        self.mu_proj      = nn.Linear(cond_dim, pred_length)
        self.log_sig_proj = nn.Linear(cond_dim, pred_length)

        # FiLM conditioning weights are already zero-initialised inside POCO.__init__
        # (standalone_poco.py lines 838-841) — no manual init needed.

        # Initialise log_sig bias to predict ~0.5 std initially (log(0.5) ≈ -0.69)
        nn.init.constant_(self.log_sig_proj.bias, -0.69)
        nn.init.zeros_(self.log_sig_proj.weight)

    # ------------------------------------------------------------------
    # Override only the final projection inside forward()
    # ------------------------------------------------------------------

    def forward(self, x_list, unit_indices=None, unit_timestamps=None):
        """
        Args:
            x_list : list of (L, B, D) tensors — one per session.

        Returns:
            dist   : list of torch.distributions.Normal, one per session.
                     Each distribution has event shape (pred_length, B, D).
                     Use dist.mean for point predictions, dist.sample() for
                     stochastic rollouts, dist.log_prob(y) for NLL.
        """
        import itertools
        import torch

        bsz = [x.size(1) for x in x_list]
        L   = x_list[0].size(0)

        # ---- replicate POCO.forward up to the embedding step ----
        x = torch.concatenate(
            [x.permute(1, 2, 0).reshape(-1, L) for x in x_list], dim=0
        )  # sum(B*D), L

        if L != self.Tin:
            x = x[:, -self.Tin:]

        # tokenise (no tokenizer by default → simple reshape)
        out = x.reshape(x.shape[0], self.Tin // self.T_step, self.T_step)

        d_list = self.input_size

        if unit_indices is None:
            sum_channels = 0
            unit_indices = []
            for b, d in zip(bsz, self.input_size):
                indices = (
                    torch.arange(d, device=x.device)
                    .unsqueeze(0).repeat(b, 1).reshape(-1)
                )
                unit_indices.append(indices + sum_channels)
                sum_channels += d
            unit_indices = torch.cat(unit_indices, dim=0)

        if unit_timestamps is None:
            unit_timestamps = (
                torch.zeros_like(unit_indices).unsqueeze(1)
                + torch.arange(0, self.Tin, self.T_step, device=x.device)
            )

        input_seqlen = torch.cat(
            [torch.full((b,), d, device=x.device) for b, d in zip(bsz, self.input_size)], dim=0
        )
        session_index = torch.cat(
            [torch.full((b,), i, device=x.device) for i, b in enumerate(bsz)], dim=0
        )
        dataset_index = torch.cat(
            [torch.full((b,), self.dataset_idx[i], device=x.device) for i, b in enumerate(bsz)], dim=0
        )

        embed = self.decoder(
            out,
            unit_indices=unit_indices,
            unit_timestamps=unit_timestamps,
            input_seqlen=input_seqlen,
            session_index=session_index,
            dataset_index=dataset_index,
        )  # sum(B*D), embedding_dim

        split_size = [b * d for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0)
        embed = [xx.reshape(b, d, self.embedding_dim)
                 for xx, b, d in zip(embed, bsz, d_list)]  # (B, D, E)

        # ---- probabilistic head ----
        dists = []
        for e, d, x_in in zip(embed, self.input_size, x_list):
            alpha   = self.conditioning_alpha(e)       # B, D, cond_dim
            beta    = self.conditioning_beta(e)        # B, D, cond_dim
            inp     = x_in.permute(1, 2, 0)            # B, D, L
            weights = self.in_proj(inp) * alpha + beta # B, D, cond_dim

            mu      = self.mu_proj(weights)            # B, D, pred_length
            log_sig = self.log_sig_proj(weights)       # B, D, pred_length
            log_sig = log_sig.clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX)
            sigma   = torch.nn.functional.softplus(log_sig) + 1e-4

            # Return in (pred_length, B, D) to match POCO convention
            mu    = mu.permute(2, 0, 1)
            sigma = sigma.permute(2, 0, 1)
            dists.append(Normal(mu, sigma))

        return dists


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def nll_loss(dists: list[Normal], targets: list[torch.Tensor]) -> torch.Tensor:
    """
    Mean Gaussian NLL across all sessions.

    Args:
        dists   : list of Normal distributions, one per session.
        targets : list of (pred_length, B, D) tensors — ground truth.
    """
    total = sum(
        -dist.log_prob(y).mean() for dist, y in zip(dists, targets)
    )
    return total / len(dists)


# ---------------------------------------------------------------------------
# Dataset — sliding window over (T, N) traces
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    def __init__(self, traces: np.ndarray, context_len: int, pred_len: int):
        traces = traces.astype(np.float32)
        # expects already z-scored traces — normalisation done before splitting

        seq_len = context_len + pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])
        self.X = torch.tensor(np.array(X))   # (S, ctx, N)
        self.Y = torch.tensor(np.array(Y))   # (S, pred, N)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device):
    model.train()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        # POCO expects (L, B, D) — convert from (B, L, D)
        x_list = [X.permute(1, 0, 2)]
        y_list = [Y.permute(1, 0, 2)]

        optimiser.zero_grad()
        dists = model(x_list)
        loss  = nll_loss(dists, y_list)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # max norm 5 per paper
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
        total_mae += sum(
            (dist.mean - y).abs().mean().item() * len(X)
            for dist, y in zip(dists, y_list)
        ) / len(dists)
        n += len(X)
    return total_nll / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_poco_prob.pt"
    RESULTS_PATH = "results/poco_prob_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ---- Hyperparameters ----
    N_PCS      = 128    # features (top PCs); None = use all
    CONTEXT    = 48     # input steps — matches paper main experiments (C=48)
    PRED_LEN   = 16     # forecast horizon — matches paper main experiments (P=16)
    BATCH_SIZE = 64     # matches paper spec
    EPOCHS     = 50
    LR         = 3e-4
    # 3:1:1 train/val/test split — matches paper partitioning ratio
    TRAIN_FRAC = 0.6
    VAL_FRAC   = 0.2
    # TEST_FRAC  = 0.2  (remainder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    print(f"Loading {DATA_PATH} ...")
    data = np.load(DATA_PATH)
    raw  = data["PC"].astype(np.float32)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    raw    = raw[:, :N_PCS] if N_PCS else raw
    T, N   = raw.shape
    print(f"Traces: {T} steps x {N} features")

    train_end = int(T * TRAIN_FRAC)
    val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

    # z-score each neuron over the full recording before splitting
    mu  = raw.mean(0, keepdims=True)
    sd  = raw.std(0,  keepdims=True) + 1e-8
    raw = (raw - mu) / sd

    train_ds = CalciumDataset(raw[:train_end],        CONTEXT, PRED_LEN)
    val_ds   = CalciumDataset(raw[train_end:val_end], CONTEXT, PRED_LEN)
    test_ds  = CalciumDataset(raw[val_end:],          CONTEXT, PRED_LEN)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}  |  Test windows: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model ----
    config = NeuralPredictionConfig()
    config.seq_length          = CONTEXT + PRED_LEN
    config.pred_length         = PRED_LEN
    config.compression_factor  = 16       # tokens per context
    config.decoder_hidden_size = 128    # matches paper
    config.conditioning_dim    = 1024   # matches paper
    config.decoder_num_layers  = 1
    config.decoder_num_heads   = 16     # matches paper
    config.poyo_num_latents    = 8

    input_size = [[N]]   # single session with N neurons
    model = ProbabilisticPOCO(config, input_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    # ---- Training ----
    PATIENCE = 10   # early stopping — stop if val NLL doesn't improve for 10 epochs
    train_nlls, val_nlls, val_maes = [], [], []
    best_nll   = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_nll        = train_epoch(model, train_loader, optimiser, device)
        val_nll, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step(val_nll)

        train_nlls.append(train_nll)
        val_nlls.append(val_nll)
        val_maes.append(val_mae)

        tag = " *" if val_nll < best_nll else ""
        if val_nll < best_nll:
            best_nll   = val_nll
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_nll={train_nll:.4f}  "
              f"val_nll={val_nll:.4f}  "
              f"val_mae={val_mae:.4f}{tag}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest val NLL: {best_nll:.4f}  — saved to {MODEL_PATH}")

    # --- Test evaluation (best checkpoint, never seen during training) ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    test_nll, test_mae = eval_epoch(model, test_loader, device)
    print(f"Test  NLL: {test_nll:.4f}  |  Test MAE: {test_mae:.4f}")

    np.savez(RESULTS_PATH,
             train_nlls=train_nlls, val_nlls=val_nlls, val_maes=val_maes,
             test_nll=test_nll, test_mae=test_mae)
    print(f"Loss history saved to {RESULTS_PATH}")
