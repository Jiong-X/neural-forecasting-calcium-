# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
Uncertainty quantification for ProbabilisticPOCO.

Decomposes predictive uncertainty into:
  - Aleatoric  : irreducible data noise  → mean of predicted variances  E[σ²]
  - Epistemic  : model / parameter uncertainty → variance of predicted means  Var[μ]
  - Total      : aleatoric + epistemic  (law of total variance)

Method: Monte Carlo Dropout — run T stochastic forward passes with dropout
kept active at inference time.  Each pass samples a different sub-network,
giving T draws from an approximate posterior over model parameters.

Outputs saved to results/uncertainty.npz:
  mu_mean   (pred_len, N)   — mean prediction across MC samples
  aleatoric (pred_len, N)   — per-step, per-neuron aleatoric std
  epistemic (pred_len, N)   — per-step, per-neuron epistemic std
  total     (pred_len, N)   — total predictive std
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ProbabilisticForecaster
from src.dataset import get_splits

# ---------------------------------------------------------------------------
# MC Dropout helper
# ---------------------------------------------------------------------------

def enable_dropout(model: torch.nn.Module):
    """Set all Dropout layers to train mode so they stay active at inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, x, T: int = 50):
    """
    Run T stochastic forward passes with MC Dropout.

    Args:
        model : ProbabilisticForecaster (loaded, on correct device)
        x     : (B, context_len, N) tensor
        T     : number of MC samples

    Returns:
        mus    : (T, B, pred_len, N)  — predicted means per sample
        vars_  : (T, B, pred_len, N)  — predicted variances per sample
    """
    model.eval()
    enable_dropout(model)

    mus, vars_ = [], []
    for _ in range(T):
        pred = model(x)
        mus.append(pred.mean)             # (B, pred_len, N)
        vars_.append(pred.variance)       # (B, pred_len, N)

    mus   = torch.stack(mus)    # (T, B, pred_len, N)
    vars_ = torch.stack(vars_)  # (T, B, pred_len, N)
    return mus, vars_


def decompose_uncertainty(mus, vars_):
    """
    Law of total variance decomposition.

    Args:
        mus   : (T, B, pred_len, N)
        vars_ : (T, B, pred_len, N)

    Returns:
        mu_mean   : (B, pred_len, N)  — mean prediction
        aleatoric : (B, pred_len, N)  — sqrt(E[σ²])
        epistemic : (B, pred_len, N)  — sqrt(Var[μ])
        total     : (B, pred_len, N)  — sqrt(aleatoric² + epistemic²)
    """
    mu_mean   = mus.mean(0)                     # E[μ]
    aleatoric = vars_.mean(0).sqrt()            # sqrt(E[σ²])
    epistemic = mus.var(0).sqrt()               # sqrt(Var[μ])
    total     = (vars_.mean(0) + mus.var(0)).sqrt()
    return mu_mean, aleatoric, epistemic, total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    MODEL_PATH   = "models/saved/model.pt"
    RESULTS_PATH = "results/uncertainty.npz"
    os.makedirs("results", exist_ok=True)

    SEQ_LENGTH = 64
    PRED_LEN   = 16
    N_CHANNELS = 128
    BATCH_SIZE = 16
    MC_SAMPLES = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data (same 60/20/20 split as train.py / test.py) ----
    _, _, test_ds = get_splits(seq_length=SEQ_LENGTH, pred_length=PRED_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    model = ProbabilisticForecaster(
        seq_length=SEQ_LENGTH, pred_length=PRED_LEN, n_channels=N_CHANNELS
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"Loaded weights from {MODEL_PATH}")

    # ---- MC Dropout over test set ----
    all_aleatoric, all_epistemic, all_total, all_mu = [], [], [], []

    for batch_idx, (X, Y) in enumerate(test_loader):
        X = X.to(device)              # (B, context_len, N)

        mus, vars_ = mc_predict(model, X, T=MC_SAMPLES)
        mu_mean, aleatoric, epistemic, total = decompose_uncertainty(mus, vars_)

        # Average over batch dimension (dim 0) → (pred_len, N)
        all_mu.append(mu_mean.mean(0).cpu().numpy())
        all_aleatoric.append(aleatoric.mean(0).cpu().numpy())
        all_epistemic.append(epistemic.mean(0).cpu().numpy())
        all_total.append(total.mean(0).cpu().numpy())

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(test_loader)}  "
                  f"aleatoric={aleatoric.mean():.4f}  "
                  f"epistemic={epistemic.mean():.4f}")

    # Average across all batches
    mu_mean   = np.mean(all_mu,        axis=0)   # (pred_len, N)
    aleatoric = np.mean(all_aleatoric, axis=0)
    epistemic = np.mean(all_epistemic, axis=0)
    total     = np.mean(all_total,     axis=0)

    print(f"\n--- Uncertainty Summary (averaged over test set) ---")
    print(f"Aleatoric (data noise) : {aleatoric.mean():.4f} +/- {aleatoric.std():.4f}")
    print(f"Epistemic (model unc.) : {epistemic.mean():.4f} +/- {epistemic.std():.4f}")
    print(f"Total predictive       : {total.mean():.4f} +/- {total.std():.4f}")
    print(f"Epistemic / Total ratio: {(epistemic.mean() / total.mean()):.3f}")

    np.savez(
        RESULTS_PATH,
        mu_mean=mu_mean,
        aleatoric=aleatoric,
        epistemic=epistemic,
        total=total,
    )
    print(f"\nSaved to {RESULTS_PATH}")
