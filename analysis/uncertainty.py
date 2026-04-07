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


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poco_src.POCO_prob import ProbabilisticPOCO, nll_loss
from poco_src.standalone_poco import NeuralPredictionConfig

# ---------------------------------------------------------------------------
# MC Dropout helper
# ---------------------------------------------------------------------------

def enable_dropout(model: torch.nn.Module):
    """Set all Dropout layers to train mode so they stay active at inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, x_list, T: int = 50):
    """
    Run T stochastic forward passes with MC Dropout.

    Args:
        model  : ProbabilisticPOCO (loaded, on correct device)
        x_list : list of (L, B, D) tensors — one per session
        T      : number of MC samples

    Returns:
        mus    : (T, pred_len, B, D)  — predicted means per sample
        vars_  : (T, pred_len, B, D)  — predicted variances per sample
    """
    model.eval()
    enable_dropout(model)

    mus, vars_ = [], []
    for _ in range(T):
        dists = model(x_list)
        # single-session: take dists[0]
        mus.append(dists[0].mean)             # (pred_len, B, D)
        vars_.append(dists[0].variance)       # (pred_len, B, D)

    mus   = torch.stack(mus)    # (T, pred_len, B, D)
    vars_ = torch.stack(vars_)  # (T, pred_len, B, D)
    return mus, vars_


def decompose_uncertainty(mus, vars_):
    """
    Law of total variance decomposition.

    Args:
        mus   : (T, pred_len, B, D)
        vars_ : (T, pred_len, B, D)

    Returns:
        mu_mean   : (pred_len, B, D)  — mean prediction
        aleatoric : (pred_len, B, D)  — sqrt(E[σ²])
        epistemic : (pred_len, B, D)  — sqrt(Var[μ])
        total     : (pred_len, B, D)  — sqrt(aleatoric² + epistemic²)
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

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_poco_prob.pt"
    RESULTS_PATH = "results/uncertainty.npz"
    os.makedirs("results", exist_ok=True)

    N_PCS      = 128
    CONTEXT    = 48
    PRED_LEN   = 16
    BATCH_SIZE = 16
    MC_SAMPLES = 50     # number of stochastic forward passes
    VAL_FRAC   = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    data = np.load(DATA_PATH)
    raw  = data["PC"].astype(np.float32)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    raw  = raw[:, :N_PCS]
    T, N = raw.shape

    split  = int(T * (1 - VAL_FRAC))
    val_ds = CalciumDataset(raw[split:], context_len=CONTEXT, pred_len=PRED_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model ----
    config = NeuralPredictionConfig()
    config.seq_length          = CONTEXT + PRED_LEN
    config.pred_length         = PRED_LEN
    config.compression_factor  = 16
    config.decoder_hidden_size = 64
    config.conditioning_dim    = 128
    config.decoder_num_layers  = 1
    config.decoder_num_heads   = 8
    config.poyo_num_latents    = 8

    model = ProbabilisticPOCO(config, [[N]]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"Loaded weights from {MODEL_PATH}")

    # ---- MC Dropout over validation set ----
    all_aleatoric, all_epistemic, all_total, all_mu = [], [], [], []

    for batch_idx, (X, Y) in enumerate(val_loader):
        X = X.to(device)
        x_list = [X.permute(1, 0, 2)]   # (L, B, D)

        mus, vars_ = mc_predict(model, x_list, T=MC_SAMPLES)
        mu_mean, aleatoric, epistemic, total = decompose_uncertainty(mus, vars_)

        # Average over batch dimension → (pred_len, D)
        all_mu.append(mu_mean.mean(1).cpu().numpy())
        all_aleatoric.append(aleatoric.mean(1).cpu().numpy())
        all_epistemic.append(epistemic.mean(1).cpu().numpy())
        all_total.append(total.mean(1).cpu().numpy())

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(val_loader)}  "
                  f"aleatoric={aleatoric.mean():.4f}  "
                  f"epistemic={epistemic.mean():.4f}")

    # Average across all batches
    mu_mean   = np.mean(all_mu,        axis=0)   # (pred_len, N)
    aleatoric = np.mean(all_aleatoric, axis=0)
    epistemic = np.mean(all_epistemic, axis=0)
    total     = np.mean(all_total,     axis=0)

    print(f"\n--- Uncertainty Summary (averaged over val set) ---")
    print(f"Aleatoric (data noise) : {aleatoric.mean():.4f} ± {aleatoric.std():.4f}")
    print(f"Epistemic (model unc.) : {epistemic.mean():.4f} ± {epistemic.std():.4f}")
    print(f"Total predictive       : {total.mean():.4f} ± {total.std():.4f}")
    print(f"Epistemic / Total ratio: {(epistemic.mean() / total.mean()):.3f}")

    np.savez(
        RESULTS_PATH,
        mu_mean=mu_mean,
        aleatoric=aleatoric,
        epistemic=epistemic,
        total=total,
    )
    print(f"\nSaved to {RESULTS_PATH}")
