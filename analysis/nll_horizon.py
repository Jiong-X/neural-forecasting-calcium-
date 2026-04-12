"""
nll_horizon.py
-------------------
Compare Gaussian NLL of POCO (prob.) and MLP as a function of prediction horizon step.

For each step h in [1, PRED_LEN] we compute:
  - Mean Gaussian NLL averaged over all test windows and all 128 PCs

Outputs:
    results/figures/horizon_nll.png
    results/horizon_nll.npz
"""

import os
import sys
import numpy as np
import torch
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util                import trainingConfig, CalciumDataset
from src.dataset             import _load_traces
from src.model               import ProbabilisticForecaster
from src.baseline_models.MLP import MLPHead

os.makedirs("results/figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_LEN    = 64
PRED_LEN   = 16
CONTEXT    = SEQ_LEN - PRED_LEN   # 48
COND_DIM   = 1024
BATCH_SIZE = 64
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------
traces  = _load_traces()
T, N    = traces.shape
val_end = int(T * (TRAIN_FRAC + VAL_FRAC))

test_ds     = CalciumDataset(traces[val_end:], context_len=CONTEXT, pred_len=PRED_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test windows: {len(test_ds)}  |  N channels: {N}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def _load_prob_poco():
    model = ProbabilisticForecaster(seq_length=SEQ_LEN, pred_length=PRED_LEN, n_channels=N).to(device)
    path  = trainingConfig(model_name="ProbabilisticPOCO").save_path
    print(f"Loading prob POCO from: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_mlp():
    model = MLPHead(n_neurons=N, context_len=CONTEXT, cond_dim=COND_DIM, pred_len=PRED_LEN).to(device)
    path  = trainingConfig(model_name="MLP").save_path
    print(f"Loading MLP from: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


prob_model = _load_prob_poco()
mlp_model  = _load_mlp()

# ---------------------------------------------------------------------------
# Per-step NLL evaluation
# ---------------------------------------------------------------------------

def eval_nll_per_step(model, loader):
    """
    Return (PRED_LEN,) array of mean Gaussian NLL per forecast step.
    NLL = -log p(y | mu, sigma), averaged over batch and PCs.
    """
    nll_sum = np.zeros(PRED_LEN)
    n_batch = 0
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)   # (B, context_len, N)
            Y = Y.to(device)   # (B, pred_len,    N)

            pred  = model(X)
            dist  = Normal(pred.mean, pred.sigma)        # (B, pred_len, N)
            nll   = -dist.log_prob(Y)                    # (B, pred_len, N)
            nll_sum += nll.mean(dim=(0, 2)).cpu().numpy()  # mean over B and N
            n_batch += 1

    return nll_sum / n_batch


print("\nEvaluating POCO (prob.) NLL ...")
poco_nll = eval_nll_per_step(prob_model, test_loader)

print("Evaluating MLP NLL ...")
mlp_nll  = eval_nll_per_step(mlp_model,  test_loader)

steps = np.arange(1, PRED_LEN + 1)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.savez("results/horizon_nll.npz",
         steps=steps,
         poco_nll=poco_nll,
         mlp_nll=mlp_nll)
print("\nSaved results/horizon_nll.npz")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps, mlp_nll,  color="#2980b9", linewidth=2.2,
        marker="^", markersize=5, label="MLP")
ax.plot(steps, poco_nll, color="#e74c3c", linewidth=2.2,
        marker="s", markersize=5, label="POCO (prob.)")

ax.axhline(mlp_nll.mean(),  color="#2980b9", linewidth=1.0, linestyle="--",
           label=f"MLP mean NLL = {mlp_nll.mean():.4f}")
ax.axhline(poco_nll.mean(), color="#e74c3c", linewidth=1.0, linestyle="--",
           label=f"POCO (prob.) mean NLL = {poco_nll.mean():.4f}")

ax.set_xlabel("Prediction step  (frames)", fontsize=12)
ax.set_ylabel("Gaussian NLL", fontsize=12)
ax.set_title("Gaussian NLL vs prediction horizon\n"
             "C=48  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=12, fontweight="bold")
ax.set_xticks(steps)
ax.legend(fontsize=10, framealpha=0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0.5, PRED_LEN + 0.5)

plt.tight_layout()
out = "results/figures/horizon_nll.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print(f"\n{'Step':>5}  {'MLP NLL':>10}  {'POCO NLL':>10}")
print("-" * 30)
for h, (ml, pc) in enumerate(zip(mlp_nll, poco_nll), start=1):
    print(f"{h:>5}  {ml:>10.4f}  {pc:>10.4f}")
