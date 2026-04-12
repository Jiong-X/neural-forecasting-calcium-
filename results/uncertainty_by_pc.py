# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
uncertainty_by_pc.py
--------------------
Plots the distribution of predicted σ (aleatoric uncertainty) from the
probabilistic POCO model, broken down by principal component.

Shows 5 representative PCs spread across the 128 components to illustrate
whether the model learns different uncertainty profiles for early PCs
(dominant population dynamics) vs later PCs (noisier variance).

Usage (run from repo root):
    python results/uncertainty_by_pc.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import get_splits
from src.model   import ProbabilisticForecaster

CONTEXT         = 48
PRED_LEN        = 16
N_CHANNELS      = 128
DEVICE          = "cpu"
MODEL_PATH_PROB = "models/saved/model.pt"
FIGURES_DIR     = "results/figures"

# Representative PCs spread across the 128 components (0-indexed)
REPRESENTATIVE_PCS = [0, 31, 63, 95, 127]
PC_LABELS = [f"PC {p + 1}" for p in REPRESENTATIVE_PCS]   # 1-indexed for display

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading probabilistic POCO...")
model = ProbabilisticForecaster(
    seq_length=CONTEXT + PRED_LEN, pred_length=PRED_LEN, n_channels=N_CHANNELS
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH_PROB, map_location=DEVICE, weights_only=True))
model.eval()

# ---------------------------------------------------------------------------
# Run inference on test set, collect σ per PC
# ---------------------------------------------------------------------------
print("Running inference on test set...")
_, _, test_ds = get_splits(seq_length=CONTEXT + PRED_LEN, pred_length=PRED_LEN)

# sigma_store[pc_idx] → list of σ values (one per pred step per window)
sigma_store = {pc: [] for pc in REPRESENTATIVE_PCS}

with torch.no_grad():
    for i in range(len(test_ds)):
        ctx, _ = test_ds[i]                          # (context_len, N)
        x      = ctx.unsqueeze(0).to(DEVICE)         # (1, context_len, N)
        pred   = model(x)                            # Prediction(mean, logvar)
        sigma  = pred.sigma[0].cpu().numpy()         # (pred_len, N)

        for pc in REPRESENTATIVE_PCS:
            sigma_store[pc].extend(sigma[:, pc].tolist())

print(f"Collected σ from {len(test_ds)} windows.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(REPRESENTATIVE_PCS)))

fig, axes = plt.subplots(1, len(REPRESENTATIVE_PCS),
                         figsize=(14, 4), sharey=False)

for ax, pc, label, color in zip(axes, REPRESENTATIVE_PCS, PC_LABELS, colors):
    sigmas = np.array(sigma_store[pc])
    ax.hist(sigmas, bins=40, color=color, edgecolor="none", alpha=0.85)
    ax.axvline(np.median(sigmas), color="black", lw=1.2, linestyle="--",
               label=f"median={np.median(sigmas):.3f}")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("σ (predicted std)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

axes[0].set_ylabel("Count", fontsize=10)

fig.suptitle(
    "Predicted uncertainty (σ) distribution by principal component\n"
    "Probabilistic POCO — test set",
    fontsize=12, fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs(FIGURES_DIR, exist_ok=True)
out = os.path.join(FIGURES_DIR, "uncertainty_by_pc.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
