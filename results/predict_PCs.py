# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
predict_PCs.py
--------------
Generates predictions from all three models for a single window from the
held-out test set and plots context + ground truth + all three forecasts
for 3 randomly chosen PCs, stacked vertically.

Usage (run from repo root):
    python results/predict_PCs.py             
    python results/predict_PCs.py --window 160 --seed 7 --split val
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure repo root is on the path when running from results/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import get_splits
from src.model   import ProbabilisticForecaster, DeterministicPOCO
from src.baseline_models.MLP import MLPHead

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTEXT    = 48
PRED_LEN   = 16
N_CHANNELS = 128
DEVICE     = "cpu"

MODEL_PATH_PROB = "models/saved/model.pt"
MODEL_PATH_DET  = "models/saved/best_DeterministicPoco.pt"
MODEL_PATH_MLP  = "models/saved/best_MLP.pt"

FIGURES_DIR = "results/figures"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=100,
                    help="Window index (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for PC selection (default: 0)")
parser.add_argument("--split", choices=["test", "val"], default="test",
                    help="Which split to draw the window from (default: test)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load splits via the same pipeline used in train.py
# ---------------------------------------------------------------------------
_, val_ds, test_ds = get_splits(seq_length=CONTEXT + PRED_LEN, pred_length=PRED_LEN)

ds = test_ds if args.split == "test" else val_ds

assert args.window < len(ds), \
    f"--window {args.window} out of range — {args.split} set has {len(ds)} windows"

context, target = ds[args.window]   # (context_len, N), (pred_len, N) — tensors

# Add batch dimension for inference: (1, context_len, N)
x = context.unsqueeze(0).to(DEVICE)

# ---------------------------------------------------------------------------
# Load models and run inference
# ---------------------------------------------------------------------------
def load_prob():
    m = ProbabilisticForecaster(
        seq_length=CONTEXT + PRED_LEN, pred_length=PRED_LEN, n_channels=N_CHANNELS
    ).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH_PROB, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

def load_det():
    m = DeterministicPOCO(
        seq_length=CONTEXT + PRED_LEN, pred_length=PRED_LEN, n_channels=N_CHANNELS
    ).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH_DET, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

def load_mlp():
    m = MLPHead(
        n_neurons=N_CHANNELS, context_len=CONTEXT, cond_dim=1024, pred_len=PRED_LEN
    ).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH_MLP, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

print("Loading models and running inference...")
with torch.no_grad():
    # Probabilistic POCO → Prediction(mean, sigma), (B, pred_len, N)
    prob_pred  = load_prob()(x)
    prob_mean  = prob_pred.mean[0].cpu().numpy()    # (pred_len, N)
    prob_sigma = prob_pred.sigma[0].cpu().numpy()   # (pred_len, N)

    # Deterministic POCO → Prediction(mean), (B, pred_len, N)
    det_pred   = load_det()(x)
    det_mean   = det_pred.mean[0].cpu().numpy()     # (pred_len, N)

    # MLP → Prediction(mean, logvar), (B, pred_len, N)
    mlp_pred  = load_mlp()(x)
    mlp_mean  = mlp_pred.mean[0].cpu().numpy()     # (pred_len, N)
    mlp_sigma = mlp_pred.sigma[0].cpu().numpy()    # (pred_len, N)

# ---------------------------------------------------------------------------
# Pick 3 random PCs
# ---------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
pcs = sorted(rng.choice(N_CHANNELS, size=3, replace=False).tolist())
print(f"Plotting PCs: {pcs}")

ctx_steps  = np.arange(CONTEXT)
pred_steps = np.arange(CONTEXT, CONTEXT + PRED_LEN)

context_np = context.cpu().numpy()
target_np  = target.cpu().numpy()

# ---------------------------------------------------------------------------
# Stacked plot — one row per PC
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for ax, pc in zip(axes, pcs):
    ax.plot(ctx_steps,  context_np[:, pc], color="black", lw=1.5, label="Context")
    ax.plot(pred_steps, target_np[:, pc],  color="black", lw=1.5, linestyle="--",
            label="Ground truth")

    ax.plot(pred_steps, prob_mean[:, pc], color="tab:blue", lw=1.5, label="Prob POCO")
    ax.fill_between(pred_steps,
                    prob_mean[:, pc] - prob_sigma[:, pc],
                    prob_mean[:, pc] + prob_sigma[:, pc],
                    color="tab:blue", alpha=0.2)

    ax.plot(pred_steps, det_mean[:, pc], color="tab:orange", lw=1.5, label="Det POCO")

    ax.plot(pred_steps, mlp_mean[:, pc], color="tab:green", lw=1.5, label="MLP")
    ax.fill_between(pred_steps,
                    mlp_mean[:, pc] - mlp_sigma[:, pc],
                    mlp_mean[:, pc] + mlp_sigma[:, pc],
                    color="tab:green", alpha=0.2)

    ax.axvline(x=CONTEXT - 0.5, color="grey", linestyle=":", lw=1)
    ax.set_ylabel("PC activity (z-scored)")
    ax.set_title(f"PC {pc}")
    ax.legend(loc="upper left", fontsize=8)

axes[-1].set_xlabel("Timestep")
fig.suptitle(f"{args.split.capitalize()} window {args.window}  (context={CONTEXT}, horizon={PRED_LEN})",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])

os.makedirs(FIGURES_DIR, exist_ok=True)
out = os.path.join(FIGURES_DIR, f"predict_one_{args.split}_w{args.window}_s{args.seed}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
