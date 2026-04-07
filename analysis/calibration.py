"""
Calibration analysis for Probabilistic POCO.

Produces:
  1. Reliability diagram (calibration curve)
  2. Coverage vs forecast horizon
  3. Sharpness (mean sigma) vs horizon
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scipy.stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poco_src.POCO_prob import ProbabilisticPOCO, CalciumDataset
from poco_src.standalone_poco import NeuralPredictionConfig

# ---------------------------------------------------------------------------
# Config — must match POCO_prob.py training settings
# ---------------------------------------------------------------------------
DATA_PATH   = "data/processed/0.npz"
MODEL_PATH  = "models/best_poco_prob.pt"
OUT_DIR     = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

N_PCS       = 128
CONTEXT     = 48
PRED_LEN    = 16
TRAIN_FRAC  = 0.6
VAL_FRAC    = 0.2
BATCH_SIZE  = 64

HIDDEN_DIM  = 128
COND_DIM    = 1024
NUM_LATENTS = 8
NUM_LAYERS  = 1
NUM_HEADS   = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load data — test set only (never seen during training)
# ---------------------------------------------------------------------------
data   = np.load(DATA_PATH)
raw    = data["PC"].astype(np.float32)
if raw.shape[0] < raw.shape[1]:
    raw = raw.T
traces = raw[:, :N_PCS]
T, N   = traces.shape

train_end = int(T * TRAIN_FRAC)
val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

test_ds     = CalciumDataset(traces[val_end:], context_len=CONTEXT, pred_len=PRED_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test windows: {len(test_ds)}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
config = NeuralPredictionConfig()
config.seq_length          = CONTEXT + PRED_LEN
config.pred_length         = PRED_LEN
config.compression_factor  = 16
config.decoder_hidden_size = HIDDEN_DIM
config.conditioning_dim    = COND_DIM
config.decoder_num_layers  = NUM_LAYERS
config.decoder_num_heads   = NUM_HEADS
config.poyo_num_latents    = NUM_LATENTS

model = ProbabilisticPOCO(config, [[N]]).to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Model loaded.")

# ---------------------------------------------------------------------------
# Collect all predictions over the test set
# ---------------------------------------------------------------------------
all_mu    = []   # (total_windows, pred_len, N)
all_sigma = []
all_y     = []

with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        x_list = [X.permute(1, 0, 2)]
        dists  = model(x_list)
        dist   = dists[0]   # (pred_len, B, N)

        all_mu.append(dist.mean.permute(1, 0, 2).cpu().numpy())
        all_sigma.append(dist.scale.permute(1, 0, 2).cpu().numpy())
        all_y.append(Y.numpy())

all_mu    = np.concatenate(all_mu,    axis=0)   # (W, pred_len, N)
all_sigma = np.concatenate(all_sigma, axis=0)
all_y     = np.concatenate(all_y,     axis=0)

print(f"Collected predictions: shape {all_mu.shape}")

# ---------------------------------------------------------------------------
# Figure 1 — Reliability diagram (calibration curve)
# ---------------------------------------------------------------------------
alphas   = np.linspace(0.05, 0.95, 19)   # confidence levels
coverage = []

for alpha in alphas:
    z       = scipy.stats.norm.ppf((1 + alpha) / 2)
    inside  = (all_y > all_mu - z * all_sigma) & \
              (all_y < all_mu + z * all_sigma)
    coverage.append(inside.mean())

coverage = np.array(coverage)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
ax.plot(alphas, coverage, "o-", color="#7d3c98", lw=2.0,
        ms=6, label="POCO prob.")

# shade overconfident / underconfident regions
ax.fill_between([0, 1], [0, 0], [0, 1],
                alpha=0.05, color="red",   label="Overconfident region")
ax.fill_between([0, 1], [0, 1], [1, 1],
                alpha=0.05, color="blue",  label="Underconfident region")

# ECE — expected calibration error
ece = np.abs(coverage - alphas).mean()
ax.set_title(f"Reliability Diagram — Probabilistic POCO\n"
             f"Expected Calibration Error (ECE) = {ece:.4f}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Nominal confidence level α", fontsize=11)
ax.set_ylabel("Empirical coverage", fontsize=11)
ax.legend(fontsize=9, framealpha=0.7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "calibration_curve.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2 — Coverage vs forecast horizon (at 50%, 90% confidence)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
steps = np.arange(1, PRED_LEN + 1)

for alpha, color, label in [
    (0.50, "#e8a838", "50% interval"),
    (0.90, "#7d3c98", "90% interval"),
]:
    z        = scipy.stats.norm.ppf((1 + alpha) / 2)
    cov_h    = []
    for h in range(PRED_LEN):
        inside = (all_y[:, h, :] > all_mu[:, h, :] - z * all_sigma[:, h, :]) & \
                 (all_y[:, h, :] < all_mu[:, h, :] + z * all_sigma[:, h, :])
        cov_h.append(inside.mean())
    ax.plot(steps, cov_h, "o-", color=color, lw=2.0, ms=6, label=f"Empirical {label}")
    ax.axhline(y=alpha, color=color, lw=1.2, ls="--",
               label=f"Nominal {label} ({alpha:.0%})", alpha=0.6)

ax.set_xlabel("Forecast step (horizon)", fontsize=11)
ax.set_ylabel("Empirical coverage", fontsize=11)
ax.set_title("Coverage vs Forecast Horizon\n"
             "Dashed = nominal target   |   Solid = empirical",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.7, ncol=2)
ax.set_ylim(0, 1.05)
ax.set_xticks(steps)
ax.grid(alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "calibration_vs_horizon.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3 — Sharpness (mean sigma) vs horizon
# ---------------------------------------------------------------------------
mean_sigma = all_sigma.mean(axis=(0, 2))   # (pred_len,) — avg over windows and PCs

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, mean_sigma, "o-", color="#c0392b", lw=2.0, ms=6)
ax.fill_between(steps, 0, mean_sigma, alpha=0.15, color="#c0392b")
ax.set_xlabel("Forecast step (horizon)", fontsize=11)
ax.set_ylabel("Mean σ  (z-score units)", fontsize=11)
ax.set_title("Sharpness vs Forecast Horizon\n"
             "Mean predicted standard deviation across all PCs and windows",
             fontsize=11, fontweight="bold")
ax.set_xticks(steps)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "sharpness_vs_horizon.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nCalibration summary:")
print(f"  ECE (expected calibration error): {ece:.4f}")
print(f"  Mean sigma (sharpness):           {all_sigma.mean():.4f}")
print(f"  90% coverage (overall):           {coverage[alphas >= 0.89][0]:.4f}  (target 0.90)")
print(f"  50% coverage (overall):           {coverage[alphas >= 0.49][0]:.4f}  (target 0.50)")
