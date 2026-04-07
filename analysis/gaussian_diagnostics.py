# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
gaussian_diagnostics.py
-----------------------
Empirical validation of the Gaussian likelihood assumption for POCO_prob.

Tests performed:
  1. Shapiro-Wilk normality test  (on a random sample of neurons)
  2. Q-Q plots                    (standardised residuals vs Normal)
  3. Residual histogram           (overlaid with fitted Normal)

Usage:
  python gaussian_diagnostics.py
  python gaussian_diagnostics.py --checkpoint models/saved/model.pt
                                 --n_neurons   16
                                 --n_samples   2000
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro, probplot, norm

import torch

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import get_test_dataset
from src.model   import ProbabilisticForecaster


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="models/saved/model.pt")
parser.add_argument("--n_neurons",  type=int, default=16,
                    help="Number of neurons (PCs) to plot Q-Q for")
parser.add_argument("--n_samples",  type=int, default=2000,
                    help="Max test windows to evaluate")
parser.add_argument("--context",    type=int, default=48)
parser.add_argument("--pred",       type=int, default=16)
parser.add_argument("--n_channels", type=int, default=128)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load model & data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model …")
model = ProbabilisticForecaster(
    seq_length  = args.context + args.pred,
    pred_length = args.pred,
    n_channels  = args.n_channels,
).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()

print("Loading test dataset …")
test_ds = get_test_dataset(
    seq_length  = args.context + args.pred,
    pred_length = args.pred,
)

# subsample for speed
idx = np.random.choice(len(test_ds), min(args.n_samples, len(test_ds)), replace=False)
X   = torch.stack([test_ds[i][0] for i in idx]).to(DEVICE)   # (S, C, N)
Y   = torch.stack([test_ds[i][1] for i in idx]).to(DEVICE)   # (S, P, N)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Get predictions and compute standardised residuals
# ─────────────────────────────────────────────────────────────────────────────
print("Running forward pass …")
with torch.no_grad():
    mean, logvar = model(X)                 # (S, P, N)
sigma = (0.5 * logvar).exp()               # (S, P, N)

# standardised residuals: z = (y - mu) / sigma
residuals_raw = (Y - mean).cpu().numpy()           # (S, P, N)
sigma_np      = sigma.cpu().numpy()
residuals_std = residuals_raw / (sigma_np + 1e-8)  # (S, P, N)

# flatten to (S*P, N) → one column per neuron
S, P, N = residuals_std.shape
res_flat = residuals_std.reshape(-1, N)            # (S*P, N)
raw_flat = residuals_raw.reshape(-1, N)

print(f"Residual matrix shape: {res_flat.shape}  (windows×steps, neurons)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Shapiro-Wilk test  (max 5000 samples per neuron)
# ─────────────────────────────────────────────────────────────────────────────
MAX_SW = 5000
sw_stats, sw_pvals = [], []
for n in range(N):
    col = res_flat[:, n]
    col = col[np.isfinite(col)]
    if len(col) > MAX_SW:
        col = np.random.choice(col, MAX_SW, replace=False)
    stat, p = shapiro(col)
    sw_stats.append(stat)
    sw_pvals.append(p)

sw_stats = np.array(sw_stats)
sw_pvals = np.array(sw_pvals)
frac_normal = (sw_pvals > 0.05).mean()

print(f"\nShapiro-Wilk (standardised residuals, {N} neurons):")
print(f"  Mean W statistic : {sw_stats.mean():.4f}")
print(f"  Fraction p > 0.05: {frac_normal*100:.1f}%  "
      f"(= {int(frac_normal*N)}/{N} neurons not rejected at α=0.05)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Figure layout
#    Row 0: Shapiro-Wilk W statistics  +  p-value histogram
#    Row 1: Residual histogram (all neurons pooled)
#    Row 2+: Q-Q plots for n_neurons sample neurons
# ─────────────────────────────────────────────────────────────────────────────
n_qq   = min(args.n_neurons, N)
n_cols = 4
n_rows_qq = (n_qq + n_cols - 1) // n_cols

fig = plt.figure(figsize=(5 * n_cols, 4 * (3 + n_rows_qq)))
gs  = gridspec.GridSpec(3 + n_rows_qq, n_cols, figure=fig,
                        hspace=0.55, wspace=0.4)

# ── 4a. Shapiro-Wilk W distribution ─────────────────────────────────────────
ax_w = fig.add_subplot(gs[0, :2])
ax_w.hist(sw_stats, bins=30, color="#4878CF", edgecolor="white", alpha=0.85)
ax_w.axvline(sw_stats.mean(), color="crimson", lw=1.5, label=f"mean W={sw_stats.mean():.4f}")
ax_w.set_xlabel("Shapiro-Wilk W statistic", fontsize=11)
ax_w.set_ylabel("Number of neurons", fontsize=11)
ax_w.set_title("Shapiro-Wilk W across all neurons", fontsize=12)
ax_w.legend(fontsize=9)

# ── 4b. p-value distribution ─────────────────────────────────────────────────
ax_p = fig.add_subplot(gs[0, 2:])
ax_p.hist(sw_pvals, bins=30, color="#6ACC65", edgecolor="white", alpha=0.85)
ax_p.axvline(0.05, color="crimson", lw=1.5, linestyle="--", label="α = 0.05")
ax_p.set_xlabel("Shapiro-Wilk p-value", fontsize=11)
ax_p.set_ylabel("Number of neurons", fontsize=11)
ax_p.set_title(f"p-value distribution  "
               f"({frac_normal*100:.1f}% not rejected)", fontsize=12)
ax_p.legend(fontsize=9)

# ── 4c. Pooled residual histogram ────────────────────────────────────────────
ax_h = fig.add_subplot(gs[1, :])
pooled = res_flat.flatten()
pooled = pooled[np.isfinite(pooled)]
# clip extreme outliers for display
clip = np.percentile(np.abs(pooled), 99.5)
pooled_clip = np.clip(pooled, -clip, clip)

counts, bins, _ = ax_h.hist(pooled_clip, bins=120, density=True,
                             color="#4878CF", alpha=0.7, edgecolor="white",
                             label="Standardised residuals")
x_line = np.linspace(bins[0], bins[-1], 400)
ax_h.plot(x_line, norm.pdf(x_line), "r-", lw=2, label="N(0,1)")
ax_h.set_xlabel("Standardised residual  (y − μ) / σ", fontsize=11)
ax_h.set_ylabel("Density", fontsize=11)
ax_h.set_title("Pooled residual distribution vs Standard Normal", fontsize=12)
ax_h.legend(fontsize=10)

# ── 4d. Annotation row ───────────────────────────────────────────────────────
ax_txt = fig.add_subplot(gs[2, :])
ax_txt.axis("off")
summary = (
    f"Shapiro-Wilk normality test on standardised residuals  ·  "
    f"N = {N} neurons  ·  {S*P} observations per neuron\n"
    f"Mean W = {sw_stats.mean():.4f}   |   "
    f"{frac_normal*100:.1f}% of neurons not rejected at α = 0.05   |   "
    f"Residuals computed as  z = (y − μ) / σ_predicted"
)
ax_txt.text(0.5, 0.5, summary, ha="center", va="center",
            fontsize=10, transform=ax_txt.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))

# ── 4e. Q-Q plots per neuron ─────────────────────────────────────────────────
neuron_ids = np.linspace(0, N - 1, n_qq, dtype=int)
for k, nid in enumerate(neuron_ids):
    row = 3 + k // n_cols
    col = k  % n_cols
    ax  = fig.add_subplot(gs[row, col])

    col_data = res_flat[:, nid]
    col_data = col_data[np.isfinite(col_data)]

    (osm, osr), (slope, intercept, r) = probplot(col_data, dist="norm")
    ax.plot(osm, osr,    ".", ms=2, alpha=0.4, color="#4878CF")
    ax.plot(osm, slope * np.array(osm) + intercept,
            "r-", lw=1.5, label=f"R²={r**2:.3f}")
    ax.set_title(f"PC {nid+1}", fontsize=9)
    ax.set_xlabel("Theoretical quantiles", fontsize=7)
    ax.set_ylabel("Sample quantiles",      fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle(
    "Gaussian Likelihood Diagnostics — POCO_prob Residuals\n"
    "(Shapiro-Wilk Tests + Q-Q Plots of Standardised Residuals)",
    fontsize=14, fontweight="bold", y=1.002
)

out_path = "results/gaussian_diagnostics.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
plt.show()
