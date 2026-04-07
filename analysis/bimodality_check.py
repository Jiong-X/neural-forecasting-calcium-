# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
bimodality_check.py
-------------------
Tests whether the standardised residuals of POCO_prob show bimodality.

Tests performed:
  1. Bimodality coefficient (BC)  — simple closed-form statistic
  2. KDE plots per sampled neuron — visual inspection
  3. Pooled KDE vs Normal         — global picture

Bimodality coefficient:
  BC = (skewness^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
  BC > 0.555 (uniform distribution threshold) suggests bimodality.

Usage:
  python bimodality_check.py
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, skew, kurtosis
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import get_test_dataset
from src.model   import ProbabilisticForecaster

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="models/saved/model.pt")
parser.add_argument("--n_neurons",  type=int, default=16)
parser.add_argument("--n_samples",  type=int, default=2000)
parser.add_argument("--context",    type=int, default=48)
parser.add_argument("--pred",       type=int, default=16)
parser.add_argument("--n_channels", type=int, default=128)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model & data ─────────────────────────────────────────────────────────
print("Loading model …")
model = ProbabilisticForecaster(
    seq_length  = args.context + args.pred,
    pred_length = args.pred,
    n_channels  = args.n_channels,
).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE,
                                 weights_only=False))
model.eval()

print("Loading test dataset …")
test_ds = get_test_dataset(seq_length=args.context + args.pred,
                           pred_length=args.pred)
idx = np.random.choice(len(test_ds), min(args.n_samples, len(test_ds)),
                       replace=False)
X = torch.stack([test_ds[i][0] for i in idx]).to(DEVICE)
Y = torch.stack([test_ds[i][1] for i in idx]).to(DEVICE)

print("Running forward pass …")
with torch.no_grad():
    mean, logvar = model(X)
sigma = (0.5 * logvar).exp()

res_std = ((Y - mean) / (sigma + 1e-8)).cpu().numpy()   # (S, P, N)
S, P, N = res_std.shape
res_flat = res_std.reshape(-1, N)                        # (S*P, N)

# ── Bimodality coefficient per neuron ─────────────────────────────────────────
def bimodality_coeff(x):
    n  = len(x)
    s  = skew(x)
    k  = kurtosis(x, fisher=True)   # excess kurtosis
    correction = 3 * (n - 1)**2 / ((n - 2) * (n - 3))
    return (s**2 + 1) / (k + correction)

bc_vals = np.array([bimodality_coeff(res_flat[:, n]) for n in range(N)])
BC_THRESHOLD = 0.555   # uniform distribution value — above this → bimodal
frac_bimodal = (bc_vals > BC_THRESHOLD).mean()

print(f"\nBimodality Coefficient (BC) across {N} neurons:")
print(f"  Mean BC          : {bc_vals.mean():.4f}  (threshold = {BC_THRESHOLD})")
print(f"  Fraction BC>0.555: {frac_bimodal*100:.1f}%  "
      f"({int(frac_bimodal*N)}/{N} neurons suggest bimodality)")

# ── Plots ─────────────────────────────────────────────────────────────────────
n_qq   = min(args.n_neurons, N)
n_cols = 4
n_rows = (n_qq + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows + 2, n_cols,
                         figsize=(5 * n_cols, 4 * (n_rows + 2)))
axes = axes.flatten()

# ── Panel 0: BC histogram ─────────────────────────────────────────────────────
ax = axes[0]
ax.hist(bc_vals, bins=30, color="#4878CF", edgecolor="white", alpha=0.85)
ax.axvline(BC_THRESHOLD, color="crimson", lw=1.5, linestyle="--",
           label=f"threshold = {BC_THRESHOLD}")
ax.axvline(bc_vals.mean(), color="orange", lw=1.5,
           label=f"mean BC = {bc_vals.mean():.3f}")
ax.set_xlabel("Bimodality Coefficient", fontsize=10)
ax.set_ylabel("Neurons", fontsize=10)
ax.set_title("BC distribution across all neurons", fontsize=11)
ax.legend(fontsize=8)

# ── Panel 1: Pooled KDE ───────────────────────────────────────────────────────
ax = axes[1]
pooled = res_flat.flatten()
pooled = pooled[np.isfinite(pooled)]
clip   = np.percentile(np.abs(pooled), 99)
pooled_c = np.clip(pooled, -clip, clip)

kde  = gaussian_kde(pooled_c)
xg   = np.linspace(pooled_c.min(), pooled_c.max(), 400)
ax.plot(xg, kde(xg),       color="#4878CF", lw=2, label="KDE (residuals)")
ax.plot(xg, norm.pdf(xg),  color="crimson", lw=2, linestyle="--",
        label="N(0,1)")
ax.set_xlabel("Standardised residual", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Pooled residuals: KDE vs Normal", fontsize=11)
ax.legend(fontsize=8)

# ── Panels 2–3: skewness and excess kurtosis histograms ──────────────────────
sk_vals = np.array([skew(res_flat[:, n])     for n in range(N)])
ku_vals = np.array([kurtosis(res_flat[:, n]) for n in range(N)])  # excess

ax = axes[2]
ax.hist(sk_vals, bins=30, color="#6ACC65", edgecolor="white", alpha=0.85)
ax.axvline(0, color="crimson", lw=1.5, linestyle="--", label="Gaussian skew=0")
ax.set_xlabel("Skewness", fontsize=10)
ax.set_ylabel("Neurons", fontsize=10)
ax.set_title(f"Skewness  (mean={sk_vals.mean():.3f})", fontsize=11)
ax.legend(fontsize=8)

ax = axes[3]
ax.hist(ku_vals, bins=30, color="#D65F5F", edgecolor="white", alpha=0.85)
ax.axvline(0, color="crimson", lw=1.5, linestyle="--",
           label="Gaussian excess kurt=0")
ax.set_xlabel("Excess kurtosis", fontsize=10)
ax.set_ylabel("Neurons", fontsize=10)
ax.set_title(f"Excess Kurtosis  (mean={ku_vals.mean():.3f})", fontsize=11)
ax.legend(fontsize=8)

# ── Per-neuron KDE panels ─────────────────────────────────────────────────────
neuron_ids = np.linspace(0, N - 1, n_qq, dtype=int)
for k, nid in enumerate(neuron_ids):
    ax  = axes[4 + k]
    col = res_flat[:, nid]
    col = col[np.isfinite(col)]
    clip_n = np.percentile(np.abs(col), 99)
    col_c  = np.clip(col, -clip_n, clip_n)

    kde_n = gaussian_kde(col_c)
    xg_n  = np.linspace(col_c.min(), col_c.max(), 300)
    ax.plot(xg_n, kde_n(xg_n),      color="#4878CF", lw=1.5)
    ax.plot(xg_n, norm.pdf(xg_n),   color="crimson", lw=1.2,
            linestyle="--", alpha=0.7)
    ax.set_title(f"PC {nid+1}  BC={bc_vals[nid]:.3f}", fontsize=9)
    ax.set_xlabel("Std residual", fontsize=7)
    ax.tick_params(labelsize=7)

# hide unused axes
for k in range(4 + n_qq, len(axes)):
    axes[k].axis("off")

fig.suptitle(
    "Bimodality Check — POCO_prob Standardised Residuals\n"
    "(Bimodality Coefficient · KDE · Skewness · Kurtosis)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

out_path = "results/bimodality_check.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
plt.show()
