# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
normality_pooled.py
-------------------
Pooled normality diagnostics for POCO_prob standardised residuals.

All 128 PCs × all prediction steps are pooled into one sample and tested
together, giving a single global picture of the Gaussian assumption.

Tests:
  1. Q-Q plot               — quantile alignment with N(0,1)
  2. Residual histogram      — KDE overlay vs N(0,1)
  3. D'Agostino-Pearson test — omnibus test combining skewness + kurtosis
  4. Skewness test           — scipy.stats.skewtest
  5. Kurtosis test           — scipy.stats.kurtosistest

Usage:
  python normality_pooled.py
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (probplot, norm, gaussian_kde,
                         skewtest, kurtosistest, normaltest, skew, kurtosis)
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import get_test_dataset
from src.model   import ProbabilisticForecaster

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="models/saved/model.pt")
parser.add_argument("--n_samples",  type=int, default=2000)
parser.add_argument("--context",    type=int, default=48)
parser.add_argument("--pred",       type=int, default=16)
parser.add_argument("--n_channels", type=int, default=128)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model …")
model = ProbabilisticForecaster(
    seq_length  = args.context + args.pred,
    pred_length = args.pred,
    n_channels  = args.n_channels,
).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE,
                                 weights_only=False))
model.eval()

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test dataset …")
test_ds = get_test_dataset(seq_length=args.context + args.pred,
                           pred_length=args.pred)
idx = np.random.choice(len(test_ds), min(args.n_samples, len(test_ds)),
                       replace=False)
X = torch.stack([test_ds[i][0] for i in idx]).to(DEVICE)
Y = torch.stack([test_ds[i][1] for i in idx]).to(DEVICE)

# ── Forward pass → standardised residuals ────────────────────────────────────
print("Running forward pass …")
with torch.no_grad():
    mean, logvar = model(X)
sigma = (0.5 * logvar).exp()

res_std = ((Y - mean) / (sigma + 1e-8)).cpu().numpy()   # (S, P, N)
pooled  = res_std.flatten()
pooled  = pooled[np.isfinite(pooled)]
n_obs   = len(pooled)
print(f"Pooled residuals: {n_obs:,} observations")

# ── Statistical tests ─────────────────────────────────────────────────────────
sk_val  = skew(pooled)
ku_val  = kurtosis(pooled, fisher=True)    # excess kurtosis (Gaussian = 0)

sk_stat,  sk_p  = skewtest(pooled)
ku_stat,  ku_p  = kurtosistest(pooled)
omni_stat, omni_p = normaltest(pooled)     # D'Agostino-Pearson omnibus

print(f"\n{'='*55}")
print(f"  Pooled standardised residuals  (n = {n_obs:,})")
print(f"{'='*55}")
print(f"  Skewness            : {sk_val:+.4f}  (Gaussian = 0)")
print(f"  Excess kurtosis     : {ku_val:+.4f}  (Gaussian = 0)")
print(f"  Skewness test       : stat={sk_stat:+.2f},  p={sk_p:.2e}")
print(f"  Kurtosis test       : stat={ku_stat:+.2f},  p={ku_p:.2e}")
print(f"  D'Agostino-Pearson  : stat={omni_stat:.2f},   p={omni_p:.2e}")
print(f"{'='*55}")

# ── Figure: 3 panels in a single row ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# clip for display only (does not affect tests)
clip = np.percentile(np.abs(pooled), 99.5)
pooled_c = np.clip(pooled, -clip, clip)

# ─ Panel 1: Histogram + KDE + N(0,1) ─────────────────────────────────────────
ax = axes[0]
ax.hist(pooled_c, bins=120, density=True,
        color="#4878CF", alpha=0.55, edgecolor="none", label="Residuals")
kde  = gaussian_kde(pooled_c, bw_method=0.15)
xg   = np.linspace(pooled_c.min(), pooled_c.max(), 500)
ax.plot(xg, kde(xg),      color="#4878CF", lw=2,   label="KDE")
ax.plot(xg, norm.pdf(xg), color="crimson", lw=2,
        linestyle="--", label="N(0,1)")
ax.set_xlabel("Standardised residual  (y − μ) / σ", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Pooled Residual Distribution", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

# annotate skewness + kurtosis on plot
ax.text(0.03, 0.97,
        f"Skewness = {sk_val:+.3f}\nExcess kurtosis = {ku_val:+.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ─ Panel 2: Q-Q plot ──────────────────────────────────────────────────────────
ax = axes[1]
# subsample for Q-Q readability (probplot is slow with 1M+ points)
qq_sample = pooled_c
if len(qq_sample) > 20000:
    qq_sample = np.random.choice(qq_sample, 20000, replace=False)

(osm, osr), (slope, intercept, r) = probplot(qq_sample, dist="norm")
ax.scatter(osm, osr, s=1.5, alpha=0.3, color="#4878CF", rasterized=True)
ax.plot(osm, slope * np.array(osm) + intercept,
        color="crimson", lw=2, label=f"Reference line  R²={r**2:.4f}")
ax.set_xlabel("Theoretical quantiles  N(0,1)", fontsize=11)
ax.set_ylabel("Sample quantiles", fontsize=11)
ax.set_title("Q-Q Plot  (Pooled Residuals)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

# ─ Panel 3: Test summary table ────────────────────────────────────────────────
ax = axes[2]
ax.axis("off")

rows = [
    ["Statistic",          "Value",          "Interpretation"],
    ["Skewness",           f"{sk_val:+.4f}",  "≈0 → symmetric"],
    ["Excess kurtosis",    f"{ku_val:+.4f}",  ">0 → heavy tails"],
    ["Skewness test p",    f"{sk_p:.2e}",     "p<0.05 → asymmetric"],
    ["Kurtosis test p",    f"{ku_p:.2e}",     "p<0.05 → non-Gaussian tails"],
    ["D'Agostino p",       f"{omni_p:.2e}",   "p<0.05 → non-normal"],
    ["n (pooled)",         f"{n_obs:,}",      "large n → high power"],
]

col_widths = [0.38, 0.28, 0.34]
col_x      = [0.01, 0.39, 0.67]
row_h      = 0.12
y0         = 0.92

for r_idx, row in enumerate(rows):
    y = y0 - r_idx * row_h
    for c_idx, cell in enumerate(row):
        weight = "bold" if r_idx == 0 else "normal"
        color  = "#e8f0fe" if r_idx == 0 else ("white" if r_idx % 2 == 0 else "#f7f7f7")
        ax.text(col_x[c_idx], y, cell,
                transform=ax.transAxes,
                fontsize=9, fontweight=weight, va="top",
                bbox=dict(boxstyle="square,pad=0.2",
                          facecolor=color, edgecolor="#cccccc", lw=0.5))

ax.set_title("Test Summary", fontsize=12, fontweight="bold")

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Pooled Normality Diagnostics — POCO_prob Standardised Residuals\n"
    f"All 128 PCs × all prediction steps pooled  (n = {n_obs:,})",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()

out = "results/normality_pooled.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.show()
