# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
estimate_df.py
--------------
Estimates the best-fit Student-t degrees of freedom (ν) from the
standardised residuals of POCO_prob using two independent methods:

  Method 1 — Kurtosis formula (closed-form):
      excess_kurtosis = 6 / (ν - 4)   for ν > 4
      → ν = 6 / excess_kurtosis + 4

  Method 2 — Maximum Likelihood Estimation (MLE):
      scipy.stats.t.fit() optimises the log-likelihood of the t-distribution
      over the residuals directly, returning (df, loc, scale).

Both methods should agree closely. The MLE estimate is more robust and is
used as the recommended value for POCO_studentt.py.

Usage:
  python estimate_df.py
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as scipy_t, kurtosis
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

res_std = ((Y - mean) / (sigma + 1e-8)).cpu().numpy().flatten()
res_std = res_std[np.isfinite(res_std)]
print(f"Pooled residuals: {len(res_std):,}")

# ── Method 1: kurtosis formula ────────────────────────────────────────────────
excess_kurt = kurtosis(res_std, fisher=True)   # excess kurtosis
# Student-t excess kurtosis = 6/(ν-4)  valid for ν > 4
if excess_kurt > 0:
    nu_kurt = 6.0 / excess_kurt + 4.0
else:
    nu_kurt = float("inf")   # kurtosis ≤ 0 → effectively Gaussian
print(f"\nMethod 1 — Kurtosis formula:")
print(f"  Excess kurtosis = {excess_kurt:.4f}")
print(f"  ν estimate      = {nu_kurt:.2f}")

# ── Method 2: MLE via scipy ───────────────────────────────────────────────────
# subsample for MLE speed (still >100k points)
mle_sample = res_std
if len(mle_sample) > 100_000:
    mle_sample = np.random.choice(mle_sample, 100_000, replace=False)

print(f"\nMethod 2 — MLE (n={len(mle_sample):,}) …")
df_mle, loc_mle, scale_mle = scipy_t.fit(mle_sample, fscale=1.0)
# fscale=1.0 because residuals are already standardised (σ≈1)
print(f"  ν (df)  = {df_mle:.2f}")
print(f"  loc     = {loc_mle:.4f}  (should be ≈ 0)")
print(f"  scale   = {scale_mle:.4f}  (should be ≈ 1)")

# ── Recommended value ─────────────────────────────────────────────────────────
nu_recommended = round(df_mle)
print(f"\n{'='*45}")
print(f"  Recommended ν for POCO_studentt : {nu_recommended}")
print(f"  (MLE estimate rounded to nearest integer)")
print(f"{'='*45}")

# ── Plot: residual KDE vs fitted distributions ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

clip = np.percentile(np.abs(res_std), 99.5)
res_c = np.clip(res_std, -clip, clip)
xg    = np.linspace(-clip, clip, 500)

from scipy.stats import norm, gaussian_kde
kde = gaussian_kde(res_c, bw_method=0.15)

# ─ Panel 1: full distribution ─────────────────────────────────────────────────
ax = axes[0]
ax.hist(res_c, bins=150, density=True,
        color="#4878CF", alpha=0.45, edgecolor="none", label="Residuals")
ax.plot(xg, kde(xg),                          color="#4878CF", lw=2,
        label="KDE")
ax.plot(xg, norm.pdf(xg),                     color="crimson",  lw=2,
        linestyle="--", label="Gaussian  N(0,1)")
ax.plot(xg, scipy_t.pdf(xg, df=nu_kurt,  loc=0, scale=1),
        color="darkorange", lw=2, linestyle="-.",
        label=f"Student-t  ν={nu_kurt:.1f}  (kurtosis)")
ax.plot(xg, scipy_t.pdf(xg, df=df_mle, loc=loc_mle, scale=scale_mle),
        color="green", lw=2, linestyle=":",
        label=f"Student-t  ν={df_mle:.1f}  (MLE)")
ax.set_xlabel("Standardised residual", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Residuals vs fitted distributions", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)

# ─ Panel 2: log-scale to highlight tails ─────────────────────────────────────
ax = axes[1]
ax.semilogy(xg, kde(xg),                          color="#4878CF", lw=2,
            label="KDE")
ax.semilogy(xg, norm.pdf(xg),                     color="crimson",  lw=2,
            linestyle="--", label="Gaussian")
ax.semilogy(xg, scipy_t.pdf(xg, df=nu_kurt,  loc=0, scale=1),
            color="darkorange", lw=2, linestyle="-.",
            label=f"Student-t ν={nu_kurt:.1f}")
ax.semilogy(xg, scipy_t.pdf(xg, df=df_mle, loc=loc_mle, scale=scale_mle),
            color="green", lw=2, linestyle=":",
            label=f"Student-t ν={df_mle:.1f} (MLE)")
ax.set_xlabel("Standardised residual", fontsize=11)
ax.set_ylabel("Density (log scale)", fontsize=11)
ax.set_title("Tail behaviour (log scale)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(-clip, clip)

fig.suptitle(
    f"Student-t Degrees of Freedom Estimation\n"
    f"Kurtosis method: ν={nu_kurt:.2f}   |   MLE: ν={df_mle:.2f}   |   "
    f"Recommended: ν={nu_recommended}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()

out = "results/estimate_df.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.show()
