# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
compare_calibration.py
----------------------
Compares calibration of two probabilistic POCO variants on the held-out test set:

  1. POCO_prob     — Gaussian likelihood; model predicts μ and σ per neuron
  2. POCO_studentt — Student-t likelihood (ν estimated from residuals)

For each model, we plot:
  • Reliability diagram (empirical vs nominal coverage)
  • Expected Calibration Error (ECE)
  • Coverage vs forecast horizon at 50% and 90% confidence

Usage:
  python compare_calibration.py

  # run after training both models:
  #   python POCO_prob.py       → models/best_poco_prob.pt
  #   python POCO_studentt.py   → models/best_poco_studentt.pt
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poco_src.standalone_poco import NeuralPredictionConfig
from poco_src.POCO_prob    import ProbabilisticPOCO
from poco_src.POCO_studentt import StudentTPOCO

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/processed/0.npz"
PROB_PATH   = "models/best_poco_prob.pt"
ST_PATH     = "models/best_poco_studentt.pt"
OUT_DIR     = "results"
os.makedirs(OUT_DIR, exist_ok=True)

N_PCS       = 128
CONTEXT     = 48
PRED_LEN    = 16
TRAIN_FRAC  = 0.6
VAL_FRAC    = 0.2
BATCH_SIZE  = 64
# load ν from saved results if available, otherwise fall back to default
_st_results = "results/poco_studentt_losses.npz"
_st_files   = sorted(__import__("glob").glob("results/poco_studentt_df*_losses.npz"))
if _st_files:
    _st_results = _st_files[-1]   # most recently trained
try:
    DF = float(np.load(_st_results)["df"])
    print(f"Student-t ν loaded from {_st_results}: ν={DF:.2f}")
except Exception:
    DF = 7.0
    print(f"Student-t results not found, defaulting to ν={DF}")

ALPHAS = np.linspace(0.05, 0.95, 19)   # nominal confidence levels
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class CalciumDataset(Dataset):
    def __init__(self, traces, context_len, pred_len):
        traces = traces.astype(np.float32)
        mu = traces.mean(0, keepdims=True)
        sd = traces.std(0,  keepdims=True) + 1e-8
        traces = (traces - mu) / sd
        win = context_len + pred_len
        X, Y = [], []
        for t in range(len(traces) - win + 1):
            X.append(traces[t           : t + context_len])
            Y.append(traces[t + context_len : t + win    ])
        self.X = torch.tensor(np.array(X))
        self.Y = torch.tensor(np.array(Y))
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
raw  = data["PC"].astype(np.float32)
if raw.shape[0] < raw.shape[1]:
    raw = raw.T
raw  = raw[:, :N_PCS]
T, N = raw.shape

train_end = int(T * TRAIN_FRAC)
val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

test_ds     = CalciumDataset(raw[val_end:], CONTEXT, PRED_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Test windows: {len(test_ds)}")

# ── POCO config ───────────────────────────────────────────────────────────────
def make_config():
    cfg = NeuralPredictionConfig()
    cfg.seq_length          = CONTEXT + PRED_LEN
    cfg.pred_length         = PRED_LEN
    cfg.compression_factor  = 16
    cfg.decoder_hidden_size = 128
    cfg.conditioning_dim    = 1024
    cfg.decoder_num_layers  = 1
    cfg.decoder_num_heads   = 16
    cfg.poyo_num_latents    = 8
    return cfg

# ── Helper: compute coverage at each α ───────────────────────────────────────
def reliability(mu, sigma, y, alphas, dist="gaussian", df=7.0):
    """
    mu, sigma, y : np arrays (W, P, N)
    Returns array of empirical coverage for each nominal level in alphas.
    dist: 'gaussian' or 'studentt'
    """
    coverage = []
    for alpha in alphas:
        if dist == "gaussian":
            z = scipy.stats.norm.ppf((1 + alpha) / 2)
        else:
            z = scipy.stats.t.ppf((1 + alpha) / 2, df=df)
        lo = mu - z * sigma
        hi = mu + z * sigma
        coverage.append(((y > lo) & (y < hi)).mean())
    return np.array(coverage)

def ece(coverage, alphas):
    return np.abs(coverage - alphas).mean()

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# 1. POCO_prob — Gaussian likelihood
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(PROB_PATH):
    print(f"\nLoading POCO_prob from {PROB_PATH} …")
    prob_model = ProbabilisticPOCO(make_config(), [[N]]).to(DEVICE)
    prob_model.load_state_dict(
        torch.load(PROB_PATH, map_location=DEVICE, weights_only=False))
    prob_model.eval()

    mu_prob, sg_prob, y_prob = [], [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(DEVICE)
            dists = prob_model([X.permute(1, 0, 2)])
            d = dists[0]
            mu_prob.append(d.mean.permute(1, 0, 2).cpu().numpy())
            sg_prob.append(d.scale.permute(1, 0, 2).cpu().numpy())
            y_prob.append(Y.numpy())
    mu_prob = np.concatenate(mu_prob, axis=0)
    sg_prob = np.concatenate(sg_prob, axis=0)
    y_prob  = np.concatenate(y_prob,  axis=0)

    cov_prob = reliability(mu_prob, sg_prob, y_prob, ALPHAS, dist="gaussian")
    results["POCO_prob\n(Gaussian)"] = {
        "coverage": cov_prob,
        "ece":      ece(cov_prob, ALPHAS),
        "mu": mu_prob, "sigma": sg_prob, "y": y_prob,
        "color": "#8e44ad", "marker": "o", "dist": "gaussian",
    }
    print(f"  ECE = {ece(cov_prob, ALPHAS):.4f}")
else:
    print(f"  {PROB_PATH} not found — skipping POCO_prob")

# ─────────────────────────────────────────────────────────────────────────────
# 2. POCO_studentt — Student-t likelihood
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(ST_PATH):
    print(f"\nLoading POCO_studentt from {ST_PATH} …")
    st_model = StudentTPOCO(make_config(), [[N]], df=DF).to(DEVICE)
    st_model.load_state_dict(
        torch.load(ST_PATH, map_location=DEVICE, weights_only=False))
    st_model.eval()

    mu_st, sg_st, y_st = [], [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(DEVICE)
            dists = st_model([X.permute(1, 0, 2)])
            d = dists[0]
            mu_st.append(d.loc.permute(1, 0, 2).cpu().numpy())
            sg_st.append(d.scale.permute(1, 0, 2).cpu().numpy())
            y_st.append(Y.numpy())
    mu_st = np.concatenate(mu_st, axis=0)
    sg_st = np.concatenate(sg_st, axis=0)
    y_st  = np.concatenate(y_st,  axis=0)

    cov_st = reliability(mu_st, sg_st, y_st, ALPHAS, dist="studentt", df=DF)
    results[f"POCO_studentt\n(Student-t ν={int(DF)})"] = {
        "coverage": cov_st,
        "ece":      ece(cov_st, ALPHAS),
        "mu": mu_st, "sigma": sg_st, "y": y_st,
        "color": "#e67e22", "marker": "^", "dist": "studentt",
    }
    print(f"  ECE = {ece(cov_st, ALPHAS):.4f}")
else:
    print(f"  {ST_PATH} not found — skipping POCO_studentt")

if not results:
    print("No models found. Train at least one model first.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Reliability diagram (all models overlaid) ───────────────────────
ax_rel = fig.add_subplot(gs[0, 0])
ax_rel.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=5)
ax_rel.fill_between([0,1],[0,0],[0,1], alpha=0.04, color="red",
                    label="Overconfident (intervals too narrow)")
ax_rel.fill_between([0,1],[0,1],[1,1], alpha=0.04, color="blue",
                    label="Underconfident (intervals too wide)")

for name, r in results.items():
    label = f"{name.replace(chr(10), ' ')}  (ECE={r['ece']:.3f})"
    ax_rel.plot(ALPHAS, r["coverage"], marker=r["marker"], lw=2.0, ms=6,
                color=r["color"], label=label)

ax_rel.set_xlabel("Nominal confidence level α", fontsize=11)
ax_rel.set_ylabel("Empirical coverage", fontsize=11)
ax_rel.set_title("Reliability Diagram — Calibration Comparison", fontsize=12,
                 fontweight="bold")
ax_rel.legend(fontsize=8.5, framealpha=0.85, loc="upper left")
ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
ax_rel.grid(alpha=0.3)

# ── Panel 2: ECE bar chart ────────────────────────────────────────────────────
ax_ece = fig.add_subplot(gs[0, 1])
names  = [n.replace("\n", " ") for n in results.keys()]
eces   = [r["ece"] for r in results.values()]
colors = [r["color"] for r in results.values()]
bars   = ax_ece.bar(range(len(names)), eces, color=colors, edgecolor="white",
                    width=0.5)
for bar, val in zip(bars, eces):
    ax_ece.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_ece.set_xticks(range(len(names)))
ax_ece.set_xticklabels(names, fontsize=8)
ax_ece.set_ylabel("ECE (lower is better)", fontsize=10)
ax_ece.set_title("Expected Calibration Error", fontsize=11, fontweight="bold")
ax_ece.set_ylim(0, max(eces) * 1.3)
ax_ece.grid(axis="y", alpha=0.3)
ax_ece.spines[["top", "right"]].set_visible(False)

# ── Panels 3–4: Coverage vs horizon at 50% and 90% for each model ────────────
steps = np.arange(1, PRED_LEN + 1)
for col, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, col])
    for alpha, color_line, lbl in [(0.50, "#e8a838", "50%"),
                                    (0.90, r["color"],  "90%")]:
        if r["dist"] == "gaussian":
            z = scipy.stats.norm.ppf((1 + alpha) / 2)
        else:
            z = scipy.stats.t.ppf((1 + alpha) / 2, df=DF)
        cov_h = []
        for h in range(PRED_LEN):
            lo = r["mu"][:, h, :] - z * r["sigma"][:, h, :]
            hi = r["mu"][:, h, :] + z * r["sigma"][:, h, :]
            cov_h.append(((r["y"][:, h, :] > lo) &
                          (r["y"][:, h, :] < hi)).mean())
        ax.plot(steps, cov_h, "o-", lw=2, ms=4,
                color=color_line, label=f"Empirical {lbl}")
        ax.axhline(alpha, color=color_line, lw=1.2, ls="--", alpha=0.6,
                   label=f"Nominal {lbl}")
    ax.set_xlabel("Forecast step", fontsize=9)
    ax.set_ylabel("Coverage", fontsize=9)
    ax.set_title(name.replace("\n", "\n"), fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_xticks(steps[::2])
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

fig.suptitle(
    "Calibration Comparison: POCO_prob (Gaussian) vs POCO_studentt (Student-t)\n"
    "Reliability Diagram · ECE · Coverage vs Horizon",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()

out = os.path.join(OUT_DIR, "compare_calibration.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Model':<30}  {'ECE':>6}  {'90% cov':>8}  {'50% cov':>8}")
print("-" * 58)
for name, r in results.items():
    name_flat = name.replace("\n", " ")
    cov_90 = r["coverage"][ALPHAS >= 0.89][0]
    cov_50 = r["coverage"][ALPHAS >= 0.49][0]
    print(f"{name_flat:<30}  {r['ece']:>6.4f}  {cov_90:>8.4f}  {cov_50:>8.4f}")
