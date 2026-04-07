"""
Focused comparison: NLinear, TSMixer, POCO (det.), POCO (prob.)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Load results directly from saved .npz files
# ---------------------------------------------------------------------------
def best_mae_mse(path):
    """Return (best_val_mae, best_val_mse) from a losses .npz file."""
    if not os.path.exists(path):
        return np.nan, np.nan
    r = np.load(path)
    mae = float(np.min(r["val_maes"])) if "val_maes" in r else np.nan
    mse = float(np.min(r["val_mses"])) if "val_mses" in r else np.nan
    return mae, mse

nlinear_mae,   nlinear_mse   = best_mae_mse("results/nlinear_losses.npz")
tsmixer_mae,   tsmixer_mse   = best_mae_mse("results/tsmixer_losses.npz")
poco_mae,      poco_mse      = best_mae_mse("results/poco_losses.npz")
poco_prob_mae, poco_prob_mse = best_mae_mse("results/poco_prob_losses.npz")
# prob model trained with NLL — MSE not meaningful
poco_prob_mse = np.nan

MODELS  = ["NLinear", "TSMixer", "POCO\n(det.)", "POCO\n(prob.)"]
MAE     = np.array([nlinear_mae, tsmixer_mae, poco_mae, poco_prob_mae])
MSE     = np.array([nlinear_mse, tsmixer_mse, poco_mse, poco_prob_mse])
COLORS  = ["#e8a838", "#b94fb5", "#7d3c98", "#6c3483"]

print("Loaded results:")
for m, a, s in zip(MODELS, MAE, MSE):
    s_str = f"{s:.4f}" if not np.isnan(s) else "N/A"
    print(f"  {m.replace(chr(10), ' '):<18}  MAE={a:.4f}  MSE={s_str}")

N = len(MODELS)
x = np.arange(N)


# ---------------------------------------------------------------------------
# Figure 1 — MAE bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for xi, v, c in zip(x, MAE, COLORS):
    ax.bar(xi, v, color=c, width=0.5, zorder=3, edgecolor="white", linewidth=0.8)
    ax.text(xi, v + 0.008, f"{v:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# % improvement vs NLinear baseline
baseline = MAE[0]
for i, v in enumerate(MAE):
    if i == 0:
        continue
    pct = (baseline - v) / baseline * 100
    ax.text(i, 0.01, f"-{pct:.1f}%",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=11)
ax.set_ylabel("Validation MAE  (z-score units)", fontsize=11)
ax.set_title("Model Comparison — Validation MAE\n"
             "C=48, P=16  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, max(MAE) * 1.22)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = "results/plots/focused_mae.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Side-by-side MAE + MSE
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax_i, (vals, ylabel, title) in enumerate([
    (MAE, "MAE  (z-score units)", "Validation MAE"),
    (MSE, "MSE  (z-score units)", "Validation MSE"),
]):
    ax = axes[ax_i]
    for xi, v, c in zip(x, vals, COLORS):
        if np.isnan(v):
            ax.bar(xi, 0.02, color="#cccccc", width=0.5, zorder=3,
                   edgecolor="white", linewidth=0.8)
            ax.text(xi, 0.04, "N/A\n(NLL loss)", ha="center", va="bottom",
                    fontsize=8.5, color="#888888")
        else:
            ax.bar(xi, v, color=c, width=0.5, zorder=3,
                   edgecolor="white", linewidth=0.8)
            ax.text(xi, v + 0.01, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    valid = vals[~np.isnan(vals)]
    ax.set_ylim(0, valid.max() * 1.25)

fig.suptitle("Model Comparison — Validation Metrics\n"
             "C=48, P=16  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
out2 = "results/plots/focused_full.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Ranked dot plot
# ---------------------------------------------------------------------------
order         = np.argsort(MAE)
sorted_labels = [MODELS[i].replace("\n", " ") for i in order]
sorted_mae    = MAE[order]
sorted_colors = [COLORS[i] for i in order]

fig, ax = plt.subplots(figsize=(7, 4))
for i, (v, c) in enumerate(zip(sorted_mae, sorted_colors)):
    ax.hlines(i, 0, v, colors="#dddddd", linewidth=2.5, zorder=1)
    ax.scatter(v, i, color=c, s=150, zorder=3,
               edgecolors="white", linewidths=0.8)
    ax.text(v + 0.005, i, f"{v:.4f}", va="center",
            fontsize=10, fontweight="bold")

ax.set_yticks(range(N))
ax.set_yticklabels(sorted_labels, fontsize=10)
ax.set_xlabel("Validation MAE  (z-score units)", fontsize=11)
ax.set_title("Model Ranking by Validation MAE  (lower is better)",
             fontsize=11, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_xlim(0, sorted_mae.max() * 1.25)

plt.tight_layout()
out3 = "results/plots/focused_ranked.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved {out3}")
plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nSummary:")
print(f"{'Model':<20} {'MAE':>8} {'MSE':>8}")
print("-" * 38)
for m, a, s in zip(MODELS, MAE, MSE):
    s_str = f"{s:.4f}" if not np.isnan(s) else "   N/A"
    print(f"{m.replace(chr(10), ' '):<20} {a:>8.4f} {s_str:>8}")

