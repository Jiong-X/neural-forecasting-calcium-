"""
Ablation comparison: POCO (prob.) vs MLP-only head.

Both models are trained with Gaussian NLL so metrics are directly comparable.
Plots:
  1. Training curves — NLL over epochs
  2. Bar chart — best val NLL and val MAE
  3. Test set NLL and MAE
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
def load(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — run the model first.")
        return {}
    return dict(np.load(path))

poco = load("results/train_losses.npz")
mlp  = load("results/mlp_losses.npz")

MODELS  = ["POCO (prob.)", "MLP only"]
COLORS  = ["#6c3483", "#2e86c1"]
results = [poco, mlp]

# ---------------------------------------------------------------------------
# Figure 1 — Training curves (val NLL per epoch)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for res, label, color in zip(results, MODELS, COLORS):
    if "val_nlls" in res:
        ax.plot(res["val_nlls"], label=label, color=color, linewidth=2)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Validation NLL  (nats)", fontsize=11)
ax.set_title("Ablation: POCO (prob.) vs MLP-only — Validation NLL per Epoch\n"
             "C=48, P=16  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = "results/plots/ablation_curves.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Bar chart: best val NLL and val MAE side by side
# ---------------------------------------------------------------------------
val_nll = [float(np.min(r["val_nlls"])) if "val_nlls" in r else np.nan for r in results]
val_mae = [float(np.min(r["val_maes"])) if "val_maes" in r else np.nan for r in results]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, vals, ylabel, title in zip(
    axes,
    [val_nll, val_mae],
    ["NLL  (nats)", "MAE  (z-score units)"],
    ["Best Validation NLL", "Best Validation MAE"],
):
    x = np.arange(len(MODELS))
    for xi, v, c in zip(x, vals, COLORS):
        ax.bar(xi, v, color=c, width=0.45, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(xi, v + abs(v) * 0.02, f"{v:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # % change relative to POCO
    if not np.isnan(vals[0]) and not np.isnan(vals[1]):
        pct = (vals[1] - vals[0]) / abs(vals[0]) * 100
        sign = "+" if pct > 0 else ""
        ax.text(1, vals[1] * 0.5, f"{sign}{pct:.1f}%\nvs POCO",
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    valid = [v for v in vals if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.25)

fig.suptitle("Ablation: POCO (prob.) vs MLP-only\n"
             "C=48, P=16  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
out2 = "results/plots/ablation_bars.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Test set results
# ---------------------------------------------------------------------------
test_nll = [float(r["test_nll"]) if "test_nll" in r else np.nan for r in results]
test_mae = [float(r["test_mae"]) if "test_mae" in r else np.nan for r in results]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, vals, ylabel, title in zip(
    axes,
    [test_nll, test_mae],
    ["NLL  (nats)", "MAE  (z-score units)"],
    ["Test NLL", "Test MAE"],
):
    x = np.arange(len(MODELS))
    for xi, v, c in zip(x, vals, COLORS):
        if np.isnan(v):
            ax.bar(xi, 0, color="#cccccc", width=0.45, zorder=3)
            ax.text(xi, 0.01, "N/A", ha="center", va="bottom",
                    fontsize=10, color="#888888")
        else:
            ax.bar(xi, v, color=c, width=0.45, zorder=3,
                   edgecolor="white", linewidth=0.8)
            ax.text(xi, v + abs(v) * 0.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    valid = [v for v in vals if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.25)

fig.suptitle("Ablation: POCO (prob.) vs MLP-only — Test Set\n"
             "C=48, P=16  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
out3 = "results/plots/ablation_test.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved {out3}")
plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\nAblation Summary:")
print(f"{'Model':<18} {'Val NLL':>10} {'Val MAE':>10} {'Test NLL':>10} {'Test MAE':>10}")
print("-" * 60)
for label, vn, vm, tn, tm in zip(MODELS, val_nll, val_mae, test_nll, test_mae):
    fmt = lambda v: f"{v:>10.4f}" if not np.isnan(v) else f"{'N/A':>10}"
    print(f"{label:<18}{fmt(vn)}{fmt(vm)}{fmt(tn)}{fmt(tm)}")
