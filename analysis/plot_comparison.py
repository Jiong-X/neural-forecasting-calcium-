"""
plot_comparison.py
------------------
MAE comparison: POCO (deterministic) vs POCO_prob (probabilistic).
Reads results directly from saved .npz files.
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
def best_mae(path):
    if not os.path.exists(path):
        return np.nan
    r = np.load(path)
    return float(np.min(r["val_maes"])) if "val_maes" in r else np.nan

poco_mae      = best_mae("results/poco_losses.npz")
poco_prob_mae = best_mae("results/poco_prob_losses.npz")

MODELS = ["POCO\n(det.)", "POCO\n(prob.)"]
MAE    = np.array([poco_mae, poco_prob_mae])
COLORS = ["#7d3c98", "#6c3483"]

print(f"POCO (det.)  val MAE: {poco_mae:.4f}")
print(f"POCO (prob.) val MAE: {poco_prob_mae:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))

bars = ax.bar(MODELS, MAE, color=COLORS, width=0.4, edgecolor="white")
for bar, val in zip(bars, MAE):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

ax.set_ylabel("Best Val MAE (z-score units)", fontsize=11)
ax.set_title("POCO Deterministic vs Probabilistic\nValidation MAE Comparison",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(v for v in MAE if not np.isnan(v)) * 1.2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = "results/plots/comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out}")
plt.show()
