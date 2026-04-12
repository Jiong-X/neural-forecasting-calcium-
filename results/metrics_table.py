# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
metrics_table.py
----------------
Produces a clean comparison table (saved as PNG) of test-set metrics for:
  - Probabilistic POCO  (results/train_losses.npz)
  - Deterministic POCO  (results/DeterministicPoco_train_losses.npz)
  - MLP ablation        (results/MLP_train_losses.npz)

Usage (run from repo root):
    python results/metrics_table.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGURES_DIR = "results/figures"

# ---------------------------------------------------------------------------
# Load saved test metrics
# ---------------------------------------------------------------------------
prob = np.load("results/train_losses.npz")
det  = np.load("results/DeterministicPoco_train_losses.npz")
mlp  = np.load("results/MLP_train_losses.npz")

# ---------------------------------------------------------------------------
# Assemble table data
# ---------------------------------------------------------------------------
rows = [
    ("Prob POCO",  f"{prob['test_MAE']:.4f}", f"{prob['test_MSE']:.4f}",
                   f"{prob['test_RMSE']:.4f}", f"{float(prob['test_GNLL']):.4f}"),
    ("Det POCO",   f"{det['test_MAE']:.4f}",  f"{det['test_MSE']:.4f}",
                   f"{det['test_RMSE']:.4f}",  "N/A"),
    ("MLP (ablation)", f"{mlp['test_MAE']:.4f}", f"{mlp['test_MSE']:.4f}",
                   f"{mlp['test_RMSE']:.4f}", f"{float(mlp['test_GNLL']):.4f}"),
]

col_labels = ["Model", "MAE", "MSE", "RMSE", "GNLL"]

# ---------------------------------------------------------------------------
# Print to terminal
# ---------------------------------------------------------------------------
col_widths_terminal = [18, 8, 8, 8, 10]
header = "  ".join(f"{h:<{w}}" for h, w in zip(col_labels, col_widths_terminal))
print(header)
print("-" * len(header))
for row in rows:
    print("  ".join(f"{v:<{w}}" for v, w in zip(row, col_widths_terminal)))

# ---------------------------------------------------------------------------
# Render table as a figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.axis("off")

cell_data  = [list(r) for r in rows]
col_widths = [0.28, 0.18, 0.18, 0.18, 0.18]

table = ax.table(
    cellText    = cell_data,
    colLabels   = col_labels,
    cellLoc     = "center",
    loc         = "center",
    colWidths   = col_widths,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row shading
row_colors = ["#f7f9fc", "#eaf0fb"]
for i, row in enumerate(rows):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(row_colors[i % 2])
        table[i + 1, j].set_edgecolor("#cccccc")

ax.set_title("Test-set metrics — Prob POCO vs Det POCO vs MLP ablation",
             fontsize=12, fontweight="bold", pad=14)

plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
out = os.path.join(FIGURES_DIR, "metrics_table.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
