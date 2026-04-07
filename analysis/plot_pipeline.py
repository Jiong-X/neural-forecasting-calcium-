"""
Implementation pipeline flowchart for the COMP0197 project.
Saves to results/plots/pipeline_flowchart.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("results/plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(18, 24))
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis("off")

# ── colour palette ──────────────────────────────────────────────────────────
C = {
    "data":        "#2471a3",   # deep blue
    "preproc":     "#1e8449",   # dark green
    "split":       "#b7950b",   # gold
    "rnn":         "#c0392b",   # red
    "linear":      "#1a5276",   # navy
    "mixer":       "#6c3483",   # violet
    "poco":        "#884ea0",   # purple
    "eval":        "#117a65",   # teal
    "uncert":      "#784212",   # brown
    "output":      "#2e4057",   # dark slate
    "arrow":       "#555555",
    "bg":          "#f8f9fa",
}

def box(ax, x, y, w, h, text, color, fontsize=10, text_color="white",
        style="round,pad=0.1", alpha=1.0, bold=False):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=style,
                           facecolor=color, edgecolor="white",
                           linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight=weight, zorder=4,
            multialignment="center")

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def label_arrow(ax, x, y, text, fontsize=8, color="#444444"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic", zorder=5)

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=9, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                      edgecolor="none", alpha=0.15), zorder=5)

# ============================================================
# TITLE
# ============================================================
ax.text(9, 23.4, "COMP0197 Project — Implementation Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#1a1a2e")
ax.text(9, 23.0, "Neural Activity Forecasting from Zebrafish Calcium Imaging",
        ha="center", va="center", fontsize=11, color="#555555")

# ============================================================
# STAGE 1 — RAW DATA
# ============================================================
box(ax, 9, 22.2, 8.5, 0.75,
    "RAW DATA  |  4 Zebrafish Subjects  (TimeSeries.h5)\n"
    "CellResp: (T × N_neurons) fluorescence traces  •  ~2500–3500 frames",
    C["data"], fontsize=10, bold=True)

arrow(ax, 9, 21.82, 9, 21.38)

# ============================================================
# STAGE 2 — PREPROCESSING  (preprocess.py)
# ============================================================
section_label(ax, 0.3, 21.1, "① preprocess.py", C["preproc"])

box(ax, 9, 21.1, 8.5, 0.75,
    "PREPROCESSING  (per subject)\n"
    "Z-score per neuron  →  PCA (sklearn, SVD)  →  Top 128 PCs\n"
    "Save: data/processed/{0,1,2,3}.npz  [keys: PC, M, valid_indices]",
    C["preproc"], fontsize=9.5)

arrow(ax, 9, 20.72, 9, 20.28)

# ============================================================
# STAGE 3 — SLIDING WINDOW DATASET
# ============================================================
box(ax, 9, 20.0, 8.5, 0.65,
    "SLIDING WINDOW DATASET   (SessionDataset)\n"
    "Context window: 48 frames   →   Prediction horizon: 16 frames   |   Batch size: 16",
    C["split"], fontsize=9.5, text_color="#1a1a1a")

# fork arrow down-left and down-right
arrow(ax, 9, 19.67, 4.5, 19.15)
arrow(ax, 9, 19.67, 13.5, 19.15)
label_arrow(ax, 6.2, 19.45, "Subjects 0, 1, 2  (train)")
label_arrow(ax, 12.1, 19.45, "Subject 3  (val / cross-animal)")

# ============================================================
# STAGE 4 — MODEL BRANCHES
# ============================================================
section_label(ax, 0.3, 18.85, "② Models", C["rnn"])

# ── SINGLE-SESSION block ────────────────────────────────────
box(ax, 4.5, 18.85, 8.5, 0.55,
    "SINGLE-SESSION MODELS   (Subjects 0 train  |  Subject 0 val  —  80/20 temporal split)",
    C["split"], fontsize=9, text_color="#1a1a1a", alpha=0.85)

# Row A — recurrent
arrow(ax, 4.5, 18.57, 2.2, 18.05)
arrow(ax, 4.5, 18.57, 4.5, 18.05)

box(ax, 2.2, 17.75, 3.6, 0.55,
    "Vanilla RNN\n(nn.RNN, tanh, 1 layer)", C["rnn"], fontsize=8.5)
box(ax, 4.5, 17.75, 2.7, 0.55,   # Moved right slightly to avoid LSTM overlap
    "LSTM\n(1 layer, hidden=256)", C["rnn"], fontsize=8.5)

# Row B — linear
arrow(ax, 4.5, 18.57, 6.7, 18.05)
arrow(ax, 4.5, 18.57, 8.9, 18.05)

box(ax, 6.7, 17.75, 2.7, 0.55,
    "NLinear\n(subtract last val)", C["linear"], fontsize=8.5)
box(ax, 8.9, 17.75, 2.7, 0.55,
    "DLinear\n(trend + seasonal)", C["linear"], fontsize=8.5)

# Row C — advanced
arrow(ax, 4.5, 18.57, 2.2, 16.95)   # will be overridden; use indirect

# TSMixer & POCO side by side below
arrow(ax, 2.2, 17.47, 2.2, 16.95)
arrow(ax, 4.5, 17.47, 4.5, 16.95)
arrow(ax, 6.7, 17.47, 6.7, 16.95)
arrow(ax, 8.9, 17.47, 6.0, 16.95)   # DLinear feeds down to TSMixer area

# reposition: use a convergence connector
for sx in [2.2, 4.5, 6.7, 8.9]:
    arrow(ax, sx, 17.47, sx, 17.22)
# horizontal connector
ax.plot([2.2, 8.9], [17.22, 17.22], color=C["arrow"], lw=1.4, zorder=2)
ax.plot([4.5, 4.5], [17.22, 16.95], color=C["arrow"], lw=1.4, zorder=2)   # dummy; real below

# Cleaner: just draw arrows from single-session block direct
arrow(ax, 3.35, 18.57, 3.35, 16.98)   # TSMixer lane

box(ax, 3.35, 16.70, 2.7, 0.55,
    "TSMixer\n(MLP time+feat mixing)", C["mixer"], fontsize=8.5)
box(ax, 6.2,  16.70, 2.7, 0.55,
    "POCO (det.)\nPerceiver-IO, MSE", C["poco"], fontsize=8.5)
box(ax, 6.2,  15.85, 2.7, 0.65,
    "POCO (prob.)\nGaussian head\nμ, σ  |  NLL loss", C["poco"], fontsize=8.5)

arrow(ax, 6.2, 16.42, 6.2, 16.17)
label_arrow(ax, 7.45, 16.29, "+Gaussian output head")

# AR off to the side
arrow(ax, 4.5, 18.57, 0.85, 17.75)
box(ax, 0.85, 17.48, 1.5, 0.55,
    "AR\n(OLS)", "#7f8c8d", fontsize=8.5)
ax.text(0.85, 17.15, "closed-form\nno training loop",
        ha="center", va="top", fontsize=7, color="#888888", style="italic")

# ── MULTI-SESSION block ─────────────────────────────────────
box(ax, 13.5, 18.85, 8.0, 0.55,
    "MULTI-SESSION MODELS   (Subjects 0,1,2 train  |  Subject 3 val  —  cross-animal)",
    C["split"], fontsize=9, text_color="#1a1a1a", alpha=0.85)

arrow(ax, 13.5, 18.57, 12.3, 17.75)
arrow(ax, 13.5, 18.57, 14.7, 17.75)

box(ax, 12.3, 17.48, 3.6, 0.65,
    "POCO multi-session (det.)\nMultiSessionLoader\ninput_size=[[128],[128],[128]]",
    C["poco"], fontsize=8.5)
box(ax, 14.7, 17.00, 3.6, 0.85,
    "POCO multi-session (prob.)\nGaussian head\nNLL loss\nCross-animal val",
    C["poco"], fontsize=8.5)

arrow(ax, 12.3, 17.15, 12.3, 16.42)
arrow(ax, 14.7, 16.57, 14.7, 16.42)

# ============================================================
# STAGE 5 — EVALUATION
# ============================================================
section_label(ax, 0.3, 16.0, "③ Evaluation", C["eval"])

# join all models to eval box
ax.plot([3.35, 14.7], [15.85, 15.85], color="#aaaaaa", lw=1.2,
        ls="--", zorder=1)
# arrows from model rows
arrow(ax, 3.35,  16.42, 3.35,  15.58)
arrow(ax, 6.2,   15.52, 6.2,   15.35)
arrow(ax, 12.3,  16.12, 12.3,  15.58)
arrow(ax, 14.7,  15.60, 14.7,  15.35)

box(ax, 9, 15.15, 8.5, 0.65,
    "EVALUATION  —  val split / held-out subject\n"
    "MAE  |  MSE  (deterministic models)   •   NLL  (probabilistic models)",
    C["eval"], fontsize=9.5, bold=True)

arrow(ax, 9, 14.82, 9, 14.38)

# ============================================================
# STAGE 6 — UNCERTAINTY QUANTIFICATION
# ============================================================
section_label(ax, 0.3, 14.1, "④ Uncertainty (prob. POCO only)", C["uncert"])

box(ax, 9, 14.1, 8.5, 0.65,
    "UNCERTAINTY QUANTIFICATION   (MC Dropout, T = 50–100 passes)\n"
    "Enable dropout at inference  •  collect {μ_t, σ_t}  for t = 1…T",
    C["uncert"], fontsize=9.5)

# split into two sub-boxes
arrow(ax, 9, 13.77, 6.5, 13.25)
arrow(ax, 9, 13.77, 11.5, 13.25)

box(ax, 6.5, 13.0, 5.5, 0.65,
    "Aleatoric  =  √E_t[σ²_t]\n(irreducible noise in data)",
    "#b9770e", fontsize=9)
box(ax, 11.5, 13.0, 5.5, 0.65,
    "Epistemic  =  √Var_t[μ_t]\n(model uncertainty)",
    "#6c3483", fontsize=9)

arrow(ax, 6.5,  12.67, 9, 12.22)
arrow(ax, 11.5, 12.67, 9, 12.22)

box(ax, 9, 12.0, 8.5, 0.55,
    "Total  =  √( E[σ²] + Var[μ] )      [law of total variance — Kendall & Gal 2017]",
    C["output"], fontsize=9)

arrow(ax, 9, 11.72, 9, 11.38)

# ── MC Dropout comparison ───────────────────────────────────
box(ax, 9, 11.15, 8.5, 0.55,
    "Dropout comparison:  Original (lin=0.4, epistemic ratio ≈ 7.5%)   vs   High-dropout (lin=0.6, ratio ≈ 13%)",
    C["uncert"], fontsize=8.5, alpha=0.85)

arrow(ax, 9, 10.87, 9, 10.43)

# ============================================================
# STAGE 7 — OUTPUTS / VISUALISATIONS
# ============================================================
section_label(ax, 0.3, 10.15, "⑤ Outputs", C["output"])

box(ax, 9, 10.15, 8.5, 0.65,
    "SAVED OUTPUTS   (models/  •  results/  •  results/plots/)",
    C["output"], fontsize=9.5, bold=True)

# fan out to 4 output types
for xi, text in [
    (2.5,  "Model weights\n.pt checkpoint\n(best val loss)"),
    (6.2,  "Loss curves\n.npz arrays\n(train / val)"),
    (11.0, "MAE/MSE/NLL\ncomparison\nbar charts"),
    (15.2, "Uncertainty bands\nMC Dropout plots\naleatoric vs epistemic"),
]:
    arrow(ax, 9, 9.82, xi, 9.25)
    box(ax, xi, 8.95, 3.3, 0.70, text, C["output"], fontsize=8.5, alpha=0.80)

# ============================================================
# LEGEND
# ============================================================
legend_items = [
    (C["data"],    "Raw / Processed Data"),
    (C["preproc"], "Preprocessing"),
    (C["split"],   "Dataset / Split"),
    (C["rnn"],     "Recurrent Models (RNN, LSTM)"),
    (C["linear"],  "Linear Models (NLinear, DLinear)"),
    (C["mixer"],   "MLP Mixer (TSMixer)"),
    (C["poco"],    "POCO variants"),
    (C["eval"],    "Evaluation"),
    (C["uncert"],  "Uncertainty Quantification"),
    (C["output"],  "Output / Visualisation"),
]
lx, ly = 0.3, 8.1
ax.text(lx, ly, "Legend", fontsize=9, fontweight="bold", color="#333333")
for i, (c, label) in enumerate(legend_items):
    row = i // 2
    col = i % 2
    px = lx + col * 4.8
    py = ly - 0.38 * (row + 1)
    patch = FancyBboxPatch((px, py - 0.13), 0.5, 0.26,
                           boxstyle="round,pad=0.05",
                           facecolor=c, edgecolor="none")
    ax.add_patch(patch)
    ax.text(px + 0.65, py, label, fontsize=8, va="center", color="#333333")

# background panel
bg = FancyBboxPatch((0.1, 7.5), 9.8, 0.9,
                    boxstyle="round,pad=0.1",
                    facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=1,
                    zorder=0, alpha=0.6)
ax.add_patch(bg)

# ── Final note ───────────────────────────────────────────────
ax.text(9, 7.35,
        "Environments:  poco (POCO models)  •  comp0197-pt (RNN/LSTM)     "
        "Framework: PyTorch  •  sklearn PCA  •  MC Dropout for uncertainty",
        ha="center", va="center", fontsize=8, color="#666666",
        style="italic")

plt.tight_layout()
out = "results/plots/pipeline_flowchart.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.close()
