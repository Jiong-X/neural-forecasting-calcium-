"""
Compare aleatoric vs epistemic uncertainty between:
  - Original POCO_prob   (lin_dropout=0.4, atn_dropout=0.0)
  - High-dropout POCO    (lin_dropout=0.6, ffn_dropout=0.3, atn_dropout=0.0)
"""

import sys, os

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poco_src.prob import ProbabilisticPOCO, CalciumDataset
from src.poco_src.standalone_poco import NeuralPredictionConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "data/processed/0.npz"
OUT_DIR     = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

N_PCS       = 128
SEQ_LEN     = 64
PRED_LEN    = 16
CONTEXT_LEN = SEQ_LEN - PRED_LEN
TRAIN_FRAC  = 0.6
VAL_FRAC    = 0.2
MC_SAMPLES  = 100
EXAMPLE     = 0
N_PANELS    = 4

MODELS = {
    "Original\n(lin=0.4)":       "models/best_poco_prob.pt",
    "High dropout\n(lin=0.6)":   "models/best_poco_prob_highdrop.pt",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data   = np.load(DATA_PATH)
raw    = data["PC"].astype(np.float32)
if raw.shape[0] < raw.shape[1]:
    raw = raw.T
traces = raw[:, :N_PCS]
T, N    = traces.shape

# z-score each neuron over the full recording before splitting
mu      = traces.mean(0, keepdims=True)
sd      = traces.std(0,  keepdims=True) + 1e-8
traces  = (traces - mu) / sd

val_end = int(T * (TRAIN_FRAC + VAL_FRAC))
val_ds  = CalciumDataset(traces[val_end:], context_len=CONTEXT_LEN, pred_len=PRED_LEN)
loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

X, Y   = next(iter(loader))
X, Y   = X.to(device), Y.to(device)
x_list = [X.permute(1, 0, 2)]
Y_np   = Y.permute(1, 0, 2).cpu().numpy()
X_np   = X.permute(1, 0, 2).cpu().numpy()

# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------
def make_config():
    cfg = NeuralPredictionConfig()
    cfg.seq_length             = SEQ_LEN
    cfg.pred_length            = PRED_LEN
    cfg.compression_factor     = 16
    cfg.poyo_num_latents       = 8
    cfg.decoder_hidden_size    = 64
    cfg.conditioning_dim       = 128
    cfg.decoder_num_layers     = 1
    cfg.decoder_num_heads      = 8
    cfg.decoder_context_length = None
    cfg.freeze_backbone        = False
    cfg.freeze_conditioned_net = False
    return cfg

def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def patch_lin_dropout(model, p):
    for mod in model.modules():
        if isinstance(mod, nn.Dropout) and abs(mod.p - 0.4) < 1e-6:
            mod.p = p

# ---------------------------------------------------------------------------
# Run MC Dropout for each model
# ---------------------------------------------------------------------------
results = {}

for label, path in MODELS.items():
    print(f"\nLoading {label.strip()} from {path} ...")
    model = ProbabilisticPOCO(make_config(), [[N]]).to(device)

    if "High" in label:
        patch_lin_dropout(model, 0.6)
        for mod in model.modules():
            if isinstance(mod, nn.Dropout) and abs(mod.p - 0.2) < 1e-6:
                mod.p = 0.3

    _ckpt = torch.load(path, map_location=device, weights_only=True)
    if all(k.startswith("poco.") for k in _ckpt):
        _ckpt = {k[len("poco."):]: v for k, v in _ckpt.items()}
    model.load_state_dict(_ckpt)
    model.eval()
    model.apply(enable_dropout)

    mus_mc, vars_mc = [], []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            d = model(x_list)[0]
            mus_mc.append(d.mean.cpu().numpy())
            vars_mc.append(d.scale.cpu().numpy() ** 2)

    mus_mc  = np.stack(mus_mc)    # (MC, pred_len, B, N)
    vars_mc = np.stack(vars_mc)

    results[label] = {
        "mu":         mus_mc.mean(0),
        "aleatoric":  np.sqrt(vars_mc.mean(0)),
        "epistemic":  np.sqrt(mus_mc.var(0)),
        "total":      np.sqrt(vars_mc.mean(0) + mus_mc.var(0)),
    }

    al = results[label]["aleatoric"][:, EXAMPLE, 0].mean()
    ep = results[label]["epistemic"][:, EXAMPLE, 0].mean()
    print(f"  PC0 — aleatoric={al:.4f}  epistemic={ep:.4f}  ratio={ep/al:.3f}")

# ---------------------------------------------------------------------------
# Figure 1: side-by-side uncertainty bands (PC 0 only, both models)
# ---------------------------------------------------------------------------
t_ctx  = np.arange(CONTEXT_LEN)
t_pred = np.arange(CONTEXT_LEN, CONTEXT_LEN + PRED_LEN)
colours = {"aleatoric": "#e67e22", "epistemic": "#8e44ad", "total": "#2c3e50"}

fig, axes = plt.subplots(2, N_PANELS, figsize=(14, 6), sharey="row")
fig.suptitle("Uncertainty decomposition: Original vs High Dropout\n"
             "Orange=aleatoric  Purple=epistemic  Dark=total",
             fontsize=11, fontweight="bold")

for row, (label, res) in enumerate(results.items()):
    for col in range(N_PANELS):
        ax  = axes[row, col]
        mn  = res["mu"][:, EXAMPLE, col]
        al  = res["aleatoric"][:, EXAMPLE, col]
        ep  = res["epistemic"][:, EXAMPLE, col]
        tot = res["total"][:, EXAMPLE, col]
        gt  = Y_np[:, EXAMPLE, col]
        ctx = X_np[:, EXAMPLE, col]

        ax.plot(t_ctx,  ctx, color="#4a90d9", lw=1.0)
        ax.plot(t_pred, gt,  color="#2ecc71", lw=1.5, zorder=5)
        ax.plot(t_pred, mn,  color="black",   lw=1.2, ls="--", zorder=6)
        ax.fill_between(t_pred, mn-tot, mn+tot, alpha=0.12, color=colours["total"])
        ax.fill_between(t_pred, mn-al,  mn+al,  alpha=0.30, color=colours["aleatoric"])
        ax.fill_between(t_pred, mn-ep,  mn+ep,  alpha=0.55, color=colours["epistemic"])
        ax.axvline(x=CONTEXT_LEN-0.5, color="grey", ls=":", lw=0.8)
        ax.set_title(f"PC {col}", fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(label.replace("\n", " "), fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT_DIR, "uncertainty_comparison_bands.png"), dpi=150, bbox_inches="tight")
print("\nSaved: uncertainty_comparison_bands.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: uncertainty magnitude vs horizon for both models (PC 0)
# ---------------------------------------------------------------------------
steps = np.arange(1, PRED_LEN + 1)
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle("Aleatoric / Epistemic / Total vs Forecast Horizon (PC 0)",
              fontsize=11, fontweight="bold")

for ax, key, title in zip(axes2,
                           ["aleatoric", "epistemic", "total"],
                           ["Aleatoric  √E[σ²]", "Epistemic  √Var[μ]", "Total"]):
    for label, res in results.items():
        vals = res[key][:, EXAMPLE, 0]
        ax.plot(steps, vals, lw=2, marker="o", ms=4, label=label.replace("\n", " "))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Step ahead", fontsize=9)
    ax.set_ylabel("σ (z-score)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "uncertainty_vs_horizon_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved: uncertainty_vs_horizon_comparison.png")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: bar chart summary — mean aleatoric vs epistemic, both models
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(7, 4))
labels = list(results.keys())
x = np.arange(len(labels))
w = 0.3

al_means = [results[l]["aleatoric"][:, EXAMPLE, :N_PANELS].mean() for l in labels]
ep_means = [results[l]["epistemic"][:, EXAMPLE, :N_PANELS].mean() for l in labels]

bars1 = ax3.bar(x - w/2, al_means, w, label="Aleatoric", color=colours["aleatoric"], alpha=0.85)
bars2 = ax3.bar(x + w/2, ep_means, w, label="Epistemic", color=colours["epistemic"], alpha=0.85)

ax3.bar_label(bars1, fmt="%.4f", fontsize=8, padding=2)
ax3.bar_label(bars2, fmt="%.4f", fontsize=8, padding=2)
ax3.set_xticks(x)
ax3.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=9)
ax3.set_ylabel("Mean σ (z-score, averaged over\npred steps & first 4 PCs)", fontsize=9)
ax3.set_title("Aleatoric vs Epistemic Uncertainty — Original vs High Dropout", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "uncertainty_bar_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved: uncertainty_bar_comparison.png")
plt.close()
