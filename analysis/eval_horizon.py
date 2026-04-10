"""
Evaluate POCO (det.) and POCO (prob.) as a function of prediction horizon step.

For each step h in [1, PRED_LEN] we compute:
  - MAE averaged over all validation windows and all 128 PCs

This replicates the horizon-vs-accuracy plot style used in the original
POCO study (Figure 3 / supplementary performance curves).

Usage:
    python3 eval_horizon.py

Outputs:
    results/plots/horizon_mae.png
    results/horizon_mae.npz
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poco_src.standalone_poco import POCO, NeuralPredictionConfig
from src.poco_src.prob import ProbabilisticPOCO, nll_loss, CalciumDataset

os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Config — must match the training scripts exactly
# ---------------------------------------------------------------------------
DATA_PATH      = "data/processed/0.npz"
DET_MODEL_PATH = "models/best_calcium_poco.pt"
PROB_MODEL_PATH= "models/saved/model.pt"

N_PCS        = 128
SEQ_LEN      = 64       # context (48) + horizon (16)
PRED_LEN     = 16
BATCH_SIZE   = 64
TRAIN_FRAC   = 0.6
VAL_FRAC     = 0.2

COMPRESSION  = 16
NUM_LATENTS  = 8
HIDDEN_DIM   = 128
COND_DIM     = 1024
NUM_LAYERS   = 1
NUM_HEADS    = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH)
raw  = data["PC"].astype(np.float32)
if raw.shape[0] < raw.shape[1]:
    raw = raw.T
traces = raw[:, :N_PCS]
T, N    = traces.shape

# z-score each neuron over the full recording before splitting
mu      = traces.mean(0, keepdims=True)
sd      = traces.std(0,  keepdims=True) + 1e-8
traces  = (traces - mu) / sd

val_end    = int(T * (TRAIN_FRAC + VAL_FRAC))
val_ds     = CalciumDataset(traces[val_end:], context_len=SEQ_LEN - PRED_LEN, pred_len=PRED_LEN)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test windows: {len(val_ds)}")

# ---------------------------------------------------------------------------
# Build configs — must match each training script exactly
# ---------------------------------------------------------------------------
def det_config():
    """Matches POCO.py: seq_length=SEQ_LEN, COND_DIM=256."""
    cfg = NeuralPredictionConfig()
    cfg.seq_length             = SEQ_LEN
    cfg.pred_length            = PRED_LEN
    cfg.compression_factor     = COMPRESSION
    cfg.poyo_num_latents       = NUM_LATENTS
    cfg.decoder_hidden_size    = HIDDEN_DIM
    cfg.conditioning_dim       = COND_DIM
    cfg.decoder_num_layers     = NUM_LAYERS
    cfg.decoder_num_heads      = NUM_HEADS
    cfg.decoder_context_length = None
    cfg.freeze_backbone        = False
    cfg.freeze_conditioned_net = False
    return cfg

def prob_config():
    """Matches POCO_prob.py: seq_length=CONTEXT+PRED_LEN, COND_DIM=128."""
    cfg = NeuralPredictionConfig()
    cfg.seq_length             = SEQ_LEN        # CONTEXT + PRED_LEN = 64
    cfg.pred_length            = PRED_LEN
    cfg.compression_factor     = COMPRESSION
    cfg.poyo_num_latents       = NUM_LATENTS
    cfg.decoder_hidden_size    = HIDDEN_DIM
    cfg.conditioning_dim       = COND_DIM
    cfg.decoder_num_layers     = NUM_LAYERS
    cfg.decoder_num_heads      = NUM_HEADS
    cfg.decoder_context_length = None
    cfg.freeze_backbone        = False
    cfg.freeze_conditioned_net = False
    return cfg

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
det_model = POCO(det_config(), [[N]]).to(device)
_ckpt_det = torch.load(DET_MODEL_PATH, map_location=device, weights_only=True)
if all(k.startswith("poco.") for k in _ckpt_det):
    _ckpt_det = {k[len("poco."):]: v for k, v in _ckpt_det.items()}
det_model.load_state_dict(_ckpt_det)
det_model.eval()
print("Deterministic POCO loaded.")

prob_model = ProbabilisticPOCO(prob_config(), [[N]]).to(device)
_ckpt_prob = torch.load(PROB_MODEL_PATH, map_location=device, weights_only=True)
if all(k.startswith("poco.") for k in _ckpt_prob):
    _ckpt_prob = {k[len("poco."):]: v for k, v in _ckpt_prob.items()}
prob_model.load_state_dict(_ckpt_prob)
prob_model.eval()
print("Probabilistic POCO loaded.")

# ---------------------------------------------------------------------------
# Evaluate: collect per-step MAE over the full validation set
# ---------------------------------------------------------------------------
# det_errors[h] = sum of |pred - true| at step h across all windows & neurons
det_errors  = np.zeros(PRED_LEN)
prob_errors = np.zeros(PRED_LEN)
n_total     = 0   # total windows * neurons contribution denominator

with torch.no_grad():
    for X, Y in val_loader:
        X      = X.to(device)                    # (B, context_len, N)
        Y      = Y.to(device)                    # (B, pred_len, N)
        x_list = [X.permute(1, 0, 2)]           # (context_len, B, N)
        y_poco = Y.permute(1, 0, 2)             # (pred_len, B, N)

        # Deterministic
        preds = det_model(x_list)[0]             # (pred_len, B, N)
        ae    = (preds - y_poco).abs()
        det_errors  += ae.mean(dim=(1, 2)).cpu().numpy()

        # Probabilistic (use mean prediction for MAE)
        dist  = prob_model(x_list)[0]
        mu_p  = dist.mean                        # (pred_len, B, N)
        ae_p  = (mu_p - y_poco).abs()
        prob_errors += ae_p.mean(dim=(1, 2)).cpu().numpy()

        n_total += 1   # number of batches

# Average over batches
det_mae_per_step  = det_errors  / n_total
prob_mae_per_step = prob_errors / n_total

steps = np.arange(1, PRED_LEN + 1)

# Save
np.savez("results/horizon_mae.npz",
         steps=steps,
         det_mae=det_mae_per_step,
         prob_mae=prob_mae_per_step)
print("Saved results/horizon_mae.npz")

# ---------------------------------------------------------------------------
# Also add a copy baseline: repeat last context step
# ---------------------------------------------------------------------------
copy_errors = np.zeros(PRED_LEN)
n_copy = 0
with torch.no_grad():
    for X, Y in val_loader:
        X       = X.to(device)                  # (B, context_len, N)
        Y       = Y.to(device)                  # (B, pred_len, N)
        y_poco  = Y.permute(1, 0, 2)           # (pred_len, B, N)
        last    = X[:, -1:, :].permute(1, 0, 2).expand(PRED_LEN, -1, -1)
        ae      = (last - y_poco).abs()
        copy_errors += ae.mean(dim=(1, 2)).cpu().numpy()
        n_copy  += 1

copy_mae_per_step = copy_errors / n_copy

# Also add mean baseline: mean of context window
mean_errors = np.zeros(PRED_LEN)
n_mean = 0
with torch.no_grad():
    for X, Y in val_loader:
        X        = X.to(device)
        Y        = Y.to(device)
        y_poco   = Y.permute(1, 0, 2)
        ctx_mean = X.mean(dim=1, keepdim=True).permute(1, 0, 2).expand(PRED_LEN, -1, -1)
        ae       = (ctx_mean - y_poco).abs()
        mean_errors += ae.mean(dim=(1, 2)).cpu().numpy()
        n_mean  += 1

mean_mae_per_step = mean_errors / n_mean

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps, copy_mae_per_step, color="#aaaaaa", linewidth=1.8,
        linestyle=":", marker=".", markersize=5, label="Copy baseline")
ax.plot(steps, mean_mae_per_step, color="#cccccc", linewidth=1.8,
        linestyle="--", marker=".", markersize=5, label="Mean baseline")
ax.plot(steps, det_mae_per_step,  color="#7d3c98", linewidth=2.2,
        marker="o", markersize=5, label="POCO (det.)")
ax.plot(steps, prob_mae_per_step, color="#e74c3c", linewidth=2.2,
        marker="s", markersize=5, label="POCO (prob.)")

ax.set_xlabel("Prediction step  (frames)", fontsize=12)
ax.set_ylabel("MAE  (z-score units)", fontsize=12)
ax.set_title("Forecasting error vs prediction horizon\n"
             "C=48  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=12, fontweight="bold")
ax.set_xticks(steps)
ax.legend(fontsize=10, framealpha=0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0.5, PRED_LEN + 0.5)

plt.tight_layout()
out = "results/plots/horizon_mae.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print(f"\n{'Step':>5}  {'Copy':>8}  {'Mean':>8}  {'POCO det':>10}  {'POCO prob':>10}")
print("-" * 50)
for h, (cp, mn, dt, pb) in enumerate(zip(
        copy_mae_per_step, mean_mae_per_step,
        det_mae_per_step, prob_mae_per_step), start=1):
    print(f"{h:>5}  {cp:>8.4f}  {mn:>8.4f}  {dt:>10.4f}  {pb:>10.4f}")
