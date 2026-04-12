"""
eval_horizon.py
---------------
Evaluate POCO (det.), POCO (prob.), and MLP MAE as a function of prediction horizon step.

For each step h in [1, PRED_LEN] we compute:
  - MAE averaged over all test windows and all 128 PCs

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util                  import trainingConfig, CalciumDataset
from src.dataset               import _load_traces
from src.model                 import ProbabilisticForecaster, DeterministicPOCO
from src.baseline_models.MLP   import MLPHead

os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Config — mirrors trainingConfig defaults
# ---------------------------------------------------------------------------
SEQ_LEN    = 64
PRED_LEN   = 16
CONTEXT    = SEQ_LEN - PRED_LEN   # 48
COND_DIM   = 1024
BATCH_SIZE = 64
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Build test set from held-out 20% — same split as trainer.py
# ---------------------------------------------------------------------------
traces  = _load_traces()
T       = traces.shape[0]
N       = traces.shape[1]
val_end = int(T * (TRAIN_FRAC + VAL_FRAC))

test_ds     = CalciumDataset(traces[val_end:], context_len=CONTEXT, pred_len=PRED_LEN)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test windows: {len(test_ds)}  |  N channels: {N}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def _load_det_poco():
    model = DeterministicPOCO(seq_length=SEQ_LEN, pred_length=PRED_LEN, n_channels=N).to(device)
    path  = trainingConfig(model_name="DeterministicPoco").save_path
    print(f"Loading det POCO from: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # checkpoints saved from bare POCO lack the 'poco.' prefix — add it
    model_keys = set(model.state_dict().keys())
    if not set(ckpt.keys()).issubset(model_keys):
        ckpt = {f"poco.{k}": v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def _load_prob_poco():
    model = ProbabilisticForecaster(seq_length=SEQ_LEN, pred_length=PRED_LEN, n_channels=N).to(device)
    path  = trainingConfig(model_name="ProbabilisticPOCO").save_path
    print(f"Loading prob POCO from: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_mlp():
    model = MLPHead(n_neurons=N, context_len=CONTEXT, cond_dim=COND_DIM, pred_len=PRED_LEN).to(device)
    path  = trainingConfig(model_name="MLP").save_path
    print(f"Loading MLP from: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


det_model  = _load_det_poco()
prob_model = _load_prob_poco()
mlp_model  = _load_mlp()

# ---------------------------------------------------------------------------
# Per-step MAE evaluation
# ---------------------------------------------------------------------------

def eval_per_step(model, loader):
    """Return (PRED_LEN,) array of mean MAE per forecast step."""
    errors  = np.zeros(PRED_LEN)
    n_batch = 0
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)   # (B, context_len, N)
            Y = Y.to(device)   # (B, pred_len,    N)
            pred = model(X)
            mu   = pred.mean   # (B, pred_len, N)
            errors  += (mu - Y).abs().mean(dim=(0, 2)).cpu().numpy()
            n_batch += 1
    return errors / n_batch


print("\nEvaluating deterministic POCO ...")
det_mae  = eval_per_step(det_model,  test_loader)

print("Evaluating probabilistic POCO ...")
prob_mae = eval_per_step(prob_model, test_loader)

print("Evaluating MLP ...")
mlp_mae  = eval_per_step(mlp_model,  test_loader)

# ---------------------------------------------------------------------------
# Copy baseline  (repeat last context frame)
# ---------------------------------------------------------------------------
copy_errors = np.zeros(PRED_LEN)
mean_errors = np.zeros(PRED_LEN)
n_batch = 0
with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        Y = Y.to(device)
        last     = X[:, -1:, :].expand(-1, PRED_LEN, -1)
        ctx_mean = X.mean(dim=1, keepdim=True).expand(-1, PRED_LEN, -1)
        copy_errors += (last     - Y).abs().mean(dim=(0, 2)).cpu().numpy()
        mean_errors += (ctx_mean - Y).abs().mean(dim=(0, 2)).cpu().numpy()
        n_batch += 1

copy_mae = copy_errors / n_batch
mean_mae = mean_errors / n_batch

steps = np.arange(1, PRED_LEN + 1)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.savez("results/horizon_mae.npz",
         steps=steps,
         det_mae=det_mae,
         prob_mae=prob_mae,
         mlp_mae=mlp_mae,
         copy_mae=copy_mae,
         mean_mae=mean_mae)
print("\nSaved results/horizon_mae.npz")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps, copy_mae, color="#aaaaaa", linewidth=1.5,
        linestyle=":", marker=".", markersize=4, label="Copy baseline")
ax.plot(steps, mean_mae, color="#bbbbbb", linewidth=1.5,
        linestyle="--", marker=".", markersize=4, label="Mean baseline")
ax.plot(steps, mlp_mae,  color="#2980b9", linewidth=2.2,
        marker="^", markersize=5, label="MLP")
ax.plot(steps, det_mae,  color="#7d3c98", linewidth=2.2,
        marker="o", markersize=5, label="POCO (det.)")
ax.plot(steps, prob_mae, color="#e74c3c", linewidth=2.2,
        marker="s", markersize=5, label="POCO (prob.)")

# Mean MAE horizontal lines
ax.axhline(mlp_mae.mean(),  color="#2980b9", linewidth=1.0, linestyle="--",
           label=f"MLP mean MAE = {mlp_mae.mean():.4f}")
ax.axhline(det_mae.mean(),  color="#7d3c98", linewidth=1.0, linestyle="--",
           label=f"POCO (det.) mean MAE = {det_mae.mean():.4f}")
ax.axhline(prob_mae.mean(), color="#e74c3c", linewidth=1.0, linestyle="--",
           label=f"POCO (prob.) mean MAE = {prob_mae.mean():.4f}")

ax.set_xlabel("Prediction step  (frames)", fontsize=12)
ax.set_ylabel("MAE  (z-score units)", fontsize=12)
ax.set_title("Forecasting error vs prediction horizon\n"
             "C=48  |  Top-128 PCs  |  Zebrafish Ahrens (Subject 0)",
             fontsize=12, fontweight="bold")
ax.set_xticks(steps)
ax.legend(fontsize=9, framealpha=0.7)
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
print(f"\n{'Step':>5}  {'Copy':>8}  {'Mean':>8}  {'MLP':>8}  {'POCO det':>10}  {'POCO prob':>10}")
print("-" * 60)
for h, (cp, mn, ml, dt, pb) in enumerate(
        zip(copy_mae, mean_mae, mlp_mae, det_mae, prob_mae), start=1):
    print(f"{h:>5}  {cp:>8.4f}  {mn:>8.4f}  {ml:>8.4f}  {dt:>10.4f}  {pb:>10.4f}")
