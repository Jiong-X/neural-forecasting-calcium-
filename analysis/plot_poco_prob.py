"""
Probabilistic POCO visualisation.

Figure 1: Prob. POCO mean prediction + ±1σ / ±2σ interval
          (4 PC panels, single forward pass)
Figure 2: Uncertainty decomposition — aleatoric / epistemic / total
          (MC Dropout, T=50 passes, same 4 panels)
Figure 3: Uncertainty vs forecast horizon (all three bands per PC)

Decomposition (Kendall & Gal, 2017):
    aleatoric  = sqrt( E_t[sigma^2] )
    epistemic  = sqrt( Var_t[mu]    )
    total      = sqrt( E[sigma^2] + Var[mu] )
"""
import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from poco_src.POCO_prob import ProbabilisticPOCO, CalciumDataset
from poco_src.standalone_poco import NeuralPredictionConfig

# ---------------------------------------------------------------------------
# Config — must match training settings in POCO.py and POCO_prob.py
# ---------------------------------------------------------------------------
DATA_PATH       = "data/processed/0.npz"
PROB_MODEL_PATH = "models/saved/model.pt"
OUT_DIR         = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

N_PCS       = 128
SEQ_LEN     = 64
PRED_LEN    = 16
CONTEXT_LEN = SEQ_LEN - PRED_LEN   # 48
COMPRESSION = 16
TRAIN_FRAC  = 0.6
VAL_FRAC    = 0.2
MC_SAMPLES  = 50

NUM_LATENTS = 8
HIDDEN_DIM  = 128   # matches paper + updated training config
COND_DIM    = 1024  # matches paper + updated training config
NUM_LAYERS  = 1
NUM_HEADS   = 16    # matches paper + updated training config

N_PANELS = 4
EXAMPLE  = 0    # which example in the batch to visualise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data   = np.load(DATA_PATH)
raw    = data["PC"].astype(np.float32)
if raw.shape[0] < raw.shape[1]:
    raw = raw.T
traces = raw[:, :N_PCS]
T, N   = traces.shape

split    = int(T * TRAIN_FRAC)                      # end of train
val_end  = int(T * (TRAIN_FRAC + VAL_FRAC))         # end of val

# z-score each neuron over the full recording before splitting
mu     = traces.mean(0, keepdims=True)
sd     = traces.std(0,  keepdims=True) + 1e-8
traces = (traces - mu) / sd

val_ds     = CalciumDataset(traces[split:val_end], context_len=CONTEXT_LEN, pred_len=PRED_LEN)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

X, Y   = next(iter(val_loader))
X, Y   = X.to(device), Y.to(device)
x_list = [X.permute(1, 0, 2)]               # (context_len, B, N)
Y_np   = Y.permute(1, 0, 2).cpu().numpy()   # (pred_len,    B, N)
X_np   = X.permute(1, 0, 2).cpu().numpy()   # (context_len, B, N)

t_context = np.arange(CONTEXT_LEN)
t_pred    = np.arange(CONTEXT_LEN, CONTEXT_LEN + PRED_LEN)

# ---------------------------------------------------------------------------
# Build shared config
# ---------------------------------------------------------------------------
def build_config(hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM):
    cfg = NeuralPredictionConfig()
    cfg.seq_length             = SEQ_LEN
    cfg.pred_length            = PRED_LEN
    cfg.compression_factor     = COMPRESSION
    cfg.poyo_num_latents       = NUM_LATENTS
    cfg.decoder_hidden_size    = hidden_dim
    cfg.conditioning_dim       = cond_dim
    cfg.decoder_num_layers     = NUM_LAYERS
    cfg.decoder_num_heads      = NUM_HEADS
    cfg.decoder_context_length = None
    cfg.freeze_backbone        = False
    cfg.freeze_conditioned_net = False
    return cfg

# ---------------------------------------------------------------------------
# Load probabilistic POCO
# ---------------------------------------------------------------------------
prob_model = ProbabilisticPOCO(build_config(HIDDEN_DIM, COND_DIM), [[N]]).to(device)

# Checkpoint may come from ProbabilisticForecaster (train.py → self.poco = ...)
# which prefixes all keys with "poco.".  Strip that prefix if present.
_ckpt = torch.load(PROB_MODEL_PATH, map_location=device, weights_only=True)
if all(k.startswith("poco.") for k in _ckpt):
    _ckpt = {k[len("poco."):]: v for k, v in _ckpt.items()}
prob_model.load_state_dict(_ckpt)
prob_model.eval()
print("Probabilistic POCO loaded.")

with torch.no_grad():
    dists     = prob_model(x_list)
    dist      = dists[0]
    mu_prob   = dist.mean.cpu().numpy()     # (pred_len, B, N)
    sigma_prob = dist.scale.cpu().numpy()

# ---------------------------------------------------------------------------
# MC Dropout passes (prob model only)
# ---------------------------------------------------------------------------
def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()

prob_model.eval()
prob_model.apply(enable_dropout)

mus_mc, vars_mc = [], []
with torch.no_grad():
    for _ in range(MC_SAMPLES):
        d = prob_model(x_list)[0]
        mus_mc.append(d.mean.cpu().numpy())
        vars_mc.append(d.scale.cpu().numpy() ** 2)

mus_mc  = np.stack(mus_mc,  axis=0)   # (MC, pred_len, B, N)
vars_mc = np.stack(vars_mc, axis=0)

aleatoric = np.sqrt(vars_mc.mean(0))
epistemic = np.sqrt(mus_mc.var(0))
total     = np.sqrt(vars_mc.mean(0) + mus_mc.var(0))
mu_mc     = mus_mc.mean(0)

print(f"Aleatoric (PC0 mean): {aleatoric[:, EXAMPLE, 0].mean():.4f}")
print(f"Epistemic (PC0 mean): {epistemic[:, EXAMPLE, 0].mean():.4f}")
print(f"Total     (PC0 mean): {total[:, EXAMPLE, 0].mean():.4f}")

# ---------------------------------------------------------------------------
# Figure 1 — Prob POCO mean prediction + uncertainty intervals
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(N_PANELS, 1, figsize=(12, 3 * N_PANELS), sharex=True)
fig.suptitle("Probabilistic POCO — Prediction with Uncertainty Intervals\n"
             "Shaded region = ±1σ / ±2σ from probabilistic head",
             fontsize=12, fontweight="bold")

for i, ax in enumerate(axes):
    ctx  = X_np[:, EXAMPLE, i]
    gt   = Y_np[:, EXAMPLE, i]
    m_p  = mu_prob[:, EXAMPLE, i]
    sg   = sigma_prob[:, EXAMPLE, i]

    # context
    ax.plot(t_context, ctx, color="#4a90d9", lw=1.2, label="Context (ground truth)")
    # ground truth future
    ax.plot(t_pred, gt, color="#2ecc71", lw=2.0, label="Ground truth (future)", zorder=6)
    # probabilistic POCO mean + intervals
    ax.plot(t_pred, m_p, color="#8e44ad", lw=1.8, ls="--",
            label="POCO prob. μ", zorder=7)
    ax.fill_between(t_pred, m_p - sg,   m_p + sg,
                    alpha=0.30, color="#8e44ad", label="±1σ")
    ax.fill_between(t_pred, m_p - 2*sg, m_p + 2*sg,
                    alpha=0.12, color="#8e44ad", label="±2σ")

    ax.axvline(x=CONTEXT_LEN - 0.5, color="grey", ls=":", lw=1.0, alpha=0.7)
    ax.set_ylabel(f"PC {i}  (z-score)", fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.7)

axes[-1].set_xlabel("Time  (frames,  1 frame ≈ 0.5 s)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = os.path.join(OUT_DIR, "poco_prob_prediction.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Uncertainty decomposition (MC Dropout)
# ---------------------------------------------------------------------------
colours = {"aleatoric": "#e67e22", "epistemic": "#8e44ad", "total": "#2c3e50"}

fig2, axes2 = plt.subplots(N_PANELS, 1, figsize=(12, 3 * N_PANELS), sharex=True)
fig2.suptitle("Probabilistic POCO — Uncertainty Decomposition (MC Dropout, T=50)\n"
              "Aleatoric = E[σ²]^½   |   Epistemic = Var[μ]^½   |   Total = law of total variance",
              fontsize=11, fontweight="bold")

for i, ax in enumerate(axes2):
    ctx = X_np[:, EXAMPLE, i]
    gt  = Y_np[:, EXAMPLE, i]
    mn  = mu_mc[:, EXAMPLE, i]
    al  = aleatoric[:, EXAMPLE, i]
    ep  = epistemic[:, EXAMPLE, i]
    tot = total[:, EXAMPLE, i]

    ax.plot(t_context, ctx, color="#4a90d9", lw=1.2, label="Context")
    ax.plot(t_pred, gt,     color="#2ecc71", lw=1.5, label="Ground truth", zorder=5)
    ax.plot(t_pred, mn,     color="black",   lw=1.5, ls="--", label="MC mean μ", zorder=6)

    ax.fill_between(t_pred, mn - tot, mn + tot,
                    alpha=0.12, color=colours["total"],
                    label=f"±total  ({tot.mean():.3f})")
    ax.fill_between(t_pred, mn - al, mn + al,
                    alpha=0.30, color=colours["aleatoric"],
                    label=f"±aleatoric  ({al.mean():.3f})")
    ax.fill_between(t_pred, mn - ep, mn + ep,
                    alpha=0.45, color=colours["epistemic"],
                    label=f"±epistemic  ({ep.mean():.3f})")

    ax.axvline(x=CONTEXT_LEN - 0.5, color="grey", ls=":", lw=1.0, alpha=0.7)
    ax.set_ylabel(f"PC {i}  (z-score)", fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.7)

axes2[-1].set_xlabel("Time  (frames,  1 frame ≈ 0.5 s)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = os.path.join(OUT_DIR, "poco_uncertainty_decomposition.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Uncertainty magnitude vs horizon
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, N_PANELS, figsize=(14, 4), sharey=False)
fig3.suptitle("Uncertainty vs Forecast Horizon (MC Dropout, T=50)",
              fontsize=12, fontweight="bold")

steps = np.arange(1, PRED_LEN + 1)
for i, ax in enumerate(axes3):
    al  = aleatoric[:, EXAMPLE, i]
    ep  = epistemic[:, EXAMPLE, i]
    tot = total[:, EXAMPLE, i]

    ax.plot(steps, al,  color=colours["aleatoric"], lw=2.0, marker="o", ms=4, label="Aleatoric")
    ax.plot(steps, ep,  color=colours["epistemic"], lw=2.0, marker="s", ms=4, label="Epistemic")
    ax.plot(steps, tot, color=colours["total"],     lw=2.0, marker="^", ms=4, ls="--", label="Total")

    ax.set_title(f"PC {i}", fontsize=10)
    ax.set_xlabel("Step ahead", fontsize=9)
    ax.set_ylabel("σ  (z-score)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "poco_uncertainty_vs_horizon.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()
