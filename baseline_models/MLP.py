"""
MLP-only ablation of probabilistic POCO for predicting neural activity from
calcium imaging data.

Architecture:
  - Per-neuron MLP: in_proj (Linear + ReLU) → mu_proj + log_sig_proj (Linear)
  - Identical to POCO_prob's conditioning head, but without the POYO encoder.
  - Without the Perceiver-IO embedding, the FiLM alpha/beta terms collapse to
    zero (their initialisation in POCO), leaving:
        h       = in_proj(x)
        mu      = mu_proj(h)
        log_sig = log_sig_proj(h)
  - Trained with Gaussian NLL, same as POCO_prob.

Dimensions match POCO_prob exactly (COND_DIM=1024, PRED_LEN=16, context_len=48)
so any performance gap is attributable solely to the POYO encoder.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

from metrics import Prediction, NllLoss, MAELoss, MetricSuite, Score

# ---------------------------------------------------------------------------
# Dataset  (identical to POCO_prob.py)
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    def __init__(self, traces: np.ndarray, context_len: int, pred_len: int):

        seq_len = context_len + pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])

        # Convert numpy array to torch tensor
        self.X = torch.tensor(np.array(X))   # (S, context_len, N)
        self.Y = torch.tensor(np.array(Y))   # (S, pred_len,    N)

    def __len__(self):  
        return len(self.X)
    
    def __getitem__(self, i): 
        return self.X[i], self.Y[i]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Per-neuron probabilistic MLP — POCO_prob's conditioning head without the
    POYO encoder.

    Each neuron's context window is projected independently:
        x (B, N, context_len) → in_proj → h (B, N, cond_dim)
            → mu_proj      → mu      (B, N, pred_len)
            → log_sig_proj → log_sig (B, N, pred_len)

    Accepts x_list in POCO's (L, B, D) format and returns a list of
    Normal distributions with event shape (pred_len, B, N).
    """

    LOG_SIG_MIN = -6.0
    LOG_SIG_MAX =  2.0

    def __init__(self, n_neurons: int, context_len: int, cond_dim: int, pred_len: int):
        super().__init__()
        self.in_proj      = nn.Sequential(nn.Linear(context_len, cond_dim), nn.ReLU())
        self.mu_proj      = nn.Linear(cond_dim, pred_len)
        self.log_sig_proj = nn.Linear(cond_dim, pred_len)

        # Initialise log_sig bias to predict ~0.5 std initially — matches POCO_prob
        nn.init.constant_(self.log_sig_proj.bias, -0.69)
        nn.init.zeros_(self.log_sig_proj.weight)

    def forward(self, x_list: torch.Tensor) -> list[Normal]:
        """
        Args:
            x_list: list with one tensor of shape (context_len, B, N)
        Returns:
            list with one Normal distribution; mu/sigma shape (pred_len, B, N)
        """
        x = x_list.permute(1, 2, 0) # Reorders dimensions to (B, N, context_len)
        h = self.in_proj(x) # (B, N, cond_dim)

        mu      = self.mu_proj(h)                                  # (B, N, pred_len)
        log_sig = self.log_sig_proj(h).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        sigma   = F.softplus(log_sig) + 1e-4

        mu    = mu.permute(2, 0, 1)                                # (pred_len, B, N)
        sigma = sigma.permute(2, 0, 1)
        return Prediction(mean=mu, sigma=sigma)



# ---------------------------------------------------------------------------
# Training / evaluation  (identical to POCO_prob.py)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    score = Score.create(criterion)
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        x_list = X.permute(1, 0, 2)
        y_list = Y.permute(1, 0, 2)

        optimizer.zero_grad()
        dists = model(x_list)
        loss, scores  = criterion(dists, y_list)
        score.update(scores)
        print(score)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_nll, total_mae, n = 0.0, 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        x_list = X.permute(1, 0, 2)
        y_list = Y.permute(1, 0, 2)

        dists = model(x_list)
        loss, _ = criterion(dists, y_list)
        total_nll += loss.item() * len(X)
        n += len(X)
    return total_nll / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_mlp.pt"
    RESULTS_PATH = "results/mlp_losses.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Hyperparameters — match POCO_prob exactly for a fair ablation
    N_PCS      = 128    # cap at 128 PCs, same as POCO_prob
    CONTEXT    = 48     # input steps — matches paper (C=48)
    PRED_LEN   = 16     # forecast horizon — matches paper (P=16)
    COND_DIM   = 1024   # MLP conditioning dimension — matches paper
    BATCH_SIZE = 64     # matches POCO_prob
    EPOCHS     = 50
    LR         = 3e-4
    TRAIN_FRAC = 0.6    # 3:1:1 split — matches POCO_prob / paper
    VAL_FRAC   = 0.2
    PATIENCE   = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {DATA_PATH} ...")
    data = np.load(DATA_PATH)
    if "PC" in data:
        raw = data["PC"].astype(np.float32)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
    elif "M" in data:
        raw = data["M"].astype(np.float32)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if "valid_indices" in data:
            raw = raw[:, data["valid_indices"]]
    else:
        raise ValueError(f"No recognised key in {DATA_PATH}. Found: {list(data.keys())}")

    traces = raw[:, :N_PCS] if N_PCS is not None else raw
    T, N   = traces.shape
    print(f"Traces shape: {traces.shape}  (T={T}, features={N})")

    train_end = int(T * TRAIN_FRAC)
    val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

    train_ds = CalciumDataset(traces[:train_end],        CONTEXT, PRED_LEN)
    val_ds   = CalciumDataset(traces[train_end:val_end], CONTEXT, PRED_LEN)
    test_ds  = CalciumDataset(traces[val_end:],          CONTEXT, PRED_LEN)
    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}  |  Test windows: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MLPHead(
        n_neurons=N, context_len=CONTEXT, cond_dim=COND_DIM, pred_len=PRED_LEN,
    ).to(device)
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = MetricSuite([MAELoss()], primary=NllLoss())

    train_nlls, val_nlls = [], []
    best_nll   = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_nll = train_epoch(model, train_loader, criterion, optimizer, device)
        val_nll = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_nll)
        train_nlls.append(train_nll)
        val_nlls.append(val_nll)

        tag = " *" if val_nll < best_nll else ""
        if val_nll < best_nll:
            best_nll   = val_nll
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  train_nll={train_nll:.4f}  val_nll={val_nll:.4f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest val NLL: {best_nll:.4f}  — weights saved to {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    test_nll, test_mae = eval_epoch(model, test_loader, criterion, device)
    print(f"Test  NLL: {test_nll:.4f}  |  Test MAE: {test_mae:.4f}")

    np.savez(RESULTS_PATH, train_nlls=train_nlls, val_nlls=val_nlls,
             test_nll=test_nll, test_mae=test_mae)
    print(f"Loss history saved to {RESULTS_PATH}")
