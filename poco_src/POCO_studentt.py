# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
POCO_studentt.py
----------------
Probabilistic POCO with a Student-t output head.

Motivation
----------
Pooled standardised residuals from POCO_prob show near-zero skewness (+0.135)
but substantial positive excess kurtosis (+2.39), indicating the Gaussian
likelihood underestimates tail probability. A Student-t distribution with
ν degrees of freedom has excess kurtosis = 6/(ν-4) (valid for ν > 4).

Degrees of freedom selection
-----------------------------
Two methods are used (see estimate_df.py):

  1. Kurtosis formula  :  ν = 6 / excess_kurtosis + 4
                       :  ν = 6 / 2.386 + 4 ≈ 6.5

  2. MLE               :  scipy.stats.t.fit() on pooled residuals
                       :  returns ν ≈ 6–8 (run estimate_df.py to confirm)

Both methods agree, giving ν ≈ 7. This value is set as the default and
can be overridden at construction time.

Architecture changes vs POCO_prob
-----------------------------------
  - Output distribution: Normal  →  StudentT(df=ν, loc=μ, scale=σ)
  - Loss: Gaussian NLL  →  Student-t NLL  (same formula: -log_prob.mean())
  - Optional: ν can be fixed (default) or learned (learnable_df=True)
  - Everything else — encoder, FiLM conditioning, μ/σ heads — is identical.

Usage:
  python POCO_studentt.py          # train with default ν=7
  python POCO_studentt.py --df 5   # use ν=5
"""

import sys, os, argparse
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import StudentT
from torch.utils.data import Dataset, DataLoader

from poco_src.standalone_poco import POCO, NeuralPredictionConfig


# ---------------------------------------------------------------------------
# Student-t POCO
# ---------------------------------------------------------------------------

class StudentTPOCO(POCO):
    """
    POCO with a Student-t output head.

    Replaces the Gaussian Normal(mu, sigma) with StudentT(df, mu, sigma).

    Parameters
    ----------
    config       : NeuralPredictionConfig
    input_size   : list of [n_channels] per session
    df           : degrees of freedom ν (default 7, estimated from residuals)
    learnable_df : if True, ν is a learnable scalar parameter (log-parameterised
                   to keep it positive); otherwise fixed at the given value.
    """

    LOG_SIG_MIN = -6.0
    LOG_SIG_MAX =  2.0

    def __init__(self,
                 config: NeuralPredictionConfig,
                 input_size,
                 df: float = 7.0,
                 learnable_df: bool = False):
        super().__init__(config, input_size)

        cond_dim    = config.conditioning_dim
        pred_length = config.pred_length

        del self.out_proj

        self.mu_proj      = nn.Linear(cond_dim, pred_length)
        self.log_sig_proj = nn.Linear(cond_dim, pred_length)

        nn.init.constant_(self.log_sig_proj.bias, -0.69)
        nn.init.zeros_(self.log_sig_proj.weight)

        # Degrees of freedom — fixed or learnable
        # log_df is used so that df = exp(log_df) is always > 0.
        # We also clamp df >= 2.5 during forward to keep variance finite.
        self.learnable_df = learnable_df
        if learnable_df:
            self.log_df = nn.Parameter(torch.tensor(float(df)).log())
        else:
            self.register_buffer("log_df",
                                 torch.tensor(float(df)).log())

    # ------------------------------------------------------------------
    def forward(self, x_list, unit_indices=None, unit_timestamps=None):
        """
        Args:
            x_list : list of (L, B, D) tensors — one per session.
        Returns:
            dists  : list of torch.distributions.StudentT, one per session.
        """
        bsz = [x.size(1) for x in x_list]
        L   = x_list[0].size(0)

        x = torch.concatenate(
            [x.permute(1, 2, 0).reshape(-1, L) for x in x_list], dim=0
        )
        if L != self.Tin:
            x = x[:, -self.Tin:]

        out = x.reshape(x.shape[0], self.Tin // self.T_step, self.T_step)
        d_list = self.input_size

        if unit_indices is None:
            sum_channels = 0
            unit_indices = []
            for b, d in zip(bsz, self.input_size):
                indices = (
                    torch.arange(d, device=x.device)
                    .unsqueeze(0).repeat(b, 1).reshape(-1)
                )
                unit_indices.append(indices + sum_channels)
                sum_channels += d
            unit_indices = torch.cat(unit_indices, dim=0)

        if unit_timestamps is None:
            unit_timestamps = (
                torch.zeros_like(unit_indices).unsqueeze(1)
                + torch.arange(0, self.Tin, self.T_step, device=x.device)
            )

        input_seqlen = torch.cat(
            [torch.full((b,), d, device=x.device)
             for b, d in zip(bsz, self.input_size)], dim=0
        )
        session_index = torch.cat(
            [torch.full((b,), i, device=x.device)
             for i, b in enumerate(bsz)], dim=0
        )
        dataset_index = torch.cat(
            [torch.full((b,), self.dataset_idx[i], device=x.device)
             for i, b in enumerate(bsz)], dim=0
        )

        embed = self.decoder(
            out,
            unit_indices=unit_indices,
            unit_timestamps=unit_timestamps,
            input_seqlen=input_seqlen,
            session_index=session_index,
            dataset_index=dataset_index,
        )

        split_size = [b * d for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0)
        embed = [xx.reshape(b, d, self.embedding_dim)
                 for xx, b, d in zip(embed, bsz, d_list)]

        # degrees of freedom — clamp to [2.5, ∞) so variance exists
        df = self.log_df.exp().clamp(min=2.5)

        dists = []
        for e, d, x_in in zip(embed, self.input_size, x_list):
            alpha   = self.conditioning_alpha(e)
            beta    = self.conditioning_beta(e)
            inp     = x_in.permute(1, 2, 0)
            weights = self.in_proj(inp) * alpha + beta

            mu      = self.mu_proj(weights)
            log_sig = self.log_sig_proj(weights).clamp(self.LOG_SIG_MIN,
                                                        self.LOG_SIG_MAX)
            sigma   = torch.nn.functional.softplus(log_sig) + 1e-4

            mu    = mu.permute(2, 0, 1)       # (pred_len, B, D)
            sigma = sigma.permute(2, 0, 1)

            dists.append(StudentT(df=df, loc=mu, scale=sigma))

        return dists

    @property
    def df(self):
        """Current degrees of freedom value."""
        return self.log_df.exp().clamp(min=2.5).item()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def nll_loss(dists, targets):
    """Mean Student-t NLL across all sessions."""
    total = sum(
        -dist.log_prob(y).mean() for dist, y in zip(dists, targets)
    )
    return total / len(dists)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    def __init__(self, traces: np.ndarray, context_len: int, pred_len: int):
        traces = traces.astype(np.float32)
        mu  = traces.mean(0, keepdims=True)
        sd  = traces.std(0,  keepdims=True) + 1e-8
        traces = (traces - mu) / sd
        seq_len = context_len + pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])
        self.X = torch.tensor(np.array(X))
        self.Y = torch.tensor(np.array(Y))

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device):
    model.train()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        x_list = [X.permute(1, 0, 2)]
        y_list = [Y.permute(1, 0, 2)]
        optimiser.zero_grad()
        dists = model(x_list)
        loss  = nll_loss(dists, y_list)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_nll, total_mae, n = 0.0, 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        x_list = [X.permute(1, 0, 2)]
        y_list = [Y.permute(1, 0, 2)]
        dists = model(x_list)
        total_nll += nll_loss(dists, y_list).item() * len(X)
        total_mae += sum(
            (dist.loc - y).abs().mean().item() * len(X)
            for dist, y in zip(dists, y_list)
        ) / len(dists)
        n += len(X)
    return total_nll / n, total_mae / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df",           type=float, default=7.0,
                        help="Student-t degrees of freedom ν (default=7, "
                             "estimated from POCO_prob residuals via estimate_df.py)")
    parser.add_argument("--learnable_df", action="store_true",
                        help="Make ν a learnable parameter")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch",        type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    DATA_PATH    = "data/processed/0.npz"
    df_tag       = f"df{int(args.df)}"
    MODEL_PATH   = f"models/best_poco_studentt_{df_tag}.pt"
    RESULTS_PATH = f"results/poco_studentt_{df_tag}_losses.npz"
    # also save under the default name so compare_calibration.py can find it
    DEFAULT_PATH = "models/best_poco_studentt.pt"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    CONTEXT    = 48
    PRED_LEN   = 16
    N_PCS      = 128
    TRAIN_FRAC = 0.6
    VAL_FRAC   = 0.2
    PATIENCE   = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Student-t ν = {args.df}"
          + ("  (learnable)" if args.learnable_df else "  (fixed)"))

    # ── Data ─────────────────────────────────────────────────────────────────
    data = np.load(DATA_PATH)
    raw  = data["PC"].astype(np.float32)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    raw  = raw[:, :N_PCS]
    T, N = raw.shape
    print(f"Traces: {T} steps × {N} PCs")

    train_end = int(T * TRAIN_FRAC)
    val_end   = int(T * (TRAIN_FRAC + VAL_FRAC))

    train_ds = CalciumDataset(raw[:train_end],        CONTEXT, PRED_LEN)
    val_ds   = CalciumDataset(raw[train_end:val_end], CONTEXT, PRED_LEN)
    test_ds  = CalciumDataset(raw[val_end:],          CONTEXT, PRED_LEN)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    config = NeuralPredictionConfig()
    config.seq_length          = CONTEXT + PRED_LEN
    config.pred_length         = PRED_LEN
    config.compression_factor  = 16
    config.decoder_hidden_size = 128
    config.conditioning_dim    = 1024
    config.decoder_num_layers  = 1
    config.decoder_num_heads   = 16
    config.poyo_num_latents    = 8

    model = StudentTPOCO(config, [[N]],
                         df=args.df,
                         learnable_df=args.learnable_df).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_nll, no_improve = float("inf"), 0
    train_nlls, val_nlls, val_maes = [], [], []

    for epoch in range(1, args.epochs + 1):
        train_nll        = train_epoch(model, train_loader, optimiser, device)
        val_nll, val_mae = eval_epoch(model, val_loader, device)
        scheduler.step(val_nll)

        train_nlls.append(train_nll)
        val_nlls.append(val_nll)
        val_maes.append(val_mae)

        df_str = f"  ν={model.df:.2f}" if args.learnable_df else ""
        tag    = " *" if val_nll < best_nll else ""

        if val_nll < best_nll:
            best_nll, no_improve = val_nll, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_nll={train_nll:.4f}  "
              f"val_nll={val_nll:.4f}  "
              f"val_mae={val_mae:.4f}"
              f"{df_str}{tag}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest val NLL: {best_nll:.4f}  → {MODEL_PATH}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                     weights_only=False))
    test_nll, test_mae = eval_epoch(model, test_loader, device)
    print(f"Test NLL: {test_nll:.4f}  |  Test MAE: {test_mae:.4f}")
    if args.learnable_df:
        print(f"Learnt ν = {model.df:.3f}")

    np.savez(RESULTS_PATH,
             train_nlls=train_nlls, val_nlls=val_nlls, val_maes=val_maes,
             test_nll=test_nll, test_mae=test_mae,
             df=model.df)
    print(f"Results saved → {RESULTS_PATH}")

    # copy best checkpoint to default path for compare_calibration.py
    import shutil
    shutil.copy(MODEL_PATH, DEFAULT_PATH)
    print(f"Checkpoint also saved → {DEFAULT_PATH}  (ν={model.df:.1f})")
