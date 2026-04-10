"""
Simple Autoregressive (AR) model for forecasting neural activity
from calcium imaging data.

The model is a multivariate AR(p):

    y_t = A_1 * y_{t-1} + A_2 * y_{t-2} + ... + A_p * y_{t-p} + b

Coefficients are solved in closed form via ordinary least squares
(no gradient-based training needed).  Multi-step forecasts are
produced by rolling the model forward autoregressively.

Input data
----------
  Expects data/processed/0.npz with a "PC" key of shape (N_pcs, T).
  Input size is inferred automatically.  Set N_PCS to an integer to
  cap the number of components (e.g. 128), or None to use all.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CalciumAR:
    """
    Multivariate linear AR(p) model fitted by ordinary least squares.

    Args:
        order: lag order p — how many past time steps to use.
    """

    def __init__(self, order: int = 10):
        self.order   = order
        self.weights = None   # (p*N + 1, N)  fitted coefficients + bias

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, traces: np.ndarray) -> None:
        """
        Fit AR coefficients via OLS on a (T, N) trace array.

        Builds the design matrix X of shape (T-p, p*N+1) and solves
        X @ W ≈ Y  in least-squares sense.
        """
        T, N = traces.shape
        p    = self.order

        # Build design matrix: each row = [y_{t-1}, ..., y_{t-p}, 1]
        rows = T - p
        X = np.empty((rows, p * N + 1), dtype=np.float64)
        for i in range(rows):
            X[i, :-1] = traces[i : i + p].ravel()   # flatten p frames
            X[i,  -1] = 1.0                          # bias term

        Y = traces[p:]   # (rows, N) — targets

        # Closed-form OLS: W = (X^T X)^{-1} X^T Y  via lstsq
        self.weights, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (p*N+1, N)
        print(f"AR({p}) fitted on {rows} samples, {N} features.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _step(self, window: np.ndarray) -> np.ndarray:
        """
        Predict one step ahead.

        Args:
            window: (p, N) — last p observed frames.
        Returns:
            pred:   (N,)
        """
        x = np.append(window.ravel(), 1.0)   # (p*N + 1,)
        return x @ self.weights              # (N,)

    def forecast(self, context: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Autoregressive multi-step forecast.

        Args:
            context: (T, N) — observed history; the last `order` frames
                     are used as the initial window.
            horizon: number of future steps to predict.
        Returns:
            preds:   (horizon, N)
        """
        assert self.weights is not None, "Call fit() before forecast()."
        window = context[-self.order:].copy().astype(np.float64)  # (p, N)
        preds  = []

        for _ in range(horizon):
            pred   = self._step(window)             # (N,)
            preds.append(pred)
            window = np.vstack([window[1:], pred])  # slide window forward

        return np.array(preds)   # (horizon, N)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, traces: np.ndarray, horizon: int = 1) -> dict:
        """
        Compute MSE and MAE on a held-out trace.

        Args:
            traces:  (T, N)
            horizon: forecast horizon to evaluate.
        Returns:
            dict with 'mse' and 'mae'.
        """
        p = self.order
        mse_list, mae_list = [], []

        for t in range(p, len(traces) - horizon):
            context = traces[:t]
            pred    = self.forecast(context, horizon=horizon)   # (horizon, N)
            target  = traces[t : t + horizon]                   # (horizon, N)
            mse_list.append(np.mean((pred - target) ** 2))
            mae_list.append(np.mean(np.abs(pred - target)))

        return {"mse": float(np.mean(mse_list)),
                "mae": float(np.mean(mae_list))}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # --- Paths ---
    DATA_PATH    = "data/processed/0.npz"
    MODEL_PATH   = "models/best_calcium_ar.npz"
    RESULTS_PATH = "results/ar_metrics.npz"
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Hyperparameters ---
    N_PCS    = 128   # None = use all PCs; int to cap (e.g. 128)
    ORDER    = 48    # AR lag order p — matches paper context C=48
    HORIZON  = 16    # forecast horizon — matches paper P=16
    VAL_FRAC = 0.2

    # --- Load data ---
    print(f"Loading {DATA_PATH} ...")
    data = np.load(DATA_PATH)

    if "PC" in data:
        raw = data["PC"].astype(np.float64)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T                          # → (T, N)
    elif "M" in data:
        raw = data["M"].astype(np.float64)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if "valid_indices" in data:
            raw = raw[:, data["valid_indices"]]
    else:
        raise ValueError(f"No recognised key in {DATA_PATH}. Found: {list(data.keys())}")

    traces = raw[:, :N_PCS] if N_PCS is not None else raw

    # z-score normalise
    mu     = traces.mean(axis=0, keepdims=True)
    sd     = traces.std(axis=0,  keepdims=True) + 1e-8
    traces = (traces - mu) / sd

    T, N = traces.shape
    print(f"Traces shape: {traces.shape}  (T={T}, features={N})")

    # --- Train / val split ---
    split      = int(T * (1 - VAL_FRAC))
    train_data = traces[:split]
    val_data   = traces[split:]
    print(f"Train steps: {len(train_data)}  |  Val steps: {len(val_data)}")

    # --- Fit (closed-form OLS — no training loop) ---
    model = CalciumAR(order=ORDER)
    model.fit(train_data)

    # --- Evaluate ---
    print(f"\nEvaluating on validation set (horizon={HORIZON}) ...")
    metrics = model.evaluate(val_data, horizon=HORIZON)
    print(f"  MSE : {metrics['mse']:.6f}")
    print(f"  MAE : {metrics['mae']:.6f}")

    # --- Save weights and metrics ---
    np.savez(MODEL_PATH, weights=model.weights, order=ORDER)
    np.savez(RESULTS_PATH, **metrics)
    print(f"\nWeights saved to {MODEL_PATH}")
    print(f"Metrics  saved to {RESULTS_PATH}")

    # --- Example: forecast HORIZON steps ahead from the end of val data ---
    preds = model.forecast(val_data, horizon=HORIZON)
    print(f"\nExample {HORIZON}-step forecast shape: {preds.shape}")
