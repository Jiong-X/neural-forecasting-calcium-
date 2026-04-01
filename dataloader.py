"""
dataloader.py
-------------
Source-agnostic data loading for neural time-series forecasting.

Supports four input formats out of the box:

  1. Preprocessed .npz  (output of preprocess.py)
       Keys looked for (in priority order): PC, M, data, traces
       Shape expected: (T, N)  where T = timesteps, N = channels/neurons

  2. Raw HDF5 / .h5  (e.g. Ahrens lab TimeSeries.h5)
       Default key: "CellResp"  —  shape (T, N)
       Pass h5_key= to override.

  3. Plain NumPy .npy  —  shape (T, N)

  4. CSV / TSV  —  rows = timesteps, columns = channels (no header assumed
       unless csv_header=True).  Whitespace or comma delimited.

The same CalciumDataset and get_loaders() API is used by all training scripts,
so swapping data sources only requires changing the path / format argument.

Usage (standalone)
------------------
  from dataloader import load_traces, get_loaders

  # from preprocessed .npz
  train_loader, val_loader, N = get_loaders("data/processed/0.npz")

  # from a raw HDF5 file
  train_loader, val_loader, N = get_loaders("data/raw/TimeSeries.h5",
                                            fmt="h5", h5_key="CellResp")

  # from a plain numpy array
  import numpy as np
  arr = np.load("my_data.npy")          # (T, N)
  train_loader, val_loader, N = get_loaders(arr)

Command-line demo
-----------------
  python dataloader.py --path data/processed/0.npz
  python dataloader.py --path data/raw/TimeSeries.h5 --fmt h5 --h5_key CellResp
  python dataloader.py --path data/my_traces.csv --fmt csv
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ───────────────────────────────────────────────────────────────────────────
# 1. Format-agnostic loader
# ───────────────────────────────────────────────────────────────────────────

# Keys to try (in order) when loading an .npz file
_NPZ_KEYS = ("PC", "M", "data", "traces")


def load_traces(
    source,
    fmt: str = "auto",
    n_channels: int | None = None,
    h5_key: str = "CellResp",
    csv_header: bool = False,
    zscore: bool = True,
) -> np.ndarray:
    """
    Load neural traces from any supported source and return a (T, N) float32 array.

    Parameters
    ----------
    source : str | np.ndarray
        File path (str) or pre-loaded array (np.ndarray).
    fmt : str
        One of "auto", "npz", "h5", "npy", "csv".
        "auto" infers from the file extension.
    n_channels : int | None
        If set, only the first n_channels columns are kept.
    h5_key : str
        Dataset key inside an HDF5 file (default "CellResp").
    csv_header : bool
        Whether the CSV has a header row to skip.
    zscore : bool
        If True, z-score each channel independently before returning.
        Set False if data is already normalised (e.g. preprocessed .npz).

    Returns
    -------
    traces : np.ndarray  shape (T, N), float32
    """
    # ── Already a numpy array ──────────────────────────────────────────────
    if isinstance(source, np.ndarray):
        traces = source.astype(np.float32)
        if traces.ndim == 1:
            traces = traces[:, None]
        if n_channels is not None:
            traces = traces[:, :n_channels]
        if zscore:
            traces = _zscore(traces)
        return traces

    path = os.path.expanduser(str(source))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # ── Auto-detect format ─────────────────────────────────────────────────
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = {".npz": "npz", ".npy": "npy",
               ".h5": "h5", ".hdf5": "h5",
               ".csv": "csv", ".tsv": "csv"}.get(ext, "npz")

    # ── Load ───────────────────────────────────────────────────────────────
    if fmt == "npz":
        data = np.load(path, allow_pickle=False)
        traces = None
        for key in _NPZ_KEYS:
            if key in data:
                traces = data[key].astype(np.float32)
                break
        if traces is None:
            raise KeyError(
                f"None of the expected keys {_NPZ_KEYS} found in {path}.\n"
                f"Available keys: {list(data.keys())}"
            )
        # Ensure 2-D
        if traces.ndim == 1:
            traces = traces[:, None]
        # preprocess.py stores PC as (N_PCS, T) — transpose if needed.
        # PC scores are NOT uniformly z-scored (variance decreases per component),
        # so z-scoring is applied here unless the caller explicitly sets zscore=False.

    elif fmt == "h5":
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to load HDF5 files: pip install h5py")
        with h5py.File(path, "r") as f:
            if h5_key not in f:
                raise KeyError(
                    f"Key '{h5_key}' not found in {path}.\n"
                    f"Available keys: {list(f.keys())}"
                )
            traces = f[h5_key][()].astype(np.float32)
        # HDF5 from Ahrens lab: (T, N) — already correct orientation
        if traces.ndim == 1:
            traces = traces[:, None]

    elif fmt == "npy":
        traces = np.load(path).astype(np.float32)
        if traces.ndim == 1:
            traces = traces[:, None]

    elif fmt == "csv":
        skip = 1 if csv_header else 0
        traces = np.genfromtxt(path, delimiter=None, skip_header=skip,
                               dtype=np.float32)
        if traces.ndim == 1:
            traces = traces[:, None]

    else:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: npz, h5, npy, csv.")

    # ── Optional channel subset ────────────────────────────────────────────
    if n_channels is not None:
        traces = traces[:, :n_channels]

    # ── Optional z-score ──────────────────────────────────────────────────
    if zscore:
        traces = _zscore(traces)

    return traces


def _zscore(traces: np.ndarray) -> np.ndarray:
    """Z-score each channel (column) independently."""
    mu  = traces.mean(axis=0, keepdims=True)
    std = traces.std(axis=0,  keepdims=True) + 1e-8
    return (traces - mu) / std


# ───────────────────────────────────────────────────────────────────────────
# 2. Sliding-window Dataset
# ───────────────────────────────────────────────────────────────────────────

class CalciumDataset(Dataset):
    """
    Sliding-window dataset for neural time-series forecasting.

    Given traces of shape (T, N), returns (context, target) pairs:
        x : (seq_len,  N)  float32  — input context window
        y : (pred_len, N)  float32  — forecast target

    Parameters
    ----------
    traces   : np.ndarray  (T, N)
    seq_len  : int   context window length
    pred_len : int   forecast horizon
    stride   : int   step between successive windows (default 1)
    """

    def __init__(
        self,
        traces: np.ndarray,
        seq_len: int = 128,
        pred_len: int = 16,
        stride: int = 1,
    ):
        traces = np.asarray(traces, dtype=np.float32)
        assert traces.ndim == 2, f"Expected (T, N), got {traces.shape}"

        self.seq_len  = seq_len
        self.pred_len = pred_len
        window        = seq_len + pred_len

        indices = range(0, len(traces) - window + 1, stride)
        X = np.stack([traces[i : i + seq_len] for i in indices])
        y = np.stack([traces[i + seq_len : i + window] for i in indices])

        self.X = torch.from_numpy(X)   # (S, seq_len, N)
        self.y = torch.from_numpy(y)   # (S, pred_len, N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def n_channels(self) -> int:
        return self.X.shape[-1]


# ───────────────────────────────────────────────────────────────────────────
# 3. Convenience factory — returns DataLoaders ready for training
# ───────────────────────────────────────────────────────────────────────────

def get_loaders(
    source,
    seq_len: int = 128,
    pred_len: int = 16,
    val_split: float = 0.2,
    batch_size: int = 32,
    stride: int = 1,
    n_channels: int | None = None,
    fmt: str = "auto",
    h5_key: str = "CellResp",
    csv_header: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """
    Load data, build a sliding-window dataset, and return train/val DataLoaders.

    Parameters
    ----------
    source      : str | np.ndarray   file path or pre-loaded (T, N) array
    seq_len     : int                 context window length
    pred_len    : int                 forecast horizon
    val_split   : float               fraction of *timesteps* used for validation
                                      (temporal split — val is always the tail)
    batch_size  : int
    stride      : int                 window stride
    n_channels  : int | None          keep only first n_channels; None = all
    fmt         : str                 "auto" | "npz" | "h5" | "npy" | "csv"
    h5_key      : str                 HDF5 dataset key (default "CellResp")
    csv_header  : bool                skip first row of CSV
    num_workers : int
    drop_last   : bool                drop last incomplete batch

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    N            : int   number of channels (neurons / PCs)
    """
    traces = load_traces(source, fmt=fmt, n_channels=n_channels,
                         h5_key=h5_key, csv_header=csv_header)
    T, N = traces.shape

    split      = int(T * (1.0 - val_split))
    train_ds   = CalciumDataset(traces[:split], seq_len=seq_len,
                                pred_len=pred_len, stride=stride)
    val_ds     = CalciumDataset(traces[split:], seq_len=seq_len,
                                pred_len=pred_len, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=drop_last,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=drop_last,
                              pin_memory=torch.cuda.is_available())

    print(f"[dataloader] {T} timesteps | {N} channels | "
          f"train={len(train_ds)} / val={len(val_ds)} windows")
    return train_loader, val_loader, N


# ───────────────────────────────────────────────────────────────────────────
# 4. CLI demo
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader demo")
    parser.add_argument("--path",       required=True,
                        help="Path to data file (.npz / .h5 / .npy / .csv)")
    parser.add_argument("--fmt",        default="auto",
                        choices=["auto", "npz", "h5", "npy", "csv"])
    parser.add_argument("--h5_key",     default="CellResp")
    parser.add_argument("--csv_header", action="store_true")
    parser.add_argument("--n_channels", type=int, default=None)
    parser.add_argument("--seq_len",    type=int, default=128)
    parser.add_argument("--pred_len",   type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split",  type=float, default=0.2)
    args = parser.parse_args()

    train_loader, val_loader, N = get_loaders(
        source      = args.path,
        fmt         = args.fmt,
        h5_key      = args.h5_key,
        csv_header  = args.csv_header,
        n_channels  = args.n_channels,
        seq_len     = args.seq_len,
        pred_len    = args.pred_len,
        batch_size  = args.batch_size,
        val_split   = args.val_split,
    )

    x, y = next(iter(train_loader))
    print(f"Sample batch — x: {tuple(x.shape)}  y: {tuple(y.shape)}")
    print(f"x range  [{x.min():.3f}, {x.max():.3f}]")
    print(f"y range  [{y.min():.3f}, {y.max():.3f}]")
