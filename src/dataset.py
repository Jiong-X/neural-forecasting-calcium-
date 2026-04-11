# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
dataset.py
----------
Data loading pipeline for whole-brain calcium imaging data.

Automatic data retrieval order:
  1. Load from data/processed/0.npz  (fastest — already preprocessed)
  2. Preprocess from data/raw/subject_0/TimeSeries.h5  (if raw exists)
  3. Download subject 0 from Janelia figshare and preprocess  (automatic)

Data source:
  Ahrens lab, Janelia Research Campus
  https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/7272617
"""

import os
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Paths and remote URLs
# ---------------------------------------------------------------------------
PROCESSED_PATH = "data/processed/0.npz"
RAW_PATH       = "data/raw/subject_0/TimeSeries.h5"
ZIP_PATH       = "data/raw/subject_1.zip"

# Figshare subject_1.zip — contains TimeSeries.h5 for subject 0
# Source: https://janelia.figshare.com/articles/dataset/7272617
FIGSHARE_URL   = "https://ndownloader.figshare.com/files/13470404"

# Preprocessing defaults
N_PCS          = 128    # number of principal components to keep


# ---------------------------------------------------------------------------
# Dataset — sliding window (context, target) pairs
# ---------------------------------------------------------------------------

class CalciumDataset(Dataset):
    """
    Wraps a (T, N) trace array into sliding-window (context, target) pairs.

    Expects traces that are already z-scored — normalisation is done once
    over the full recording in get_splits() before this class is called.

    Each item:
        x : (context_len, N)  — input population activity
        y : (pred_len,    N)  — target population activity
    """
    def __init__(self, traces: np.ndarray, context_len: int, pred_len: int):
        self.traces = torch.as_tensor(traces, dtype=torch.float32)
        self.context_len = context_len
        self.pred_len = pred_len
        self.win = context_len + pred_len
        self.n_samples = len(traces) - self.win + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.traces[idx : idx + self.context_len]
        y = self.traces[idx + self.context_len : idx + self.win]
        return x, y


# ---------------------------------------------------------------------------
# Data retrieval helpers
# ---------------------------------------------------------------------------
def _download_raw():
    """Download subject_1.zip from Janelia figshare and extract TimeSeries.h5."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    # --- Step 1: download ZIP ---
    print("Downloading calcium imaging data from Janelia figshare...")
    print(f"  URL : {FIGSHARE_URL}")
    print(f"  Dest: {ZIP_PATH}")
    print("  File is ~2 GB — this may take several minutes.")

    req = urllib.request.Request(
        FIGSHARE_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response, \
         open(ZIP_PATH, "wb") as out_file:
        total      = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block      = 131072   # 128 KB chunks
        while True:
            chunk = response.read(block)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            mb = downloaded / 1024 / 1024
            if total > 0:
                pct = min(downloaded / total * 100, 100)
                print(f"\r  Progress: {pct:.1f}%  ({mb:.1f} MB)", end="", flush=True)
            else:
                print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)
    print("\n  Download complete.")

    # --- Step 2: extract TimeSeries.h5 from ZIP ---
    print("  Extracting TimeSeries.h5 ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # find the HDF5 file inside the ZIP (may be nested in a folder)
        h5_names = [n for n in zf.namelist() if n.endswith("TimeSeries.h5")]
        if not h5_names:
            raise FileNotFoundError(
                "TimeSeries.h5 not found inside the downloaded ZIP. "
                "Please download manually from: "
                "https://janelia.figshare.com/articles/dataset/7272617"
            )
        zf.extract(h5_names[0], "data/raw/subject_0/")
        # move to expected path if nested
        extracted = os.path.join("data/raw/subject_0", h5_names[0])
        if extracted != RAW_PATH and os.path.exists(extracted):
            os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
            os.rename(extracted, RAW_PATH)
    print(f"  Extracted to {RAW_PATH}")

    # clean up ZIP to save disk space
    os.remove(ZIP_PATH)
    print("  ZIP removed.")

def _preprocess_raw_chunked(raw_path: str, out_path: str, n_pcs: int = N_PCS, chunk_size: int = 4096):
    """
    Memory-efficient preprocessing for large HDF5 datasets.
    Computes PCA from XX^T in chunks over neurons.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to preprocess raw HDF5 data.\n"
            "Install with: pip install h5py"
        )

    print(f"Preprocessing {raw_path} ...")
    with h5py.File(raw_path, "r") as f:
        dset = f["CellResp"]   # shape (T, N)
        T, N = dset.shape
        n_pcs = min(n_pcs, T)

        print(f"  Dataset shape: T={T}, N={N}")
        print(f"  Building temporal covariance in chunks of {chunk_size} neurons ...")

        C = np.zeros((T, T), dtype=np.float64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = dset[:, start:end].astype(np.float32)   # (T, chunk)

            # z-score each neuron over time
            mu = X_chunk.mean(axis=0, keepdims=True)
            std = X_chunk.std(axis=0, keepdims=True) + 1e-6
            X_chunk = (X_chunk - mu) / std

            # centre columns
            X_chunk = X_chunk - X_chunk.mean(axis=0, keepdims=True)

            # accumulate XX^T
            C += X_chunk @ X_chunk.T

            print(f"    processed neurons {start}:{end}")

    C /= max(N - 1, 1)

    print("  Eigendecomposing temporal covariance ...")
    evals, evecs = np.linalg.eigh(C)

    # get the top PCs for specific idx
    idx = np.argsort(evals)[::-1]
    evals = evals[idx][:n_pcs]
    evecs = evecs[:, idx][:, :n_pcs]

    scores = evecs * np.sqrt(np.clip(evals, 0, None))
    PC = scores.T.astype(np.float32)   # (n_pcs, T)

    # z-score each PC over time
    pc_mu = PC.mean(axis=1, keepdims=True)
    pc_std = PC.std(axis=1, keepdims=True) + 1e-6
    PC = ((PC - pc_mu) / pc_std).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, PC=PC)

    print(f"Saved processed PCs to {out_path} with shape {PC.shape}")

def _load_traces() -> np.ndarray:
    """
    Return (T, N_PCS) float32 trace array.
    Auto-downloads and preprocesses if needed.
    Pre-processing includes Z-scoring
    """
    # 1. Already preprocessed
    if os.path.exists(PROCESSED_PATH):
        data = np.load(PROCESSED_PATH)
        raw  = data["PC"].astype(np.float32)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T                        # ensure (T, N)
        return raw[:, :N_PCS]

    # Nothing local — download then preprocess
    if not os.path.exists(RAW_PATH):
         _download_raw()

  
    _preprocess_raw_chunked(RAW_PATH, PROCESSED_PATH, n_pcs=N_PCS)
    
    return _load_traces()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# POCO paper temporal split ratios (Section 4 / Appendix A)
TRAIN_FRAC = 0.6   # first 60 % of recording
VAL_FRAC   = 0.2   # next  20 %
# TEST_FRAC = 0.2  # final 20 % — held out, never used during training


def get_splits(seq_length:  int   = 64,
               pred_length: int   = 16,
               train_frac:  float = TRAIN_FRAC,
               val_frac:    float = VAL_FRAC):
    """
    Return (train_ds, val_ds, test_ds) for the 60/20/20 temporal split
    used in the POCO paper.

    Split bounds:
      train : traces[0         : train_end]  — 60 %
      val   : traces[train_end : val_end  ]  — 20 %
      test  : traces[val_end   : end      ]  — 20 %

    Data is retrieved automatically:
      - loads data/processed/0.npz if available
      - preprocesses data/raw/subject_0/TimeSeries.h5 if available
      - downloads from Janelia figshare otherwise (~2 GB, first run only)

    Returns:
        train_ds, val_ds, test_ds : CalciumDataset
    """
    context_len = seq_length - pred_length

    traces = _load_traces()        # (T, N) raw float32
    T      = traces.shape[0]

    train_end = int(T * train_frac)
    val_end   = int(T * (train_frac + val_frac))

    print(f"Data loaded  : {T} timesteps × {traces.shape[1]} PCs  (z-scored over full recording)")
    print(f"Split bounds : train 0–{train_end}  |  "
          f"val {train_end}–{val_end}  |  test {val_end}–{T}")

    train_ds = CalciumDataset(traces[:train_end],          context_len, pred_length)
    val_ds   = CalciumDataset(traces[train_end:val_end],   context_len, pred_length)
    test_ds  = CalciumDataset(traces[val_end:],            context_len, pred_length)

    print(f"Windows      : train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Backwards-compatible shims
# ---------------------------------------------------------------------------

def get_dataset(seq_length: int = 64, pred_length: int = 16,
                train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Returns (train_ds, val_ds). Use get_splits() for all three at once."""
    train_ds, val_ds, _ = get_splits(seq_length, pred_length, train_frac, val_frac)
    return train_ds, val_ds


def get_test_dataset(seq_length: int = 64, pred_length: int = 16,
                     train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Returns the held-out test_ds only. Use get_splits() for all three at once."""
    _, _, test_ds = get_splits(seq_length, pred_length, train_frac, val_frac)
    return test_ds
