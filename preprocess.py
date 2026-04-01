"""
preprocess.py
-------------
Converts raw Zebrafish Ahrens calcium imaging data into preprocessed .npz
files ready for model training.

Raw data source
---------------
  ~/POCO/data/raw_zebrafish_ahrens/subject_N/TimeSeries.h5
  Keys used:
    CellResp  (T, n_neurons) float32 — motion-corrected fluorescence traces.
                                        Already a corrected signal; no ΔF/F needed.

Preprocessing pipeline (per subject)
--------------------------------------
  1. Load CellResp  — shape (T, n_neurons)  →  transpose to (n_neurons, T)
  2. ROI filtering  — keep all neurons (valid_indices = all True)
                      [extend here if you want to filter by e.g. baseline > threshold]
  3. Z-score        — per-neuron: subtract mean, divide by std + 1e-6
  4. PCA            — reduce to top N_PCS principal components using sklearn
  5. Save           — writes data/processed/{idx}.npz with keys:
                        valid_indices  (n_neurons,)  bool
                        M              (n_neurons, T) float32   z-scored traces
                        PC             (N_PCS, T)    float32   top PCs

Usage
-----
  /home/jiongx/micromamba/envs/poco/bin/python3 preprocess.py

  Optional args:
    --raw_dir   path to raw zebrafish ahrens directory  (default: ~/POCO/data/raw_zebrafish_ahrens)
    --out_dir   path to output processed directory      (default: data/processed)
    --n_pcs     number of principal components          (default: 2048)
    --subjects  which subject IDs to process            (default: all available)
"""

import argparse
import os
import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("preprocess.py requires h5py: pip install h5py")

try:
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError("preprocess.py requires scikit-learn: pip install scikit-learn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zscore(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each neuron (row) independently.

    Args:
        data: (n_neurons, T) float32

    Returns:
        normalised: (n_neurons, T) float32  — zero mean, unit variance per neuron
    """
    mu  = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1,  keepdims=True) + 1e-6
    return (data - mu) / std


def run_pca(data: np.ndarray, n_components: int) -> np.ndarray:
    """
    Run PCA on z-scored neural traces.

    Args:
        data:        (n_neurons, T)  — z-scored traces
        n_components: number of PCs to retain

    Returns:
        pcs: (n_components, T)  — principal component time series

    Note: sklearn PCA expects (samples, features) = (T, n_neurons), so we
    transpose before fitting and transpose the result back.
    """
    n_components = min(n_components, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    # fit on (T, n_neurons), get scores (T, n_components)
    scores = pca.fit_transform(data.T)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"    PCA: {n_components} components explain "
          f"{cumvar[-1] * 100:.1f}% of variance")
    print(f"    Top-10 individual: "
          f"{pca.explained_variance_ratio_[:10].round(3).tolist()}")
    return scores.T.astype(np.float32)   # (n_components, T)


def load_subject(h5_path: str) -> np.ndarray:
    """
    Load CellResp from a TimeSeries.h5 file.

    The raw array is (T, n_neurons); we return (n_neurons, T).
    """
    with h5py.File(h5_path, "r") as f:
        cell_resp = np.array(f["CellResp"], dtype=np.float32)   # (T, n_neurons)
    return cell_resp.T   # (n_neurons, T)


def process_subject(
    h5_path:      str,
    n_pcs:        int = 2048,
    roi_indices:  np.ndarray = None,
) -> dict:
    """
    Full preprocessing pipeline for one subject.

    Args:
        h5_path:     path to TimeSeries.h5
        n_pcs:       number of PCs to compute
        roi_indices: optional boolean mask (n_neurons,); None = keep all

    Returns:
        dict with keys: valid_indices, M, PC
    """
    print(f"  Loading {h5_path} ...")
    data = load_subject(h5_path)                        # (n_neurons, T)
    n_neurons, T = data.shape
    print(f"  Shape: {n_neurons} neurons × {T} time steps")

    # Step 1 — ROI filtering
    if roi_indices is None:
        roi_indices = np.ones(n_neurons, dtype=bool)    # keep all
    n_valid = roi_indices.sum()
    print(f"  ROI filter: {n_valid}/{n_neurons} neurons kept")

    # Step 2 — Z-score (all neurons, before ROI selection for M)
    M = zscore(data)                                    # (n_neurons, T)

    # Step 3 — PCA on valid neurons only
    valid_data = M[roi_indices]                         # (n_valid, T)
    print(f"  Running PCA (n_components={n_pcs}) ...")
    PC = run_pca(valid_data, n_pcs)                     # (n_pcs, T)

    return {
        "valid_indices": roi_indices,   # (n_neurons,) bool
        "M":             M,             # (n_neurons, T) float32  z-scored
        "PC":            PC,            # (n_pcs, T)    float32  PCA scores
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Zebrafish Ahrens calcium imaging data")
    parser.add_argument("--raw_dir",  default=os.path.expanduser("~/POCO/data/raw_zebrafish_ahrens"),
                        help="Directory containing subject_N subdirectories")
    parser.add_argument("--out_dir",  default="data/processed",
                        help="Output directory for .npz files")
    parser.add_argument("--n_pcs",    type=int, default=2048,
                        help="Number of principal components to retain")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject IDs to process (default: all available)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover available subjects
    all_subjects = sorted([
        int(d.split("_")[1])
        for d in os.listdir(args.raw_dir)
        if d.startswith("subject_") and
           os.path.isfile(os.path.join(args.raw_dir, d, "TimeSeries.h5"))
    ])
    subjects = args.subjects if args.subjects else all_subjects
    print(f"Found subjects: {all_subjects}")
    print(f"Processing:     {subjects}")
    print(f"Output dir:     {args.out_dir}")
    print(f"n_PCs:          {args.n_pcs}\n")

    for out_idx, subject_id in enumerate(subjects):
        h5_path = os.path.join(args.raw_dir, f"subject_{subject_id}", "TimeSeries.h5")
        if not os.path.isfile(h5_path):
            print(f"[SKIP] subject_{subject_id}: TimeSeries.h5 not found")
            continue

        print(f"[{out_idx}] Processing subject_{subject_id} ...")
        try:
            result = process_subject(h5_path, n_pcs=args.n_pcs)
            out_path = os.path.join(args.out_dir, f"{out_idx}.npz")
            np.savez(out_path, **result)
            print(f"  Saved → {out_path}\n")
        except Exception as e:
            print(f"  [ERROR] subject_{subject_id}: {e}\n")

    print("Done.")
