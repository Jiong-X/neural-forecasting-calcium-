import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import _load_traces

from typing import Dict, Tuple
from abc import abstractmethod, ABC

# ---------------------------------------------------------------------------
# Dataset 
# ---------------------------------------------------------------------------

class _Abstract_CalciumDataset(ABC, Dataset):
    @abstractmethod
    def __init__(): ...
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# Recurrent approach
class RecurrentCalciumDataset(_Abstract_CalciumDataset):
    def __init__(self, traces: np.ndarray, seq_len: int = 50, pred_steps: int = 1):
        X, y = [], []
        for t in range(len(traces) - seq_len - pred_steps + 1):
            X.append(traces[t : t + seq_len])
            y.append(traces[t + seq_len : t + seq_len + pred_steps])

        self.X = torch.tensor(np.array(X))
        self.y = torch.tensor(np.array(y))

# sliding-window approach
class CalciumDataset(_Abstract_CalciumDataset):
    def __init__(self, traces: np.ndarray, seq_len: int, pred_len: int):
        context_len = seq_len - pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])
        self.X = torch.tensor(np.array(X))   # (S, context_len, N)
        self.Y = torch.tensor(np.array(Y))   # (S, pred_len,    N)

def collate_fn(batch):
    X = torch.stack([b[0] for b in batch], dim=1)   # (context_len, B, N)
    Y = torch.stack([b[1] for b in batch], dim=1)   # (pred_len,    B, N)
    return X, Y

_DATASET_CACHE: Dict[str, Tuple[_Abstract_CalciumDataset, Dict]] = {
    "TSMixer":(CalciumDataset, {"collate_fn":collate_fn}),
    "TexFilter":(CalciumDataset, {"collate_fn":collate_fn}),
    "RNN":(RecurrentCalciumDataset, {}),
    "NLinear":(CalciumDataset ,{}),
    "LSTM":(RecurrentCalciumDataset, {}),
    "DLinear":(CalciumDataset, {}),
    "AR":(..., {}),
}

def fetch_data_loaders(model_type:str,
                seq_length: int = 64,
                pred_length: int = 16,
                train_frac: float = 0.6,
                val_frac: float   = 0.2,
                batch_size: float = 64) -> Tuple[DataLoader, DataLoader, int]:
    """
    Returns (train_dataset, val_dataset) of CalciumDataset objects.

    Data is retrieved automatically:
      - loads data/processed/0.npz if available
      - preprocesses data/raw/subject_0/TimeSeries.h5 if available
      - downloads from Janelia figshare otherwise

    Args:
        seq_length   : total window length = context_len + pred_length
        pred_length  : number of steps to predict
        train_frac   : fraction of timesteps for training   (default 0.6)
        val_frac     : fraction of timesteps for validation (default 0.2)
                       remaining 0.2 is held as test set

    Returns:
        train_dataset : CalciumDataset
        val_dataset   : CalciumDataset
    """
    if model_type not in _DATASET_CACHE.keys():
        print(f"invalid model type '{model_type}', must be one of '{_DATASET_CACHE.keys()}'")

    traces = _load_traces()
    T = traces.shape[0]
    n_neurons = traces.shape[1]
    print(f"Loaded traces: {T} timesteps x {traces.shape[1]} PCs")

    train_end = int(T * train_frac)
    val_end   = int(T * (train_frac + val_frac))

    Dataset_obj, args = _DATASET_CACHE[model_type]

    train_ds = Dataset_obj(traces[:train_end],         seq_length, pred_length)
    val_ds   = Dataset_obj(traces[train_end:val_end],  seq_length, pred_length)

    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")
    
    train_args = {"dataset":train_ds, "batch_size":batch_size, "shuffle":True,"drop_last":True}
    val_args = {"dataset":val_ds, "batch_size":batch_size, "shuffle":True,"drop_last":True}
    train_args.update(args)
    val_args.update(args)

    train_loader = DataLoader(**train_args)
    val_loader   = DataLoader(**val_args)
    return train_loader, val_loader, n_neurons

# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device):
    model.train()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimiser.zero_grad()
        pred = model([X])[0]             # (pred_len, B, N)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total += loss.item() * X.shape[1]
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    for X, Y in loader:
        X, Y  = X.to(device), Y.to(device)
        pred  = model([X])[0]
        total_mse += F.mse_loss(pred, Y).item() * X.shape[1]
        total_mae += (pred - Y).abs().mean().item() * X.shape[1]
        n += X.shape[1]
    return total_mse / n, total_mae / n