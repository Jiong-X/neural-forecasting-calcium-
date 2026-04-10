from dataclasses import dataclass, field
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import _load_traces

from typing import Dict, Tuple, Union, Optional

# ---------------------------------------------------------------------------
# Dataset 
# ---------------------------------------------------------------------------

# sliding-window approach
class CalciumDataset():
    def __init__(self, traces: np.ndarray, context_len: int, pred_len: int):
        seq_len = context_len + pred_len
        X, Y = [], []
        for t in range(len(traces) - seq_len + 1):
            X.append(traces[t : t + context_len])
            Y.append(traces[t + context_len : t + seq_len])
        self.X = torch.tensor(np.array(X))   # (S, context_len, N)
        self.Y = torch.tensor(np.array(Y))   # (S, pred_len,    N)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def collate_fn(batch):
    X = torch.stack([b[0] for b in batch], dim=1)   # (context_len, B, N)
    Y = torch.stack([b[1] for b in batch], dim=1)   # (pred_len,    B, N)
    return X, Y

_DATASET_CACHE: Dict[str, Dict] = {
    "ProbabilisticPOCO":{},
    "TSMixer":{"collate_fn":collate_fn},
    "TexFilter":{"collate_fn":collate_fn},
    "RNN":{},
    "NLinear":{},
    "LSTM":{},
    "DLinear":{},
    "AR":{},
}

def fetch_data_loaders(model_type:str,
                context_length: int = 48,
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

    args = _DATASET_CACHE[model_type]

    train_ds = CalciumDataset(traces[:train_end],         context_length, pred_length)
    val_ds   = CalciumDataset(traces[train_end:val_end],  context_length, pred_length)

    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")
    
    train_args = {"dataset":train_ds, "batch_size":batch_size, "shuffle":True,"drop_last":True}
    val_args = {"dataset":val_ds, "batch_size":batch_size, "shuffle":True,"drop_last":True}
    train_args.update(args)
    val_args.update(args)

    train_loader = DataLoader(**train_args)
    val_loader   = DataLoader(**val_args)
    return train_loader, val_loader, n_neurons

@dataclass
class trainingConfig:
    model_name: str = field()
    seed: Union[int, None] = field(default=42)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")  
    sequence_length: int = field(default=64)       # context (48) + horizon (16)
    pred_length: int = field(default=16)       # prediction horizon

    n_channels: int = field(default=512)      # top-128 principal components
    batch_size: int = field(default=64)
    epochs: int = field(default=50)
    patience    = field(default=10)       # early stopping patience (epochs)
    SAVE_FOLDER   = field(default="models/saved")
    RESULTS_FOLDER = field(default="results")

    def __post_init__(self):
        if self.model_name not in _DATASET_CACHE.keys():
            raise ValueError(f"invalid model name '{self.model_name}', must be one of '{_DATASET_CACHE.keys()}'")
        if type(self.seed) not in [int, type(None)]:
            raise TypeError(f"seed must either be 'None' or an 'int', got: {self.seed}")

    @property
    def save_path(self) -> str:
        if self.model_name == "ProbabilisticPOCO": # for backwards compatibility with old code, base model was stored under model.pt
            return os.path.join(self.SAVE_FOLDER, f"model.pt") 
        return os.path.join(self.SAVE_FOLDER, f"best_{self.model_name}.pt")

    @property
    def results_path(self) -> str:
        if self.model_name == "ProbabilisticPOCO": # for backwards compatibility with old code, base model losses were stored under train_losses
            return os.path.join(self.RESULTS_FOLDER, f"train_losses.npz")
        return os.path.join(self.RESULTS_FOLDER, f"{self.model_name}_train_losses.npz")
    
