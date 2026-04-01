# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
dataset.py
----------
Data loading and preprocessing pipeline.
Replace the placeholder logic with your chosen dataset.
"""

import torch
from torch.utils.data import Dataset, random_split


class TimeSeriesDataset(Dataset):
    """
    Wraps a time series into (context, target) pairs.
    x: (seq_length,)  — input context window
    y: (pred_length,) — future values to predict
    """

    def __init__(self, data, seq_length, pred_length):
        self.data        = data
        self.seq_length  = seq_length
        self.pred_length = pred_length
        self.total       = seq_length + pred_length

    def __len__(self):
        return len(self.data) - self.total + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length : idx + self.total]
        return x.float(), y.float()


def get_dataset(seq_length=64, pred_length=16, test=False, val_split=0.2):
    """
    Downloads / loads the dataset and returns train/val (or val/test) splits.
    TODO: replace the synthetic data below with your real dataset.
    """
    # ── Placeholder: replace with real data loading ───────────────────────────
    N    = 10_000
    t    = torch.linspace(0, 100, N)
    data = torch.sin(t) + 0.1 * torch.randn(N)
    # ─────────────────────────────────────────────────────────────────────────

    dataset   = TimeSeriesDataset(data, seq_length, pred_length)
    n_val     = int(len(dataset) * val_split)
    n_train   = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    return train_set, val_set
