# GEN AI STATEMENT
# this script was fully written by a human


from dataclasses import dataclass, field
import os
import numpy as np
import torch

from typing import Union

@dataclass
class trainingConfig:
    model_name: str = field()
    seed: Union[int, None] = field(default=42)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")  
    sequence_length: int = field(default=64)       # context (48) + horizon (16)
    pred_length: int = field(default=16)       # prediction horizon

    n_channels: int = field(default=128)      # top-128 principal components
    batch_size: int = field(default=64)
    epochs: int = field(default=50)
    patience:int    = field(default=10)       # early stopping patience (epochs)
    SAVE_FOLDER:str   = field(default="models/saved")
    RESULTS_FOLDER:str = field(default="results")

    def __post_init__(self):
        if type(self.seed) not in [int, type(None)]:
            raise TypeError(f"seed must either be 'None' or an 'int', got: {self.seed}")

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True   # force deterministic CUDA kernels
            torch.backends.cudnn.benchmark = False  # disable auto-tuner (picks same algo each run)

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
    
