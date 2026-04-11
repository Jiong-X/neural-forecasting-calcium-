# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
train.py
--------
End-to-end training pipeline for probabilistic neural activity forecasting.

Data is retrieved automatically:
  - loads data/processed/0.npz if available
  - preprocesses data/raw/subject_0/TimeSeries.h5 if available
  - downloads from Janelia figshare otherwise (~2 GB, first run only)

Run:
    python train.py

Outputs:
    models/saved/model.pt          best model checkpoint
    results/train_losses.npz       loss curves (train NLL, val NLL, val MAE)
"""

import torch

from src.model      import ProbabilisticForecaster

from src.metrics import MetricSuite, GaussianNllLoss, MAELoss, StudentTNllLoss, MSELoss
from src.util import trainingConfig
from src.trainer import train
     
if __name__ == "__main__":
    config = trainingConfig(model_name="ProbabilisticPOCO")
    model = ProbabilisticForecaster(
    seq_length  = config.sequence_length,
    pred_length = config.pred_length,
    n_channels  = config.n_channels,
    ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), MSELoss(RMSE=True), StudentTNllLoss()], primary=GaussianNllLoss())
    train(model, config, optimizer, criterion)

"""
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH  = 64       # context (48) + horizon (16)
PRED_LENGTH = 16       # prediction horizon
N_CHANNELS  = 128      # top-128 principal components
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 3e-4     # AdamW learning rate (paper default)
WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
PATIENCE    = 10       # early stopping patience (epochs)
SAVE_PATH   = "models/saved/model.pt"
RESULTS_PATH= "results/train_losses.npz"

model = ProbabilisticForecaster(
    seq_length  = SEQ_LENGTH,
    pred_length = PRED_LENGTH,
    n_channels  = N_CHANNELS,
).to(DEVICE)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
"""