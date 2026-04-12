import torch

from src.baseline_models.MLP import MLPHead
from src.model      import DeterministicPOCO
from src.metrics import MetricSuite, GaussianNllLoss, MAELoss, MSELoss, StudentTNllLoss
from src.util import trainingConfig
from src.trainer import train


def run_MLP():
     
    config = trainingConfig(model_name="MLP")
    model = MLPHead(
        n_neurons=config.n_channels, context_len=(config.sequence_length - config.pred_length), cond_dim=1024, pred_len=config.pred_length,
    ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), MSELoss(RMSE=True), StudentTNllLoss()], primary=GaussianNllLoss())
    train(model, config, optimizer, criterion)

def run_deterministicPOCO():
    config = trainingConfig(model_name="DeterministicPoco")
    model = DeterministicPOCO(
    seq_length  = config.sequence_length,
    pred_length = config.pred_length,
    n_channels  = config.n_channels,
    ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(), StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

if __name__ == "__main__":
    run_MLP()