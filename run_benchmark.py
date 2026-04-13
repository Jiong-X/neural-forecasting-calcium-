# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

import torch

from src.baseline_models.LSTM import CalciumLSTM
from src.baseline_models.DLinear import DLinear
from src.baseline_models.NLinear import NLinear
from src.baseline_models.MLP import MLPHead
from src.baseline_models.RNN import CalciumVanillaRNN
from src.baseline_models.TSMixer import TSMixer
from src.baseline_models.TexFilter import TexFilter
from src.model import DeterministicPOCO, ProbabilisticForecaster

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

def run_StudentTProbPOCO():
    config = trainingConfig(model_name="StudentTProbabilisticPOCO")
    model = ProbabilisticForecaster(
    seq_length  = config.sequence_length,
    pred_length = config.pred_length,
    n_channels  = config.n_channels,
    ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), MSELoss(RMSE=True), GaussianNllLoss()], primary=StudentTNllLoss())
    train(model, config, optimizer, criterion)

def run_DLinear():
    config = trainingConfig(model_name="DLinear")
    model = DLinear(context_len=(config.sequence_length - config.pred_length), pred_len=config.pred_length, n_channels=config.n_channels).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(),  StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

def run_NLinear():
    config = trainingConfig(model_name="NLinear")
    model = NLinear(context_len=(config.sequence_length - config.pred_length), pred_len=config.pred_length, n_channels=config.n_channels).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(),  StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

def run_TexFilter():
    config = trainingConfig(model_name="TexFilter")
    EMBED_SIZE = 128    # filter_embed_size in POCO repo
    HIDDEN     = 512    # hidden_size in POCO repo
    DROPOUT    = 0.3    # dropout in POCO repo

    model = TexFilter(
        n_channels=config.n_channels,
        context_len=(config.sequence_length - config.pred_length),
        pred_len=config.pred_length,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN,
        dropout=DROPOUT
    ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(),  StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

def run_TSMixer():
    config = trainingConfig(model_name="TSMixer")
    FF_DIM     = 64   # hidden dim of MLP mixing blocks (paper default)
    N_LAYERS   = 2    # number of mixer blocks (paper default)
    DROPOUT    = 0.1  # paper default

    model = TSMixer(
        context_len=(config.sequence_length - config.pred_length),
        pred_len=config.pred_length,
        n_channels=config.n_channels,
        ff_dim=FF_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(),  StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

def run_LSTM():
    config = trainingConfig(model_name="LSTM")
    HIDDEN     = 256   # hidden dim of LSTM mixing blocks (paper default)
    N_LAYERS   = 2    # number of mixer blocks (paper default)
    DROPOUT    = 0.2  # paper default

    model = CalciumLSTM(
        n_neurons=config.n_channels,
        hidden_size=HIDDEN,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        default_pred_steps=config.pred_length
        ).to(config.device)
      
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(),  StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

def run_RNN():
    config = trainingConfig(model_name="RNN")
    HIDDEN     = 256   # hidden dim of RNN mixing blocks (paper default)
    N_LAYERS   = 2    # number of mixer blocks (paper default)
    DROPOUT    = 0.2  # paper default

    model = CalciumVanillaRNN(
        n_neurons=config.n_channels,
        hidden_size=HIDDEN,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        default_pred_steps=config.pred_length
        ).to(config.device)
    LR          = 3e-4     # AdamW learning rate (paper default)
    WEIGHT_DECAY= 1e-4     # AdamW weight decay  (paper default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = MetricSuite([MAELoss(), GaussianNllLoss(),  StudentTNllLoss()], primary=MSELoss(RMSE=True))
    train(model, config, optimizer, criterion)

if __name__ == "__main__":
    run_MLP()
    run_deterministicPOCO()
    run_StudentTProbPOCO()
    run_DLinear()
    run_NLinear()
    run_TexFilter()
    run_TSMixer()
    run_LSTM()
    run_RNN()
