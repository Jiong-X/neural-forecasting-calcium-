# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
model.py
--------
Probabilistic POCO forecaster — Population-Conditioned neural activity predictor.

Wraps ProbabilisticPOCO from POCO_prob.py into the standard
(x) -> (mean, logvar) interface expected by train_utils.py.

Architecture:
  - Perceiver-IO encoder with rotary positional encodings (standalone_poco.py)
  - FiLM conditioning: population embedding modulates per-neuron predictions
  - Gaussian output head: predicts mu and log-sigma per neuron per step
  - Trained with Gaussian NLL loss
"""

import sys
import os
import torch
import torch.nn as nn

# add project root to path so POCO_prob and standalone_poco are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poco_src.POCO_prob import ProbabilisticPOCO, nll_loss
from poco_src.standalone_poco import NeuralPredictionConfig


class ProbabilisticForecaster(nn.Module):
    """
    Thin wrapper around ProbabilisticPOCO that exposes a simple
    (x) -> (mean, logvar) interface compatible with train_utils.py.

    Input:
        x : (B, context_len, N)  — batch of context windows

    Output:
        mean   : (B, pred_len, N)  — predicted mean activity
        logvar : (B, pred_len, N)  — predicted log-variance (aleatoric uncertainty)
    """

    name:str = "ProbabilisticPOCO"

    def __init__(self,
                 seq_length:  int = 64,
                 pred_length: int = 16,
                 n_channels:  int = 128):
        super().__init__()

        self.pred_length  = pred_length
        self.context_len  = seq_length - pred_length
        self.n_channels   = n_channels

        # Build POCO config — matching paper defaults
        config = NeuralPredictionConfig()
        config.seq_length          = seq_length
        config.pred_length         = pred_length
        config.compression_factor  = 16
        config.decoder_hidden_size = 128
        config.conditioning_dim    = 1024
        config.decoder_num_layers  = 1
        config.decoder_num_heads   = 16
        config.poyo_num_latents    = 8

        self.poco = ProbabilisticPOCO(config, [[n_channels]])

    def forward(self, x: torch.Tensor):
        """
        x : (B, context_len, N)
        returns:
            mean   : (B, pred_len, N)
            logvar : (B, pred_len, N)
        """
        # POCO expects list of (L, B, N) tensors — one per session
        x_list = [x.permute(1, 0, 2)]          # (context_len, B, N)
        dists   = self.poco(x_list)             # list of Normal distributions
        dist    = dists[0]                      # single session

        mean   = dist.mean.permute(1, 0, 2)     # (B, pred_len, N)
        # convert sigma -> logvar = 2 * log(sigma)
        logvar = 2 * dist.scale.log().permute(1, 0, 2)
        return mean, logvar

    def predict_distribution(self, x: torch.Tensor):
        """
        Returns the full Normal distribution for uncertainty analysis.
        x : (B, context_len, N)
        returns: torch.distributions.Normal with event shape (pred_len, B, N)
        """
        x_list = [x.permute(1, 0, 2)]
        return self.poco(x_list)[0]
