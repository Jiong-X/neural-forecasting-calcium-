# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
model.py
--------
Probabilistic forecasting model.
Outputs a mean and log-variance for each predicted timestep,
allowing uncertainty quantification via a Gaussian likelihood.
"""

import torch
import torch.nn as nn


class ProbabilisticForecaster(nn.Module):
    """
    Transformer-based probabilistic forecaster.
    Outputs:
        mean   — (B, pred_length) point prediction
        logvar — (B, pred_length) log-variance (aleatoric uncertainty)
    """

    def __init__(self, seq_length=64, pred_length=16, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.seq_length  = seq_length
        self.pred_length = pred_length

        # Input projection: scalar → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(seq_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads — mean and log-variance
        self.head_mean   = nn.Linear(d_model, pred_length)
        self.head_logvar = nn.Linear(d_model, pred_length)

    def forward(self, x):
        """
        x: (B, seq_length)
        returns: mean (B, pred_length), logvar (B, pred_length)
        """
        B, L = x.shape
        x = x.unsqueeze(-1)                                    # (B, L, 1)
        x = self.input_proj(x)                                 # (B, L, d_model)

        pos  = torch.arange(L, device=x.device)
        x    = x + self.pos_emb(pos)                           # add positional encoding

        x    = self.encoder(x)                                 # (B, L, d_model)
        cls  = x[:, -1, :]                                     # use last token as summary

        mean   = self.head_mean(cls)                           # (B, pred_length)
        logvar = self.head_logvar(cls)                         # (B, pred_length)
        return mean, logvar

    def sample(self, x, n_samples=50):
        """Draw n_samples predictions for uncertainty estimation."""
        mean, logvar = self.forward(x)
        std  = (0.5 * logvar).exp()
        eps  = torch.randn(n_samples, *mean.shape, device=x.device)
        return mean + eps * std                                # (n_samples, B, pred_length)
