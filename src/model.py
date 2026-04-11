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
import torch
import torch.nn as nn

from src.metrics import Prediction
from src.poco_src.standalone_poco import POCO, NeuralPredictionConfig


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
        response = dists[0]                      # single session # (pred_len, B, D)
        return response


class DeterministicPOCO(nn.Module):
    name:str = "deterministicPOCO"

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

        self.poco = POCO(config, [[n_channels]])

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


        response = dists[0]                      # single session # (pred_len, B, D) 

        mean   = response.permute(1, 0, 2)     # (B, pred_len, N) 
        # convert sigma -> logvar = 2 * log(sigma)
        return Prediction(mean=mean)


# ---------------------------------------------------------------------------
# Probabilistic head
# ---------------------------------------------------------------------------


class ProbabilisticPOCO(POCO):
    """
    POCO with a Gaussian output head.

    Replaces the single out_proj (→ pred_length values per neuron) with:
        mu_proj      → pred_length means
        log_sig_proj → pred_length log-standard-deviations

    forward() returns torch.distributions.Normal(mu, sigma) where
    sigma = softplus(log_sig) + 1e-4  (always positive, no numerical issues).
    """

    LOG_SIG_MIN = -6.0   # clamp raw log_sig to avoid underflow
    LOG_SIG_MAX =  2.0   # clamp raw log_sig to avoid instability

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, input_size)

        # Replace the deterministic head with two heads of identical shape.
        cond_dim       = config.conditioning_dim
        pred_length    = config.pred_length

        # Remove the original head so its parameters are not optimised for
        # point-estimate loss by mistake.
        del self.out_proj

        self.mu_proj      = nn.Linear(cond_dim, pred_length)
        self.log_sig_proj = nn.Linear(cond_dim, pred_length)

        # FiLM conditioning weights are already zero-initialised inside POCO.__init__
        # (standalone_poco.py lines 838-841) — no manual init needed.

        # Initialise log_sig bias to predict ~0.5 std initially (log(0.5) ≈ -0.69)
        nn.init.constant_(self.log_sig_proj.bias, -0.69)
        nn.init.zeros_(self.log_sig_proj.weight)

        df = 7.0
        self.log_df = nn.Parameter(torch.tensor(float(df)).log())

    # ------------------------------------------------------------------
    # Override only the final projection inside forward()
    # ------------------------------------------------------------------

    def forward(self, x_list, unit_indices=None, unit_timestamps=None):
        """
        Args:
            x_list : list of (L, B, D) tensors — one per session.

        Returns:
            dist   : list of torch.distributions.Normal, one per session.
                     Each distribution has event shape (pred_length, B, D).
                     Use dist.mean for point predictions, dist.sample() for
                     stochastic rollouts, dist.log_prob(y) for NLL.
        """

        bsz = [x.size(1) for x in x_list]
        L   = x_list[0].size(0)

        # ---- replicate POCO.forward up to the embedding step ----
        x = torch.concatenate(
            [x.permute(1, 2, 0).reshape(-1, L) for x in x_list], dim=0
        )  # sum(B*D), L

        if L != self.Tin:
            x = x[:, -self.Tin:]

        # tokenise (no tokenizer by default → simple reshape)
        out = x.reshape(x.shape[0], self.Tin // self.T_step, self.T_step)

        d_list = self.input_size

        if unit_indices is None:
            sum_channels = 0
            unit_indices = []
            for b, d in zip(bsz, self.input_size):
                indices = (
                    torch.arange(d, device=x.device)
                    .unsqueeze(0).repeat(b, 1).reshape(-1)
                )
                unit_indices.append(indices + sum_channels)
                sum_channels += d
            unit_indices = torch.cat(unit_indices, dim=0)

        if unit_timestamps is None:
            unit_timestamps = (
                torch.zeros_like(unit_indices).unsqueeze(1)
                + torch.arange(0, self.Tin, self.T_step, device=x.device)
            )

        input_seqlen = torch.cat(
            [torch.full((b,), d, device=x.device) for b, d in zip(bsz, self.input_size)], dim=0
        )
        session_index = torch.cat(
            [torch.full((b,), i, device=x.device) for i, b in enumerate(bsz)], dim=0
        )
        dataset_index = torch.cat(
            [torch.full((b,), self.dataset_idx[i], device=x.device) for i, b in enumerate(bsz)], dim=0
        )

        embed = self.decoder(
            out,
            unit_indices=unit_indices,
            unit_timestamps=unit_timestamps,
            input_seqlen=input_seqlen,
            session_index=session_index,
            dataset_index=dataset_index,
        )  # sum(B*D), embedding_dim

        split_size = [b * d for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0)
        embed = [xx.reshape(b, d, self.embedding_dim)
                 for xx, b, d in zip(embed, bsz, d_list)]  # (B, D, E)
        
        df = self.log_df.exp().clamp(min=2.5)    # clamp to avoid extreme heavy tails / instability
        # ---- probabilistic head ----
        dists = []
        for e, d, x_in in zip(embed, self.input_size, x_list):
            alpha   = self.conditioning_alpha(e)       # B, D, cond_dim
            beta    = self.conditioning_beta(e)        # B, D, cond_dim
            inp     = x_in.permute(1, 2, 0)            # B, D, L
            weights = self.in_proj(inp) * alpha + beta # B, D, cond_dim

            mu_raw      = self.mu_proj(weights)            # B, D, pred_length
            log_sig = self.log_sig_proj(weights)       # B, D, pred_length
            log_sig = log_sig.clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX)
            sigma_raw   = torch.nn.functional.softplus(log_sig) + 1e-4
            logvar_raw = 2.0 * torch.log(sigma_raw)
            # Return in (B, pred_len, N) to match (mean, logvar) interface expected by trainer.py and metric.py
            mu    = mu_raw.permute(0, 2, 1) 
            logvar = logvar_raw.permute(0, 2, 1)
            dists.append(Prediction(df = df, mean=mu, logvar=logvar))

        return dists
