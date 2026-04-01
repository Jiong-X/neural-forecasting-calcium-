# GenAI Assistance Statement:
# Claude (Anthropic) was used to assist in structuring this file.
# All outputs were verified for technical accuracy by the group.

"""
test.py
-------
Loads saved model weights and produces final metrics and visual results.
Run: python test.py
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import get_dataset
from src.model import ProbabilisticForecaster
from src.evaluate import compute_metrics, plot_predictions

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH  = 64
PRED_LENGTH = 16
BATCH_SIZE  = 32
LOAD_PATH   = "models/saved/model.pt"

# ── Data ──────────────────────────────────────────────────────────────────────
_, test_dataset = get_dataset(seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH, test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Load model ────────────────────────────────────────────────────────────────
model = ProbabilisticForecaster(seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH).to(DEVICE)
model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
model.eval()

# ── Evaluate ──────────────────────────────────────────────────────────────────
metrics = compute_metrics(model, test_loader, DEVICE)
print("=== Test Results ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ── Visualise ─────────────────────────────────────────────────────────────────
plot_predictions(model, test_loader, DEVICE, save_dir="results/figures")
print("Figures saved to results/figures/")
