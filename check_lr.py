import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, ".")
from RNN import CalciumRNN, CalciumDataset, train, evaluate

data = np.load("data/processed/0.npz")
raw  = data["PC"].astype(np.float32)
if raw.shape[0] < raw.shape[1]:
    raw = raw.T
T, N = raw.shape
mu = raw.mean(axis=0, keepdims=True)
sd = raw.std(axis=0,  keepdims=True) + 1e-8
traces = (raw - mu) / sd

split        = int(T * 0.8)
train_ds     = CalciumDataset(traces[:split], seq_len=50, pred_steps=1)
val_ds       = CalciumDataset(traces[split:], seq_len=50, pred_steps=1)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

device = torch.device("cpu")
model  = CalciumRNN(n_neurons=N, hidden_size=256, num_layers=2,
                    dropout=0.2, default_pred_steps=1).to(device)
model.load_state_dict(torch.load("models/best_calcium_rnn.pt",
                                  map_location=device, weights_only=True))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

print(f"{'Epoch':>5}  {'LR':>10}  {'Val Loss':>10}")
for epoch in range(1, 51):
    train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    tag = " <- reduced" if new_lr < lr else ""
    print(f"{epoch:>5}  {lr:>10.2e}  {val_loss:>10.4f}{tag}")
