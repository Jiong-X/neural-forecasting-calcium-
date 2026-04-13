# Neural Activity Forecasting from Calcium Imaging

Predict future neural population activity from whole-brain calcium imaging recordings.
Multiple model families are implemented and compared — from a closed-form AR baseline
to a probabilistic Perceiver-IO architecture (POCO) with Gaussian output and MC-Dropout
uncertainty quantification.

---

## Project Structure

```
neural-forecasting-calcium/
│
├── data/
│   ├── raw/                          # Raw HDF5 source files (auto-downloaded)
│   └── processed/                    # Preprocessed .npz files (one per subject)
│
├── models/
│   ├── saved/
│   │   ├── model.pt                  # Probabilistic POCO (best checkpoint)
│   │   └── best_MLP.pt              # MLP ablation (best checkpoint)
│   ├── best_calcium_ar.npz          # AR baseline
│   ├── best_calcium_poco.pt         # Deterministic POCO
│   ├── best_dlinear.pt              # DLinear baseline
│   ├── best_lstm.pt                 # LSTM baseline
│   ├── best_nlinear.pt              # NLinear baseline
│   ├── best_tsmixer.pt              # TSMixer baseline
│   └── best_vanilla_rnn.pt          # Vanilla RNN baseline
│
├── results/
│   ├── plots/                        # Generated analysis figures
│   ├── figures/                      # Test-set prediction figures
│   ├── logs/                         # Per-model training logs
│   └── *.npz                         # Loss curves and metric arrays
│
├── analysis/
│   ├── gaussian_diagnostics.py       # Per-PC Shapiro-Wilk + Q-Q plots
│   ├── mae_horizon.py                # MAE as a function of prediction horizon
│   ├── nll_horizon.py                # NLL as a function of prediction horizon
│   ├── normality_pooled.py           # Pooled normality test (skewness, kurtosis, D'Agostino)
│   ├── plot_ablation.py              # Ablation: full POCO (prob.) vs MLP-only head
│   ├── plot_comparison.py            # Deterministic vs probabilistic POCO comparison
│   ├── plot_poco_prob.py             # Prediction intervals and uncertainty bands
│   └── uncertainty.py                # MC-Dropout: aleatoric + epistemic decomposition
│
├── src/
│   ├── baseline_models/
│   │   ├── AR.py                     # Autoregressive baseline (closed-form OLS)
│   │   ├── DLinear.py                # Decomposition-Linear baseline
│   │   ├── LSTM.py                   # LSTM baseline
│   │   ├── MLP.py                    # MLP head (used in ablation)
│   │   ├── NLinear.py                # Normalisation-Linear baseline
│   │   ├── RNN.py                    # Vanilla RNN baseline
│   │   ├── TSMixer.py                # Time-Series Mixer baseline
│   │   └── TexFilter.py              # Frequency-domain filter baseline
│   ├── poco_src/
│   │   ├── configs/                  # POCO configuration files
│   │   ├── standalone_poco.py        # POCO backbone — Perceiver-IO + rotary encodings
│   │   ├── multisession.py           # Multi-session POCO variant
│   │   ├── prob_highdrop.py          # High-dropout probabilistic variant
│   │   └── prob_multisession.py      # Probabilistic multi-session variant
│   ├── dataset.py                    # Dataset loading, preprocessing & download pipeline
│   ├── evaluate.py                   # Evaluation utilities (metrics + plotting)
│   ├── metrics.py                    # Loss functions and metric tracking
│   ├── model.py                      # ProbabilisticForecaster & DeterministicPOCO
│   ├── trainer.py                    # Training loop with early stopping
│   └── util.py                       # Configuration, data utilities, baseline data loaders
│
├── train.py                          # End-to-end training (data retrieval + all models)
├── test.py                           # Evaluation: load models, produce metrics and figures
├── run_benchmark.py                  # Train deterministic baselines and Student-t variant
├── train_all.sh                      # Train all models sequentially with live logging
├── instruction.pdf                   # Reproduction instructions (packages + steps)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Environment Setup

One environment covers everything.

```bash
micromamba create -n poco python=3.10 -y
micromamba activate poco
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `matplotlib`, `einops`, `scipy`

> **Optional — only needed to preprocess raw HDF5 files (first run of `train.py`):**
> ```bash
> pip install h5py
> ```
> PCA is implemented via `numpy.linalg.svd` — scikit-learn is not required.
> Skip entirely if `data/processed/*.npz` already exists.

> **Optional — speeds up attention on GPU:**
> ```bash
> pip install xformers
> ```

---

## Data

### Recommended dataset — Zebrafish whole-brain calcium imaging (Ahrens lab)

> **Source:** https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/7272617

The dataset contains whole-brain GCaMP fluorescence recordings from larval zebrafish
(~80,000 neuron ROIs, ~2,880 timesteps per session at ~2.1 Hz).
The `CellResp` array in each `TimeSeries.h5` is already baseline-corrected —
no additional ΔF/F step is needed.

### Automatic retrieval (recommended)

**No manual download needed.** Simply run:

```bash
python train.py
```

`src/dataset.py` handles everything automatically in this order:

```
1. data/processed/0.npz exists?  → load directly (fastest)
2. data/raw/subject_0/TimeSeries.h5 exists?  → preprocess to .npz
3. Neither found?  → download subject_1.zip (~2 GB) from Janelia figshare,
                     extract TimeSeries.h5, preprocess to .npz, remove ZIP
```

The first run takes several minutes (download + preprocessing).
All subsequent runs load instantly from `data/processed/0.npz`.

> **Note:** If you have a corrupted or empty `TimeSeries.h5` from a previous
> failed download, delete it first:
> ```bash
> rm data/raw/subject_0/TimeSeries.h5
> python train.py
> ```

### Manual placement (multi-session models)

The multi-session models (`POCO_multisession.py`, `POCO_prob_multisession.py`)
require subjects 0–3. Download them manually from the figshare link above
and place as:

```
data/raw/
├── subject_0/TimeSeries.h5
├── subject_1/TimeSeries.h5
├── subject_2/TimeSeries.h5
└── subject_3/TimeSeries.h5
```

Then preprocess all subjects:
```bash
python preprocess.py --raw_dir data/raw --out_dir data/processed --subjects 0 1 2 3
```

### Using your own data

`dataloader.py` accepts four formats with no code changes required:

| Format | Extension        | Notes                                               |
|--------|------------------|-----------------------------------------------------|
| `npz`  | `.npz`           | Output of `preprocess.py`; looks for keys `PC`, `M`, `data` |
| `h5`   | `.h5` / `.hdf5`  | Any HDF5 file; default key `CellResp`               |
| `npy`  | `.npy`           | Plain NumPy array, shape `(T, N)`                   |
| `csv`  | `.csv` / `.tsv`  | Rows = timesteps, columns = channels                |

**Requirements for your own data:**
- Shape `(T, N)` — T timesteps, N channels (neurons or PCs)
- Values should be normalised. If you have raw fluorescence, compute ΔF/F first:
  `dff = (F - F0) / F0` where `F0` is a rolling baseline percentile.
- Minimum length: `context_len + pred_len + 1` timesteps (default: 48 + 16 + 1 = 65)

```python
from dataloader import get_loaders

# Preprocessed .npz
train_loader, val_loader, N = get_loaders("data/processed/0.npz")

# Raw HDF5
train_loader, val_loader, N = get_loaders("data/raw/subject_0/TimeSeries.h5",
                                          fmt="h5", h5_key="CellResp")

# Plain NumPy array passed directly
import numpy as np
arr = np.random.randn(3000, 128).astype("float32")
train_loader, val_loader, N = get_loaders(arr)

# CSV
train_loader, val_loader, N = get_loaders("my_traces.csv", fmt="csv", csv_header=True)
```

Quick sanity check:
```bash
python dataloader.py --path data/processed/0.npz --n_channels 128
python dataloader.py --path data/raw/subject_0/TimeSeries.h5 --fmt h5
```

**Known limitations:**
- Validated on zebrafish calcium imaging; untested on electrophysiology, fMRI, or mice.
- `preprocess.py` skips ΔF/F — assumes input is already baseline-corrected.
  Raw Suite2P / CaImAn output needs a ΔF/F step beforehand.
- PCA is fit on the training split. Very short recordings (< ~500 steps) give noisy PCs.
- Multi-session POCO assumes all sessions have the same number of input channels N.

---

## Step-by-step Pipeline

### Step 1 — Train

#### All models sequentially

```bash
micromamba activate poco
bash train_all.sh
```

Each model logs to `results/logs/<model>.log`. Monitor progress:
```bash
tail -f results/logs/poco_prob.log      # follow one model live
tail -n 1 results/logs/*.log            # last line from every model
```

#### Individual models

```bash
python POCO.py        # deterministic POCO
python POCO_prob.py   # probabilistic POCO (Gaussian output head)
```

Checkpoints are saved to `models/` and loss curves to `results/`.

### Step 2 — Analysis

```bash
# Uncertainty decomposition (MC Dropout — requires POCO_prob checkpoint)
python uncertainty.py

# Prediction accuracy vs forecast horizon
python eval_horizon.py

# Calibration — does stated confidence match empirical coverage?
python calibration.py

# Gaussian likelihood validation
python normality_pooled.py     # pooled skewness, kurtosis, D'Agostino test + Q-Q
python gaussian_diagnostics.py # per-PC Shapiro-Wilk + Q-Q plots
python bimodality_check.py     # bimodality coefficient + KDE per neuron
python estimate_df.py          # estimate Student-t ν from residuals (kurtosis + MLE)
```

### Step 3 — Visualise

```bash
python plot_poco_prob.py    # prediction intervals and uncertainty bands
python plot_comparison.py   # MAE / MSE comparison (NLinear, TSMixer, POCO det, POCO prob)
```

All figures saved to `results/plots/`.

---

## Data Partitioning

Matching the original POCO paper — 3:1:1 train / val / test split applied temporally:

```
Session (T = 2880 timesteps)
|<-------- Train (60%) -------->|<-- Val (20%) -->|<-- Test (20%) -->|
0                             1728              2304               2880
```

Sliding window (stride 1) over each split:
- Window size = context C + horizon P = 48 + 16 = 64 steps
- Each window yields `x: (C=48, N)` context and `y: (P=16, N)` target
- Train: ~1665 windows | Val: ~497 windows | Test: ~497 windows

Val is used for checkpoint selection only. Test is evaluated once using the best
checkpoint — never used during training or model selection.

For multi-session POCO, Subject 3 is held out entirely as the test animal.

---

## Hyperparameters

All models follow the POCO paper defaults where applicable:

| Parameter         | Value        | Notes                                      |
|-------------------|--------------|--------------------------------------------|
| Context C         | 48 steps     | ~22.9 s at 2.1 Hz                          |
| Horizon P         | 16 steps     | ~7.6 s at 2.1 Hz                           |
| Input channels N  | 128 PCs      | Top-128 principal components               |
| Batch size        | 64           |                                            |
| Optimiser         | AdamW        | lr = 3×10⁻⁴, weight decay = 10⁻⁴          |
| Gradient clipping | max norm 5   |                                            |
| Early stopping    | patience 10  | Epochs without val improvement             |
| LR schedule       | ReduceLROnPlateau | factor 0.5, patience 5 epochs        |

POCO-specific:

| Parameter              | Value  | Notes                            |
|------------------------|--------|----------------------------------|
| Token length T         | 16     | 16 timesteps per token → 3 tokens for C=48 |
| Embedding dim d        | 128    | Hidden size in Perceiver         |
| Conditioning dim M     | 1024   | FiLM MLP width                   |
| Attention heads        | 16     |                                  |
| Latents K              | 8      | Perceiver bottleneck             |
| Self-attention layers  | 1      |                                  |
| FiLM init              | zeros  | Model starts as plain MLP        |

---

## POCO Architecture

```
Input: (C=48 timesteps, N=128 channels)
  │
  ├─ Tokenisation — group 16 steps → 3 tokens
  │   Linear projection: 16 → d=128
  │   Add UnitEmbed(neuron_id) + SessionEmbed(session_id)
  │
  ├─ Cross-Attention — 8 learned latents attend to 3×N tokens  (ENCODE)
  │   Rotary positional encodings on queries and keys
  │
  ├─ Self-Attention × 1 — latents communicate                   (PROCESS)
  │
  ├─ Cross-Attention — N neuron queries attend to 8 latents     (DECODE)
  │   Each neuron reads out its summary from the bottleneck
  │
  ├─ FiLM Conditioning
  │   alpha = Linear(per-neuron embedding)  → scale γ  (N, M)
  │   beta  = Linear(per-neuron embedding)  → shift β  (N, M)
  │   context projected: F = W_in · x_raw              (N, M)
  │   conditioned:       F' = F ⊙ γ + β               (N, M)
  │
  └─ Output head
       Deterministic : W_out ∈ ℝ^(M×P)   → point prediction μ
       Probabilistic : W_μ  ∈ ℝ^(M×P)   → mean
                       W_σ  ∈ ℝ^(M×P)   → log σ → softplus + ε → σ > 0
                       Loss: NLL = −(1/NP) Σ log N(y | μ, σ²)
```

---

## Uncertainty Quantification

MC Dropout (Kendall & Gal, 2017; Gal & Ghahramani, 2016):
Keep dropout active at inference and run T=50 forward passes on the same input.

```
μ_t, σ_t  — mean and std from pass t  (t = 1 ... T)

Aleatoric  = sqrt( E_t[σ_t²] )     average predicted variance  (irreducible data noise)
Epistemic  = sqrt( Var_t[μ_t] )    variance of predicted means  (model uncertainty)
Total      = sqrt( Aleatoric² + Epistemic² )
```

Aleatoric dominates (~87–93%) — prediction error is mostly irreducible neural stochasticity,
not model ignorance. Higher dropout increases the epistemic signal.

---

## Changing the Forecast Horizon

Each script has two constants near the top:

| Script family              | Context var  | Horizon var  | Default C | Default P |
|----------------------------|--------------|--------------|-----------|-----------|
| RNN / LSTM / AR            | `SEQ_LEN`    | `PRED_STEPS` | 48        | 16        |
| NLinear / DLinear / TSMixer| `SEQ_LEN`    | `PRED_LEN`   | 48        | 16        |
| POCO (all variants)        | `CONTEXT`    | `PRED_LEN`   | 48        | 16        |

RNN, LSTM, and AR use autoregressive rollout (one step at a time).
Linear and POCO models predict all P steps in a single forward pass.

---

## Results

Models trained on Subject 0, context C=48, horizon P=16, top-128 PCs,
3:1:1 temporal train/val/test split.

| Model         | Val MAE | Val MSE | Notes                          |
|---------------|---------|---------|--------------------------------|
| POCO (det.)   | 0.5242  | 0.5023  | Point forecast                 |
| POCO (prob.)  | 0.5276  | —       | Gaussian NLL loss; MSE not comparable |

MAE in z-score units.

**Uncertainty decomposition** (MC Dropout, T=50 passes, POCO_prob):

| Component  | Value  | Interpretation                        |
|------------|--------|---------------------------------------|
| Aleatoric  | 0.3597 | Irreducible data noise (~92.5%)       |
| Epistemic  | 0.0268 | Model uncertainty (~7.5%)             |
| Total      | 0.3597 | Dominated by aleatoric uncertainty    |

---

## Gaussian Likelihood Validation

The Gaussian output head in POCO_prob is validated both theoretically and empirically.

**Theoretical justification**
- Maximum entropy principle: Gaussian makes the fewest assumptions given only mean and variance constraints
- Central Limit Theorem: PCA projections are linear combinations of ~80,000 neurons; aggregates tend toward Gaussianity
- Standard in the literature: POCO, LFADS, NDT all use Gaussian likelihoods on z-scored / PCA-projected activity

**Empirical findings** (pooled standardised residuals, n = 1,050,624)

| Statistic        | Value   | Interpretation                          |
|------------------|---------|-----------------------------------------|
| Skewness         | +0.135  | Near-symmetric — Gaussian centre holds  |
| Excess kurtosis  | +2.386  | Mild heavy tails — isolated burst events|
| R² (Q-Q plot)    | 0.9937  | 99.4% of quantile variance explained    |

The heavy tails originate from sudden neural burst events (stimulus responses, state
transitions) — biologically meaningful activity that should not be removed. The Gaussian
σ head implicitly learns to inflate uncertainty around volatile timepoints, achieving
well-calibrated predictive intervals despite the mild tail deviation.

**Student-t comparison**
A Student-t model (ν estimated via kurtosis formula ≈ 7, MLE ≈ 12) was trained and
compared via reliability diagrams. The Gaussian model produced better-calibrated
intervals in both cases: the Student-t penalises outliers less during training,
causing the learned σ to shrink and under-cover at high confidence levels. This
confirms the Gaussian likelihood as the pragmatically appropriate choice.

---

## References

- Ahrens, M.B. et al. (2013). Whole-brain functional imaging at cellular resolution using
  light-sheet microscopy. *Nature Methods*, 10(5), 413–420.
- Azabou, M. et al. (2024). POCO: A Unified, Scalable Framework for Neural Population
  Decoding. *NeurIPS 2024*. https://arxiv.org/abs/2310.16046
- Jaegle, A. et al. (2021). Perceiver IO: A General Architecture for Structured Inputs
  & Outputs. *ICML 2022*.
- Su, J. et al. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding.
  *Neurocomputing*.
- Kendall, A. & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning
  for Computer Vision? *NeurIPS 2017*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing
  Model Uncertainty in Deep Learning. *ICML 2016*.
- Zeng, A. et al. (2023). Are Transformers Effective for Time Series Forecasting?
  *AAAI 2023*. (NLinear / DLinear)
- Chen, S. et al. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting.
  *arXiv 2303.06053*.
