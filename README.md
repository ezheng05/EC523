# T-CRL: Temporal Causal Representation Learning for Behavioral Outcome Prediction

EC523 Deep Learning — Boston University  
Alex Chen & Ellen Zheng ({afchen, ellenz}@bu.edu)

## Overview

PyTorch implementation of T-CRL, a multimodal causal representation learning framework that predicts 9 behavioral outcomes from longitudinal smartphone sensor data: depression, anxiety, stress, loneliness, mindfulness, resilience, emotion regulation (reappraisal & suppression), and social support. The model fuses RAPIDS sensor time series with pre-semester survey baselines and uses a β-VAE with a learnable adjacency matrix to discover causal structure in the latent space.

Key design: a **Sigmoid Missingness Gate** treats missing sensor data as an informative signal (MMNAR — Missing Not At Random) rather than a nuisance, weighting temporal features by their reliability before fusing with clinical baselines.

**Dataset:** [GLOBEM](https://physionet.org/content/globem/1.1/) longitudinal smartphone sensor dataset (INS-W_1 through INS-W_4, ~400–700 participants across 4 semester cohorts).

**Targets:** depression (CESD), anxiety (STAI/STAIS), stress (PSS-10), loneliness (UCLA-10), mindfulness (MAAS), resilience (BRS), emotion regulation reappraisal (ERQ), emotion regulation suppression (ERQ), social support (2waySSS received emotional)

## Repository Structure

```
├── src/
│   ├── data/dataset.py        # multi-cohort dataset, train/val/test splits
│   ├── models/
│   │   ├── components.py      # TCN_Block, MissingnessFusionGate
│   │   ├── encoder.py         # TCRL_Encoder, Baseline_Standard_Encoder
│   │   └── vae.py             # TCRL_BetaVAE
│   ├── training/
│   │   ├── loss.py            # combined MSE + KL + L1 sparsity loss
│   │   └── trainer.py         # train_epoch, evaluate
│   └── utils/metrics.py       # RMSE, MAE, Pearson R, R², AUC-ROC, AUC-PR
├── config/config.py           # ModelConfig dataclass (all hyperparameters)
├── notebooks/
│   └── DL_Project_colab.ipynb # Colab notebook (mounts Drive, imports src/)
├── train.py                   # local training entry point
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/ezheng05/EC523.git
cd EC523
pip install -r requirements.txt
```

**Dependencies:** `torch`, `pandas>=1.5.3`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Training

### Local

```bash
python train.py --data_root /path/to/globem-dataset-multi-year-datasets-for-longitudinal-human-behavior-modeling-generalization-1.1
```

Optional flags:
```
--cohorts INS-W_1 INS-W_2 INS-W_3 INS-W_4   # which cohorts to train on
--epochs 100                                   # default 100
--output_dir checkpoints/                      # where to save best model
```

### Google Colab

Open `notebooks/DL_Project_colab.ipynb`. Set `REPO_ROOT` and `DATASET_ROOT` to your Drive paths, then run all cells.

## Architecture

```
RAPIDS sensor CSV  →  GLOBEM_MultiTaskDataset  →  TCN_Block
                                                       ↓
                                             MissingnessFusionGate (δ mask)
                                                       ↓
PRE survey (CESD/STAIS/PSS baseline)  →  EHR MLP  →  TCRL_Encoder
                                                       ↓
                                               TCRL_BetaVAE
                                          (μ, σ, adj matrix, z)
                                                       ↓
                                    3 predictions: depression, anxiety, stress
```

**Loss:** `L = MSE + β·KL + λ·||A||₁`  
- β=2.0 (VAE regularization), λ=0.001 (sparsity on causal adjacency matrix A)

**Training config:** Adam lr=0.002, 100 epochs, batch=16, latent_dim=3, seq_len=30 days

## Evaluation

Reports per-target metrics for depression, anxiety, and stress:
- Regression: RMSE, MAE, Pearson R, R²
- Classification (thresholded): AUC-ROC, AUC-PR  
  (CESD-10 ≥ 10, STAIS ≥ 40, PSS-10 ≥ 14)

Ablation: `Baseline_Standard_Encoder` (same architecture, missingness gate removed) runs automatically alongside T-CRL for comparison.

## References

- Liang et al., "CRL-MMNAR: Causal Representation Learning with Missing Not At Random Data"
- Li et al., "CHiLD: Causal Health Inference from Longitudinal Data" (NeurIPS 2025)
- Morioka & Hyvarinen, ICML 2024
- Wiemken et al., "GLOBEM Dataset" (PhysioNet 2023)
