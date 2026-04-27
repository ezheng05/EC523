# T-CRL

Temporal Causal Representation Learning for behavioral outcome prediction from smartphone sensor data.

EC523 Deep Learning 
Boston University
Alex Chen & Ellen Zheng

## What it does

Predicts 5 end-of-semester mental health outcomes (depression, anxiety, stress, loneliness, resilience) from passively collected phone sensor data + a baseline survey. Trained on the [GLOBEM](https://physionet.org/content/globem/1.1/) dataset (4 cohorts, 657 students).

The main idea: missing sensor data is treated as a signal, not noise. If a student stops using their phone, that pattern itself is informative. A residual missingness gate learns to use observation patterns rather than discarding them.

## Results

T-CRL vs baseline (same model, gate removed):

| target | T-CRL R² | baseline R² | T-CRL AUC-ROC | baseline AUC-ROC |
|---|---|---|---|---|
| depression | 0.329 | 0.319 | 0.823 | 0.833 |
| anxiety | 0.462 | 0.438 | 0.844 | 0.840 |
| stress | 0.393 | 0.407 | 0.867 | 0.847 |
| loneliness | 0.271 | 0.189 | 0.793 | 0.749 |
| resilience | 0.337 | 0.327 | 0.782 | 0.778 |

Gate helps most on loneliness (+0.08 R²) where missingness patterns carry social isolation signal.

## Repo structure

```
src/
  data/dataset.py          # GLOBEM_MultiTaskDataset, make_splits
  models/
    components.py           # TCN_Block, MissingnessFusionGate
    encoder.py              # TCRL_Encoder, Baseline_Standard_Encoder
    vae.py                  # TCRL_BetaVAE (beta-VAE + learnable adjacency matrix)
  training/
    loss.py                 # MSE + beta*KL + lambda*L1(adj)
    trainer.py              # train_epoch, evaluate
  utils/metrics.py          # RMSE, MAE, R², AUC-ROC, AUC-PR
config/config.py            # all hyperparameters
notebooks/
  DL_Project_colab.ipynb    # primary notebook (Colab, mounts Drive)
train.py                    # local CLI. training entry point
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

needs: torch, pandas, numpy, matplotlib, seaborn, scikit-learn

## Running

**colab (recommended):** open `notebooks/DL_Project_colab.ipynb`, make sure the GLOBEM dataset is in your Google Drive, run all cells.

**local:**
```bash
python train.py --data_root /path/to/globem-dataset-...
```

## How it works

```
30-day sensor sequence (3390 features) --> TCN (Conv1d, k=3)
                                             |
missingness mask ---------> MissingnessFusionGate: h * (1 + sigmoid(MLP(mask)))
                                             |
                                        mean pool
                                             |
baseline PRE survey (5 scores) --> MLP --> concat --> encoder --> beta-VAE --> 5 predictions
                                                                    |
                                                          4x4 adjacency matrix
                                                          (causal structure, L1 sparse)
```

Loss: `MSE + 0.05*KL + 0.001*L1(A)`

Training: Adam lr=0.001, batch=16, dropout=0.3, early stopping, gradient clipping. 70/15/15 user-level split, z-score normalization from train stats only.

## References

- Xu et al., "GLOBEM Dataset" (PhysioNet 2023)
- Higgins et al., "beta-VAE" (ICLR 2017)
- Liang et al., "CRL-MMNAR"
- Li et al., "CHiLD" (NeurIPS 2025)
- Che et al., "GRU-D" (2018)
