### README.md

# Temporal Causal Representation Learning (T-CRL) for Behavioral Health

## Project Overview
This repository contains the PyTorch implementation of a Multimodal Temporal Causal Representation Learning (T-CRL) framework. The model is designed to fuse high frequency, longitudinal wearable data (e.g., minute/hourly-level Fitbit streams) with static Electronic Health Record (EHR) events to predict the onset of discrete cardiometabolic conditions (e.g., Type 2 Diabetes, Essential Hypertension).

**Current Status:** Stage 1 Multimodal Encoder completed and validated on synthetic data. Awaiting final dataset clearance from the NIH *All of Us* Researcher Workbench for empirical training.

## Architecture Highlights
The framework handles the extreme sparsity and asynchronous nature of wearable-EHR datasets using a two-stage approach:

1. **Stage 1: Multimodal Temporal Encoder**
   * **Temporal Convolutional Network (TCN):** Extracts local, shift invariant temporal patterns from hourly wearable sequences (heart rate, steps).
   * **Sigmoid Missingness Gate:** A specialized gating mechanism that leverages missingness masks ($\delta$) as informative features, weighing the reliability of the temporal latent representation before pooling.
   * **EHR Fusion:** Concatenates the dynamically weighted temporal features with a static dense embedding of clinical baseline measurements.

2. **Stage 2: Causal Discovery ($\beta$-VAE)** *(In Progress)*
   * Will utilize an $L_1$ sparsity penalty to disentangle the fused latent space into independent causal drivers of health.

## Repository Structure
```text
├── models/
│   ├── DL_project.ipynb    # PyTorch implementation of TCN and Missingness Gate
│   └── ...                  # Future VAE architectures
├── notebooks/
│   └── data_extraction.md   # Documentation for All of Us BigQuery SQL extraction
├── README.md
```

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* NumPy

### Running the Synthetic Validation
Currently, the pipeline includes a synthetic data generator that simulates the exact dimensionalities and informative missingness properties of the target demographic. This is used to validate gradient flow and dimensional alignment.

```bash
# Clone the repository
git clone [https://github.com/ezheng05/tcrl-cardiometabolic.git](https://github.com/ezheng05/tcrl-cardiometabolic.git)
cd tcrl-cardiometabolic

# Run the Stage 1 Encoder forward pass
python models/stage1_encoder.py
```

*Expected Output:*
```text
Model initialized successfully.
Input Shape (Fitbit): torch.Size([32, 24, 2])
Output Latent Space (Z) Shape: torch.Size([32, 16])
```

## Acknowledgments
Developed for EC523 (Deep Learning) at Boston University. Data access provided by the NIH *All of Us* Research Program.
```
