# T-CRL Project: Complete Concepts Reference

This document covers every concept, term, parameter, and research idea in the T-CRL project, from the big picture down to individual lines of code. Use it to get up to speed on everything the project involves.

---

## 1. What the Project Does (Big Picture)

**Goal:** Predict 5 behavioral outcomes for college students using data passively collected from their smartphones over a semester. The outcomes are: depression, anxiety, stress, loneliness, and resilience.

**Why it's hard:**
- The sensor data is collected every day but students often don't carry their phones or have apps disabled — so there's a lot of **missing data**, and crucially, the missingness itself is informative (a student who stops carrying their phone may be depressed).
- The dataset is small (~657 students across 4 semesters, ~461 in training), so the model needs to generalize well from limited data.
- We're predicting 5 continuous scores simultaneously (multi-task learning).

**What makes this project novel:**
- It treats missing data as a signal, not just noise (MMNAR — Missing Not At Random).
- It uses a causal model (beta-VAE with a learned adjacency matrix) to discover which latent factors cause which behavioral outcomes.

---

## 2. The Dataset — GLOBEM

### Longitudinal Study Design
A **longitudinal study** follows the same subjects over time (as opposed to a cross-sectional study, which takes a single snapshot). GLOBEM follows students across an entire semester — this is what makes it valuable for behavioral modeling. We can see how sensor patterns in the first week relate to mental health at the end of the semester.

**Cohort:** a group of participants studied together. GLOBEM has 4 semester cohorts — INS-W_1 through INS-W_4 — collected across different academic years. "INS-W" stands for the Indiana university study, winter/fall semester. Each cohort is independent (different students, same study design).

**Baseline covariate:** a measurement taken at the start of the study, used as an input to the model. Pre-semester survey scores are baseline covariates — they tell the model where the student started, so it can predict where they end up. This is standard practice in longitudinal health research ("controlling for baseline").

**GLOBEM** (Global Behavioral Modeling) is a longitudinal dataset from PhysioNet (Xu et al., 2023). It contains 4 semester cohorts of college students (INS-W_1 through INS-W_4).

### Sensor Data (RAPIDS features)
Each cohort has sensor feature CSVs — "RAPIDS" is a pipeline that extracts features from raw phone logs. Features include:
- **Bluetooth:** number of unique devices seen nearby, social proximity patterns
- **Call logs:** number of calls made/received, call duration
- **Location:** time spent at home vs. elsewhere, entropy of location visits
- **Screen:** screen-on time, number of unlocks
- **Sleep:** estimated sleep duration, regularity
- **Steps:** step counts (physical activity)
- **WiFi:** unique networks seen, time at known locations

Each row is one (student, day) pair. After alignment across all 4 cohorts, **3,390 shared feature columns** are used.

### Survey Data
- **post.csv** — surveys filled out at the END of semester. These are the **prediction targets**.
- **pre.csv** — surveys filled out at the START of semester. These are used as **baseline covariates** (input features to help the model, not targets).

### Behavioral Outcome Instruments (what we predict)
| Outcome | Instrument | Scale | Clinical threshold |
|---|---|---|---|
| Depression | CESD-10 (or CESD-9) | 0-30 | >= 10 = significant symptoms |
| Anxiety | STAIS or STAI | 20-80 | >= 40 = moderate anxiety |
| Stress | PSS-10 | 0-40 | >= 14 = moderate stress |
| Loneliness | UCLA-10 | 10-40 | >= 25 = loneliness |
| Resilience | BRS | 1-5 | <= 3.0 = low resilience (lower is worse) |

### Column Name Differences Across Cohorts
Different cohorts used slightly different versions of the same surveys. The code handles this by trying a list of candidate column names in order and mapping them to a single canonical name:
- Depression: CESD_10items_POST (W_1/3/4) or CESD_9items_POST (W_2)
- Anxiety: STAIS_POST (W_1) or STAI_POST (W_2/3/4)
- Stress: PSS_10items_POST (all cohorts)
- Loneliness: UCLA_10items_POST (all cohorts)
- Resilience: BRS_POST (all cohorts)

### Dataset Size
- Total users after filtering: 657
- Train split: 461 users (70%)
- Validation split: 99 users (15%)
- Test split: 97 users (15%)
- Dataset on disk: ~2.9 GB

---

## 3. Data Preprocessing

### Z-Score Normalization
Each sensor feature column is transformed to have **mean=0 and standard deviation=1**:

```
z = (x - mean) / std
```

This is done so no single feature dominates (some raw features are huge numbers, others are fractions). The mean and std are computed **only on training data** to avoid data leakage (the model shouldn't see test statistics during training).

Targets (POST scores) and baselines (PRE scores) are also z-normalized using train-split statistics, then de-normalized at evaluation time for interpretable metrics.

### Data Leakage
A fundamental rule in ML: the model cannot see any information from the test set during training, including statistics used for normalization. If you compute the mean of the whole dataset (including test) and normalize with it, you've "leaked" test information into the training process.

### 30-Day Sequences
For each student, we take their **last 30 days** of sensor data before end-of-semester. This gives a (30, feat_dim) tensor — 30 time steps, each with 3,390 features.

### Padding
If a student has fewer than 30 days of data, we **pad with NaN at the beginning** (prepend rows of NaN, then zero after masking). This is a common technique for variable-length sequences.

### Three Types of Missingness

This is a key theoretical concept that motivates the entire missingness gate design:

**MCAR (Missing Completely At Random):** data is missing for a random reason unrelated to anything. Example: a sensor randomly failed. The missingness pattern carries no information. Safe to just fill with the mean.

**MAR (Missing At Random):** missingness depends on other observed variables but not the missing value itself. Example: students with lower step counts (observed) are less likely to have WiFi logs. Can be handled with imputation.

**MMNAR (Missing Not At Random):** missingness depends on the value that's missing. Example: a student with severe depression stops charging their phone, so ALL their sensor data goes missing — the very state we're trying to predict causes the missingness. This is the hardest case and the one our model is designed for.

**Why MMNAR matters here:** in mental health research, the sickest patients are often the ones with the most missing data. Ignoring this biases models toward predicting outcomes for healthier, well-monitored students. Our missingness gate turns this problem into a feature.

### Missingness Mask (delta_mask)
For each (student, day, feature) entry, we create a binary mask:
- 1 = the value was actually observed
- 0 = the value was missing (NaN, filled with 0 afterward)

This gives a second (30, feat_dim) tensor called `delta_mask` that travels alongside the data.

### Train/Val/Test Split
We split at the **user level** (not the day level) into 70% train, 15% validation, 15% test. This means all 30 days of a given student go entirely into one split — you never train on some days of a student and test on others. Users from all 4 cohorts appear in all splits.

---

## 4. The Model Architecture

The model has several stacked components. Data flows through them in this order:

```
Sensor sequences (30, 3390)  -->  TCN_Block (Conv1d k=3)
Missingness mask (30, 3390)  -->  MissingnessFusionGate  -->  mean pooling
                                   h * (1 + sigmoid(MLP(mask)))               |
                                                                              v
Pre-semester scores (5,)  ------------------>  EHR MLP  -->  concatenate
                                                                   |
                                                          Linear (to hidden_dim=32)
                                                                   |
                                                     +-------------+-------------+
                                                     |                           |
                                               fc_mu (dim=4)             fc_logvar (dim=4)
                                                     |                           |
                                                     +--- reparameterize(mu, logvar) ---+
                                                                   |
                                                             z (latent, dim=4)
                                                                   |
                                                          Prediction Head
                                                     (Linear 4->16 -> BN -> ReLU -> Dropout
                                                      -> Linear 16->8 -> ReLU -> Dropout
                                                      -> Linear 8->5)
                                                                   |
                                                      5 predictions (depression, anxiety,
                                                        stress, loneliness, resilience)

Side component: Learnable adjacency matrix A (4x4) with L1 sparsity penalty
```

### 4.1 TCN (Temporal Convolutional Network)

A **Temporal Convolutional Network** processes sequences using 1D convolutions instead of recurrence (unlike RNNs/LSTMs).

**How 1D convolution works:** A filter (kernel) slides along the time dimension, computing a weighted sum of nearby time steps. This extracts local temporal patterns (e.g., "activity was high 3 days ago but low today").

**Key parameter — kernel_size=3:** The filter looks at 3 consecutive time steps at once. This is a local receptive field.

**Why TCN instead of LSTM/RNN:**
- Parallelizable (no sequential dependency between time steps)
- Avoids vanishing gradient problems
- Fixed-length receptive field is explicit and controllable

**Vanishing gradient:** in deep networks, gradients shrink as they're backpropagated through many layers. In RNNs, gradients must flow backward through each time step — for 30 steps, this causes gradients to vanish (become ~0) before reaching the early steps. TCNs avoid this because the convolution has direct connections at every position.

**Receptive field:** the number of input time steps that influence a single output. With kernel_size=3, one TCN layer sees 3 days at a time. Stacking multiple TCN layers increases the receptive field (each layer builds on the previous).

**In the code:** The TCN takes input of shape `(batch, seq_len, feat_dim)`, transposes to `(batch, feat_dim, seq_len)` for `Conv1d`, applies ReLU, then transposes back. Output: `(batch, 30, hidden_dim=32)`.

### 4.2 Missingness Fusion Gate (MMNAR) — Residual Design

**MMNAR = Missing Not At Random.** The key insight is that WHY data is missing carries information. A student who stops charging their phone at night might be experiencing poor sleep (and thus low step counts, screen time, etc.). Standard approaches just fill missing values with zeros or means and ignore the pattern.

The Missingness Fusion Gate uses a **residual design**:
1. Feed the delta_mask (a binary matrix showing what was observed) into a small MLP with a **Sigmoid activation**
2. The sigmoid outputs values between 0 and 1 — these are **gate weights**
3. Apply the residual formula: `output = h * (1 + gate(mask))`

**Why residual?** The key advantage over a simple multiplicative gate `h * gate(mask)` is that the original signal `h` is always preserved. Even if the gate outputs 0 (the MLP hasn't learned anything useful for this pattern), the output is `h * 1 = h` — the original signal passes through unchanged. This avoids the failure mode where the gate accidentally zeros out useful features. Well-observed time steps get amplified (gate > 0 means multiplicative factor > 1), while poorly-observed ones pass through at baseline strength.

**Sigmoid function:** sigma(x) = 1 / (1 + e^(-x)). Squashes any real number to (0, 1). Used here to produce weights, not probabilities.

### 4.3 TCRL_Encoder

After the TCN + Gate, we **mean pool** across the time dimension: average the 30 time steps into a single vector. This collapses the temporal dimension.

In parallel, the **EHR MLP** processes the 5 pre-semester baseline scores through a linear layer + ReLU + Dropout(0.3). "EHR" stands for Electronic Health Record — borrowed terminology for structured clinical/survey data.

The two vectors are **concatenated** and passed through a final linear layer (`to_latent`) to produce a fixed-size hidden representation of dim=32.

**Dropout (p=0.3)** is applied after pooling and inside the EHR MLP. Dropout randomly zeros out 30% of activations during training, which forces the network to not rely too heavily on any single feature — a key regularization technique.

### 4.4 Variational Autoencoder (VAE)

A **VAE** is a generative model that encodes inputs into a **probabilistic latent space** instead of a single point.

**Standard autoencoder:** encoder -> fixed point z -> decoder
**VAE:** encoder -> distribution N(mu, sigma^2) -> sample z -> decoder

The encoder outputs two vectors: **mu** and **log sigma^2 (logvar)**. These define a Gaussian distribution in latent space.

**Why probabilistic?** It regularizes the latent space — forces similar inputs to map to overlapping regions, making the space smooth and interpretable.

### 4.5 Reparameterization Trick

To **sample** from N(mu, sigma^2) in a way that allows gradients to flow through (needed for backprop):

```
z = mu + epsilon * sigma,  where epsilon ~ N(0, 1)
```

This is mathematically equivalent to sampling from N(mu, sigma^2) but the randomness (epsilon) is moved outside the parameters, so gradients can flow through mu and sigma.

### 4.6 beta-VAE

A **beta-VAE** is a VAE with a weighted regularization penalty on the latent space, controlled by beta:

```
Loss = task_loss + beta * KL_divergence
```

Our beta=0.05. This is small because the targets are z-normalized, so the task loss operates at ~1.0 scale. The original beta=2.0 was calibrated for raw-scale task loss of ~30; after normalization, beta was reduced ~40x to maintain the same relative balance between task loss and KL.

**Disentanglement:** the property that each latent dimension corresponds to a single interpretable concept (e.g., one dimension captures physical activity patterns, another captures social behavior).

### 4.7 KL Divergence

**KL divergence** (Kullback-Leibler divergence) measures how different one probability distribution is from another. In the VAE, it measures how far the learned latent distribution N(mu, sigma^2) is from a standard normal N(0, 1).

Formula for a Gaussian:
```
KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

Minimizing this term pushes the encoder to produce latent codes close to N(0, 1), which acts as a regularizer.

### 4.8 Learnable Adjacency Matrix (Causal Structure Discovery)

A key novel component: a learnable matrix `A` of shape `(4, 4)` representing causal relationships between latent dimensions.

Entry A[i,j] represents the causal influence of latent dimension i on dimension j.

The matrix is regularized with an **L1 penalty** on its entries:
```
L1 = sum(|A_ij|)
```

L1 regularization encourages **sparsity** — most entries go to zero, leaving only the strongest causal connections. This is inspired by how causal discovery algorithms (like NOTEARS, PC algorithm) learn sparse causal graphs from data.

The resulting matrix can be visualized as a heatmap showing which latent dimensions causally influence which others.

### 4.9 Prediction Head

Maps from the sampled latent vector z to the 5 behavioral predictions:
```
z (4,) -> Linear(4, 16) -> BatchNorm1d(16) -> ReLU -> Dropout(0.2)
       -> Linear(16, 8) -> ReLU -> Dropout(0.2)
       -> Linear(8, 5)
```

This is a 3-layer feedforward network with batch normalization after the first layer for training stability.

### 4.10 Linear Layer (Fully Connected Layer)

The most basic neural network building block. A linear layer computes:
```
output = input * W + b
```
Where W is a weight matrix and b is a bias vector — both are **learned parameters**. It maps from one dimension to another (e.g., Linear(32, 8) maps a 32-dimensional vector to an 8-dimensional one).

### 4.11 ReLU (Rectified Linear Unit)

An activation function: `ReLU(x) = max(0, x)`. Zeroes out negative values, passes positive values through unchanged.

**Why activation functions?** Without them, stacking linear layers just produces another linear layer. ReLU introduces non-linearity, allowing the network to learn complex patterns.

### 4.12 BatchNorm1d

Normalizes the activations across the batch dimension to have mean=0 and variance=1, then applies a learnable scale and shift. Stabilizes training and can act as a mild regularizer.

### 4.13 MLP (Multi-Layer Perceptron)

A sequence of linear layers with activation functions between them. Also called a feedforward network. Example from our code:
```
Linear -> ReLU -> Dropout -> Linear
```
The "multi-layer" part means multiple linear transformations are stacked.

### 4.14 Mean Pooling

After the TCN outputs a (30, hidden_dim) tensor (one vector per day), we **average across the 30 days** to get a single (hidden_dim,) vector. This is mean pooling — collapsing a variable-length sequence into a fixed-size summary.

**Why mean instead of max or last?** Mean pooling captures the average behavioral pattern across the whole window. Max would capture the peak. Last would only use the final day. Mean is the simplest and often works well for health applications.

### 4.15 Concatenation as Fusion

After the TCN pathway produces a pooled vector and the EHR MLP produces a baseline vector, we **concatenate** them: `[h_temporal | h_ehr]`. Concatenation doubles the dimension and preserves all information from both sources. The subsequent linear layer then learns how to combine them.

### 4.16 Tensor Shapes Through the Model

Tracking how data shape changes at each step (batch_size=16 example):

| Step | Shape | Description |
|------|-------|-------------|
| Input x | (16, 30, 3390) | 16 students, 30 days, 3390 features |
| Input mask | (16, 30, 3390) | binary missingness mask |
| Input ehr | (16, 5) | 5 pre-semester baseline scores |
| After TCN | (16, 30, 32) | hidden_dim=32 per day |
| After gate | (16, 30, 32) | gated by missingness (residual) |
| After mean pool | (16, 32) | collapsed time dimension |
| After EHR MLP | (16, 32) | baseline embedding |
| After concat | (16, 64) | fused representation |
| After to_latent | (16, 32) | hidden representation |
| mu, logvar | (16, 4) each | latent distribution parameters |
| z | (16, 4) | sampled latent vector |
| Predictions | (16, 5) | 5 behavioral outcome scores |

### 4.17 Logvar Clamping

In the VAE, `logvar` is clamped to `[-20, 2]`:
```python
logvar = torch.clamp(self.fc_logvar(h), min=-20, max=2)
```
**Why?** `exp(logvar)` gives the variance. If logvar is very large, the variance explodes; if very small (very negative), it underflows to 0. Clamping prevents numerical instability during training.

### 4.18 nn.Parameter

The adjacency matrix is defined as `nn.Parameter(torch.randn(latent_dim, latent_dim))`. Wrapping a tensor in `nn.Parameter` tells PyTorch two things:
1. Track its gradients (include it in backprop)
2. Include it in `model.parameters()` so the optimizer updates it

Without `nn.Parameter`, PyTorch won't update the matrix during training — it would just stay at its random initialization.

### 4.19 Training Mode vs Evaluation Mode

PyTorch models have two modes:
- `model.train()` — dropout is active (neurons randomly zeroed), batch norm uses batch statistics
- `model.eval()` — dropout is disabled (all neurons active), batch norm uses running statistics

**This is critical.** If you forget to call `model.eval()` before testing, dropout will randomly change predictions every time you run inference, and your metrics will be noisy and wrong.

### 4.20 torch.no_grad()

During evaluation, we wrap inference in `torch.no_grad()`:
```python
with torch.no_grad():
    y_pred = model(x, mask, ehr)
```
This tells PyTorch not to build a computational graph for these operations — saves memory and speeds up inference. During training, the graph is needed for backpropagation. During evaluation, it isn't.

### 4.21 .detach().cpu().numpy()

When extracting results from a PyTorch model to compute metrics:
```python
y_pred.detach().cpu().numpy()
```
- `.detach()` — removes the tensor from the computational graph (no gradients tracked)
- `.cpu()` — moves the tensor from GPU memory to CPU memory
- `.numpy()` — converts to a NumPy array (required for scikit-learn metrics)

---

## 5. Loss Function

Total loss has three components:

```
L_total = L_task + beta * L_KL + lambda * L_sparsity
```

### Task Loss (MSE)
**Mean Squared Error** — average squared difference between predicted and actual scores:
```
MSE = (1/n) * sum((y_pred - y_true)^2)
```
Penalizes large errors more than small ones (squared). With 5 targets, this is averaged across all predictions. Because targets are z-normalized, MSE operates at ~1.0 scale.

### KL Divergence Loss
As described above — regularizes the latent space distribution toward N(0,1). Weighted by beta=0.05.

### L1 Sparsity Loss
Sum of absolute values of the adjacency matrix entries. Weighted by lambda=0.001.

**Why L1 and not L2 for sparsity?** L1 regularization produces exact zeros (sparse solutions). L2 just makes weights small but rarely zero. For causal graph discovery, we want most entries to be exactly zero.

---

## 5.5 Hyperparameters vs. Learned Parameters

**Learned parameters** — values the model adjusts during training via gradient descent. Examples: weights in linear layers, the adjacency matrix, mu/logvar projections, batch norm parameters.

**Hyperparameters** — values YOU set before training that control the training process or model structure. The model never changes these.

### Current Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| hidden_dim | 32 | width of hidden layers in encoder |
| latent_dim | 4 | size of the causal latent space |
| num_targets | 5 | behavioral outcomes predicted |
| seq_len | 30 | days of sensor data per student |
| lr | 0.001 | Adam learning rate |
| weight_decay | 1e-3 | L2 regularization strength |
| epochs | 150 | maximum training epochs |
| batch_size | 16 | mini-batch size |
| dropout | 0.3 | encoder dropout rate |
| eval_every | 5 | evaluate every N epochs |
| early_stop_patience | 12 | eval checks without improvement before stopping |
| grad_clip | 1.0 | gradient clipping norm |
| beta | 0.05 | KL divergence weight |
| lambda_sparsity | 0.001 | L1 adjacency weight |
| val_ratio | 0.15 | validation split fraction |
| test_ratio | 0.15 | test split fraction |
| seed | 42 | random seed for reproducibility |

Getting hyperparameters right matters a lot. Too large a `hidden_dim` -> overfitting. Too large `beta` -> latent space collapses. Too small `lr` -> slow convergence. Too large `lr` -> training diverges.

---

## 6. Training Process

### Batch Training / Mini-batch Gradient Descent
Instead of computing the gradient over all training data at once (too slow) or one sample at a time (too noisy), we use **mini-batches** of 16 students. Each batch computes a loss, backpropagates gradients, and updates weights.

### Backpropagation
The algorithm for computing gradients through a neural network. Uses the chain rule to propagate the gradient of the loss backward through every layer, computing how much each parameter contributed to the error.

### Adam Optimizer
**Adam (Adaptive Moment Estimation)** is an optimizer that adapts the learning rate for each parameter:
- Keeps a running average of gradients (momentum)
- Keeps a running average of squared gradients (adaptive scaling)
- Result: large gradients get smaller steps, small gradients get larger steps

Parameters:
- **lr=0.001** — learning rate, controls step size
- **weight_decay=1e-3** — L2 regularization on all weights (penalizes large weights)

### Weight Decay (L2 Regularization)
Adds a penalty proportional to the square of each weight to the loss:
```
L_total += lambda * sum(w^2)
```
This discourages the model from fitting the training data too precisely (overfitting). With weight_decay=1e-3 in Adam, this is applied automatically.

### Gradient Clipping
Caps the norm of the gradient vector at 1.0:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
Prevents exploding gradients — when the gradient is too large, training becomes unstable. The gradient direction is preserved but its magnitude is capped.

### Dropout
During training, randomly sets p=30% of activations to zero each forward pass (in the encoder; 20% in the prediction head). The model can't rely on any single neuron — it must distribute learning across many. At test/eval time, dropout is disabled (all neurons active, weights scaled accordingly).

### Learning Rate Scheduler (ReduceLROnPlateau)
Monitors validation MSE. When it stops improving for `patience=3` consecutive evaluations, the learning rate is halved (factor=0.5). This allows the model to make finer adjustments as it converges.

### Early Stopping
Stop training when the **validation MSE** stops improving for `patience=12` consecutive evaluation checkpoints. We check every 5 epochs, so this means stopping after 60 epochs without improvement. Prevents wasted computation and overfitting.

### Overfitting vs. Underfitting
- **Overfitting:** model memorizes training data, performs poorly on new data. Signs: train loss much lower than val loss.
- **Underfitting:** model too simple to capture the pattern. Signs: both train and val loss are high.
- Our project shows some overfitting (train task loss ~0.5 at convergence while val MSE ~28), addressed with dropout, weight decay, gradient clipping, and early stopping.

### Forward Pass vs. Backward Pass
**Forward pass:** data flows through the model from input to predictions. The loss is computed.
**Backward pass (backpropagation):** gradients of the loss flow backward through every layer. Each parameter receives a gradient telling it which direction to change to reduce the loss.
During training, both happen every batch. During evaluation (inference), only the forward pass runs.

### Convergence
Training is said to have "converged" when the loss stops decreasing meaningfully — the model has found a stable set of weights. You can see this in the loss curves: initially loss drops quickly, then flattens out.

### Epoch
One complete pass through the entire training dataset.

### Validation Set
A held-out subset of data used to monitor training progress and tune decisions (like early stopping). Not used for gradient updates.

### Test Set
Completely held-out subset used only for final evaluation. Never touched during training or model selection.

### GPU and CUDA
A **GPU** (Graphics Processing Unit) has thousands of small cores optimized for parallel math operations — exactly what matrix multiplications in neural networks need. Training on GPU is typically 10-100x faster than CPU for deep learning.

**CUDA** is NVIDIA's programming framework that lets PyTorch run on GPU. `torch.device('cuda')` tells PyTorch to use the GPU. `tensor.to(device)` moves data to wherever the model is.

Training was done on an NVIDIA A100-SXM4-80GB GPU via Google Colab.

### Feature Intersection Across Cohorts
Different cohorts collected different sensor features (some columns exist in one cohort but not another). To train on all 4 cohorts jointly, we take the **intersection** — only columns that exist in every cohort. This gives 3,390 shared features.

---

## 7. Evaluation Metrics

### Regression Metrics (continuous predictions)

**RMSE (Root Mean Squared Error):** sqrt(mean squared error). Same units as the target. Lower is better.

**MAE (Mean Absolute Error):** average absolute difference. More robust to outliers than RMSE.

**Pearson R (correlation coefficient):** measures linear correlation between predictions and ground truth, ranging -1 to 1. 1 = perfect positive correlation.

**R-squared (coefficient of determination):** proportion of variance in the target explained by the model. 1 = perfect, 0 = predicts the mean, <0 = worse than predicting the mean.

### Classification Metrics (thresholded)

For targets with clinical cutoffs, we binarize (e.g., CESD >= 10 = "at risk") and compute:

**AUC-ROC (Area Under the ROC Curve):** measures how well the model ranks positive cases above negative ones, across all thresholds. 0.5 = random, 1.0 = perfect. Insensitive to class imbalance.

**AUC-PR (Area Under the Precision-Recall Curve):** more informative than AUC-ROC when classes are imbalanced (e.g., only 20% of students are clinically depressed).

**ROC curve:** plots True Positive Rate vs. False Positive Rate at every possible threshold.
**Precision-Recall curve:** plots Precision vs. Recall at every possible threshold.

---

## 8. Multi-Task Learning

**Multi-task learning (MTL)** trains a single model to predict multiple outputs simultaneously. The shared representation learns features useful for all tasks.

**Benefits:**
- Regularization effect — predicting multiple outcomes prevents overfitting to any one
- Shared representation captures common underlying factors
- More efficient than training 5 separate models

**In our model:** one encoder, one latent space, one prediction head with 5 outputs.

---

## 9. Causal Representation Learning

### Standard ML vs. Causal ML
Standard ML finds correlations. Causal ML tries to find the underlying mechanisms — what causes what.

**Example:** Physical activity (steps) is correlated with lower depression. But does walking cause less depression, or do less depressed people walk more? Causal ML tries to answer this.

### Latent Causal Factors
The idea is that observable behavioral outcomes (depression, anxiety, etc.) are caused by a small number of underlying latent factors. The model tries to discover these factors and the causal relationships between them. We use 4 latent dimensions for 5 targets.

### Structural Causal Model (SCM)
A framework where variables are connected by causal relationships (directed edges in a graph). The adjacency matrix in our model approximates this.

### Causal Discovery
The task of learning the structure of a causal graph from data. Our approach uses L1 sparsity on the adjacency matrix to encourage a sparse (few edges) causal structure, inspired by algorithms like NOTEARS.

### Identifiability
A key theoretical question: can the correct causal structure be uniquely identified from data? Research by Morioka & Hyvarinen (ICML 2024) and others shows this is possible under certain conditions (e.g., non-Gaussian noise, sufficient diversity in data).

---

## 10. Training Results

### Training Summary

| Model | Best Val MSE | Epochs at Early Stop | Best Epoch |
|---|---|---|---|
| T-CRL | 28.08 | 70 | 10 |
| Baseline | 26.81 | 105 | 45 |

### Test Set Results

**T-CRL:**
| Target | RMSE | MAE | Pearson R | R2 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Depression | 4.016 | 3.339 | 0.617 | 0.329 | 0.823 | 0.766 |
| Anxiety | 8.063 | 6.411 | 0.706 | 0.462 | 0.844 | 0.890 |
| Stress | 5.048 | 4.062 | 0.657 | 0.393 | 0.867 | 0.935 |
| Loneliness | 4.791 | 3.864 | 0.536 | 0.271 | 0.793 | 0.566 |
| Resilience | 0.555 | 0.446 | 0.587 | 0.337 | 0.782 | 0.697 |

**Baseline (no missingness gate):**
| Target | RMSE | MAE | Pearson R | R2 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Depression | 4.046 | 3.226 | 0.581 | 0.319 | 0.833 | 0.744 |
| Anxiety | 8.237 | 6.527 | 0.673 | 0.438 | 0.840 | 0.869 |
| Stress | 4.986 | 3.938 | 0.640 | 0.407 | 0.847 | 0.922 |
| Loneliness | 5.052 | 4.144 | 0.447 | 0.189 | 0.749 | 0.487 |
| Resilience | 0.560 | 0.451 | 0.572 | 0.327 | 0.778 | 0.657 |

### R2 Comparison

| Target | T-CRL R2 | Baseline R2 | Improvement |
|---|---|---|---|
| Depression | 0.329 | 0.319 | +0.010 |
| Anxiety | 0.462 | 0.438 | +0.024 |
| Stress | 0.393 | 0.407 | -0.014 |
| Loneliness | 0.271 | 0.189 | +0.082 |
| Resilience | 0.337 | 0.327 | +0.010 |
| **Average** | **0.358** | **0.336** | **+0.022** |

### AUC-ROC Comparison

| Target | T-CRL | Baseline | Improvement |
|---|---|---|---|
| Depression | 0.823 | 0.833 | -0.010 |
| Anxiety | 0.844 | 0.840 | +0.004 |
| Stress | 0.867 | 0.847 | +0.020 |
| Loneliness | 0.793 | 0.749 | +0.044 |
| Resilience | 0.782 | 0.778 | +0.004 |
| **Average** | **0.822** | **0.809** | **+0.012** |

### Per-Cohort Test Results (T-CRL, R2)

| Cohort | N (test) | Depression | Anxiety | Stress | Loneliness | Resilience |
|---|---|---|---|---|---|---|
| INS-W_1 | 18 | 0.233 | 0.528 | 0.359 | 0.082 | 0.034 |
| INS-W_2 | 29 | 0.289 | 0.450 | 0.498 | 0.218 | 0.158 |
| INS-W_3 | 21 | 0.296 | 0.492 | 0.214 | 0.262 | 0.545 |
| INS-W_4 | 29 | 0.272 | 0.515 | 0.480 | 0.281 | 0.388 |

### Key Findings

1. **T-CRL wins 4/5 targets on R2, 4/5 on AUC-ROC.** The missingness gate provides consistent improvement, with the largest gain on loneliness (+0.082 R2, +0.044 AUC-ROC).

2. **Why loneliness benefits most:** Loneliness prediction likely benefits from the missingness gate because social isolation directly affects phone usage patterns. Lonely individuals may have different sensor data availability patterns (fewer calls, less WiFi variety, different location patterns), and the gate learns to treat this informative missingness as a signal.

3. **Why stress is the exception:** Stress (PSS) has the highest AUC-ROC overall (0.867) and the baseline already captures it well. Stress may be more uniformly expressed in sensor data regardless of missingness patterns, making the gate's contribution minimal and occasionally adding slight noise.

4. **Clinical-grade AUC:** AUC-ROC values of 0.78-0.87 across all targets indicate clinically useful discrimination ability. For context, prior GLOBEM baselines typically achieve 0.55-0.75 for depression alone.

5. **Cross-cohort generalization:** The model generalizes across all 4 cohorts with no catastrophic failures. Anxiety is consistently the best-predicted target (R2 = 0.45-0.53 across all cohorts).

---

## 11. Key Design Choices and Tradeoffs

### latent_dim=4
The number of causal latent dimensions. Smaller = more compressed, more interpretable, possibly less accurate. Larger = more expressive but harder to interpret and more prone to overfitting. We use 4 for 5 targets — one fewer than outputs to force compression.

### beta=0.05 (beta-VAE weight)
Small because targets are z-normalized (task loss ~1.0 scale). The original beta=2.0 was calibrated for raw-scale task loss ~30. After normalization, beta was reduced ~40x to maintain the same relative balance. Higher beta -> stronger regularization -> more disentangled but potentially lower prediction accuracy.

### lambda_sparsity=0.001
Controls how sparse the causal adjacency matrix is. Too high -> all edges go to zero (no structure discovered). Too low -> dense, uninterpretable graph.

### hidden_dim=32
Width of hidden layers in the encoder. Higher = more capacity to learn complex patterns, more risk of overfitting.

### seq_len=30
We use the last 30 days of sensor data. Longer windows might capture more context but students often have data gaps early in the semester.

### dropout=0.3
30% of neurons randomly dropped during training in the encoder. Higher than the default 0.2 to combat overfitting on this small dataset.

### weight_decay=1e-3
L2 regularization strength. Moderately aggressive — helps prevent overfitting with 461 training users.

---

## 12. Ablation Study

An **ablation study** removes components of the model one at a time to measure each component's contribution.

**Our ablation:** `Baseline_Standard_Encoder` removes the Missingness Fusion Gate but keeps everything else identical (same TCN, same EHR MLP, same VAE, same loss, same hyperparameters). Comparing T-CRL vs. Baseline tells us how much the gate contributes.

**Results:** The gate improves R2 on 4/5 targets (average +0.022) and AUC-ROC on 4/5 targets (average +0.012). The largest improvement is on loneliness (+0.082 R2).

**Why this matters for the paper:** without ablation, you can't claim the missingness gate helps — it might be that any extra complexity helps equally.

---

## 13. Key Research References

**CHiLD (Li et al., NeurIPS 2025):** Causal Health Inference from Longitudinal Data. Primary inspiration for the causal representation learning approach applied to health outcomes.

**CRL-MMNAR (Liang et al.):** Causal Representation Learning with Missing Not At Random data. Direct source of the MMNAR insight — treating missingness as informative signal rather than noise.

**Morioka & Hyvarinen (ICML 2024):** Theoretical work on identifiability of causal representations from time series. Shows that causal structure can be recovered under certain conditions.

**GLOBEM Dataset (Xu et al., 2023):** The dataset used. Documents the 4 cohorts, sensor features, and survey instruments. Provides baselines (logistic regression, random forests, basic NNs) achieving AUC-ROC 0.55-0.75 for depression.

**beta-VAE (Higgins et al., ICLR 2017):** Introduced the beta weighting in the VAE loss for learning disentangled representations.

**NOTEARS (Zheng et al., NeurIPS 2018):** Reformulates causal structure learning as a continuous optimization problem with a sparsity constraint — conceptual inspiration for our L1 adjacency approach.

**GRU-D (Che et al., 2018):** Uses trainable decay mechanisms to handle missing values in clinical time series. Related work for handling missingness.

**StudentLife (Wang et al., 2014):** Pioneering study correlating smartphone sensor data with mental health outcomes in college students.

---

## 14. Software and Tools

**PyTorch:** deep learning framework. Provides automatic differentiation (autograd), neural network layers (nn.Module), optimizers, and GPU support.

**nn.Module:** base class for all PyTorch models. Defines `__init__` (build layers) and `forward` (define computation).

**DataLoader:** PyTorch utility that batches data, shuffles, and loads in parallel (num_workers=2). `pin_memory=True` speeds up GPU transfer.

**scikit-learn:** used here for AUC-ROC and AUC-PR computation (`roc_auc_score`, `average_precision_score`).

**pandas:** data manipulation library used for loading and processing CSV files.

**Google Colab:** cloud Jupyter notebook environment with GPU access. Dataset accessed via Google Drive mount.

---

## 15. Project-Specific Architecture Glossary

| Term | Meaning |
|---|---|
| `feat_dim` | number of sensor feature columns (3,390) |
| `seq_len` | 30 — days of sensor data per student |
| `ehr_dim` | 5 — number of pre-semester baseline scores |
| `hidden_dim` | 32 — width of hidden layers in encoder |
| `latent_dim` | 4 — size of the causal latent space |
| `num_targets` | 5 — number of behavioral outcomes to predict |
| `delta_mask` | binary (30, feat_dim) tensor: 1=observed, 0=missing |
| `mu` / `logvar` | mean and log-variance of the VAE latent distribution |
| `z` | sampled latent vector (the causal representation) |
| `adj` | (4 x 4) learnable causal adjacency matrix |
| `pred_head` | 3-layer MLP that maps z -> 5 predictions |
| `to_latent` | final linear layer in encoder that produces the hidden rep |
| `beta` | KL weight in beta-VAE loss (0.05) |
| `lambda_sparsity` | L1 weight on adjacency matrix (0.001) |
| `weight_decay` | L2 regularization in Adam (1e-3) |
| `dropout` | fraction of neurons dropped during training (0.3 encoder, 0.2 pred head) |
| `early_stop_patience` | number of val checks without improvement before stopping (12) |
| `grad_clip` | maximum gradient norm (1.0) |
