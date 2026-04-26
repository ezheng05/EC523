from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # model dims
    hidden_dim: int = 16
    latent_dim: int = 3
    num_targets: int = 3
    seq_len: int = 30

    # training
    lr: float = 0.002
    epochs: int = 100
    batch_size: int = 16

    # loss weights
    beta: float = 2.0
    lambda_sparsity: float = 0.001

    # dataset
    cohorts: List[str] = field(default_factory=lambda: ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"])
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # clinical thresholds for AUC-ROC
    cesd_thresh: float = 10.0
    stais_thresh: float = 40.0
    pss_thresh: float = 14.0
