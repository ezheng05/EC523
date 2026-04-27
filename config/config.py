from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # model dims
    hidden_dim: int = 32
    latent_dim: int = 8
    num_targets: int = 9
    seq_len: int = 30

    # training
    seed: int = 42
    lr: float = 0.002
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 16
    dropout: float = 0.2
    early_stop_patience: int = 15  # epochs (checked every 10)

    # loss weights
    beta: float = 2.0
    lambda_sparsity: float = 0.001

    # dataset
    cohorts: List[str] = field(default_factory=lambda: ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"])
    val_ratio: float = 0.15
    test_ratio: float = 0.15
