import torch
import torch.nn as nn


class TCN_Block(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_sz=3):
        super().__init__()
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_sz, padding=kernel_sz // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq, feat) -> transpose for conv1d -> transpose back
        out = self.relu(self.conv(x.transpose(1, 2)))
        return out.transpose(1, 2)


class MissingnessFusionGate(nn.Module):
    """residual sigmoid gate that amplifies temporal features by missingness pattern

    output = h * (1 + gate(mask)) — preserves the original signal while letting
    well observed time steps amplify their contribution. avoids zeroing out
    features when the gate misfires (key issue with multiplicative only gating)
    """

    def __init__(self, mask_dim, hidden_dim):
        super().__init__()
        self.mask_mlp = nn.Sequential(nn.Linear(mask_dim, hidden_dim), nn.Sigmoid())

    def forward(self, h, delta_mask):
        return h * (1.0 + self.mask_mlp(delta_mask))
