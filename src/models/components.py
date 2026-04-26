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
    """sigmoid gate that weights temporal features by observed missingness pattern"""

    def __init__(self, mask_dim, hidden_dim):
        super().__init__()
        self.mask_mlp = nn.Sequential(nn.Linear(mask_dim, hidden_dim), nn.Sigmoid())

    def forward(self, h, delta_mask):
        return h * self.mask_mlp(delta_mask)
