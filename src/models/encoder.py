import torch
import torch.nn as nn
from .components import TCN_Block, MissingnessFusionGate


class TCRL_Encoder(nn.Module):
    def __init__(self, feat_dim, seq_len=30, ehr_dim=3, hidden_dim=16, dropout=0.2):
        super().__init__()
        self.tcn = TCN_Block(feat_dim, hidden_dim)
        self.gate = MissingnessFusionGate(feat_dim, hidden_dim)
        self.ehr_mlp = nn.Sequential(nn.Linear(ehr_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.to_latent = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, ehr):
        h = self.gate(self.tcn(x), mask)
        h_pooled = self.drop(torch.mean(h, dim=1))
        fused = torch.cat([h_pooled, self.ehr_mlp(ehr)], dim=-1)
        return self.to_latent(fused)


class Baseline_Standard_Encoder(nn.Module):
    """ablation: same as TCRL_Encoder but without the missingness gate"""

    def __init__(self, feat_dim, seq_len=30, ehr_dim=3, hidden_dim=16, dropout=0.2):
        super().__init__()
        self.tcn = TCN_Block(feat_dim, hidden_dim)
        self.ehr_mlp = nn.Sequential(nn.Linear(ehr_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.to_latent = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, ehr):
        h_pooled = self.drop(torch.mean(self.tcn(x), dim=1))
        fused = torch.cat([h_pooled, self.ehr_mlp(ehr)], dim=-1)
        return self.to_latent(fused)
