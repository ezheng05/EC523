import torch
import torch.nn as nn


class TCRL_BetaVAE(nn.Module):
    def __init__(self, encoder, latent_dim=3, num_targets=3):
        super().__init__()
        self.encoder = encoder
        h_dim = encoder.to_latent.out_features

        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_logvar = nn.Linear(h_dim, latent_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, num_targets),
        )
        # learnable adjacency matrix for causal structure discovery
        self.adj = nn.Parameter(torch.randn(latent_dim, latent_dim))

    def encode(self, x, mask, ehr):
        h = self.encoder(x, mask, ehr)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-20, max=2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, mask, ehr):
        mu, logvar = self.encode(x, mask, ehr)
        z = self.reparameterize(mu, logvar)
        return self.pred_head(z), mu, logvar, z
