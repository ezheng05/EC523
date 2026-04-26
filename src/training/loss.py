import torch
import torch.nn.functional as F


def tcrl_loss(y_pred, y_true, mu, logvar, adj, beta=2.0, lam=0.001):
    task = F.mse_loss(y_pred, y_true)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_true.size(0)
    l1 = torch.sum(torch.abs(adj))
    return task + beta * kl + lam * l1, task, kl, l1
