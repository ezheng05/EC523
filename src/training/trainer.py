import torch
from .loss import tcrl_loss


def train_epoch(model, loader, optimizer, device, beta=2.0, lam=0.001):
    model.train()
    total, task_sum = 0.0, 0.0

    for x, mask, ehr, y in loader:
        x, mask, ehr, y = x.to(device), mask.to(device), ehr.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred, mu, logvar, _ = model(x, mask, ehr)
        loss, t_loss, _, _ = tcrl_loss(y_pred, y, mu, logvar, model.adj, beta, lam)
        loss.backward()
        optimizer.step()
        total += loss.item()
        task_sum += t_loss.item()

    n = len(loader)
    return {"total": total / n, "task": task_sum / n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    for x, mask, ehr, y in loader:
        x, mask, ehr = x.to(device), mask.to(device), ehr.to(device)
        y_pred, _, _, _ = model(x, mask, ehr)
        preds.append(y_pred.cpu())
        targets.append(y.cpu())

    return torch.cat(preds), torch.cat(targets)
