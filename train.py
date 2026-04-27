import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config.config import ModelConfig
from src.data.dataset import make_splits
from src.models.encoder import TCRL_Encoder, Baseline_Standard_Encoder
from src.models.vae import TCRL_BetaVAE
from src.training.trainer import train_epoch, evaluate
from src.utils.metrics import compute_regression_metrics, compute_auc_metrics, print_metrics


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_cohort_dirs(data_root, cohorts):
    dirs = [os.path.join(data_root, c) for c in cohorts]
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"cohort dirs not found: {missing}")
    return dirs


def run(cfg, cohort_dirs, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    print("building datasets...")
    train_ds, val_ds, test_ds = make_splits(
        cohort_dirs, seq_len=cfg.seq_len,
        val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    feat_dim = train_ds.feat_dim

    def _make_model(use_gate):
        enc_cls = TCRL_Encoder if use_gate else Baseline_Standard_Encoder
        enc = enc_cls(feat_dim=feat_dim, seq_len=cfg.seq_len, ehr_dim=3, hidden_dim=cfg.hidden_dim)
        return TCRL_BetaVAE(enc, latent_dim=cfg.latent_dim, num_targets=cfg.num_targets).to(device)

    models = {"tcrl": _make_model(True), "baseline": _make_model(False)}
    best_val = {k: float("inf") for k in models}

    for name, model in models.items():
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        print(f"\n--- training {name} ---")

        for epoch in range(cfg.epochs):
            stats = train_epoch(model, train_loader, opt, device, cfg.beta, cfg.lambda_sparsity)
            if (epoch + 1) % 10 == 0:
                val_pred, val_true = evaluate(model, val_loader, device)
                val_mse = float(((val_pred - val_true) ** 2).mean())
                print(f"epoch {epoch+1:03d}/{cfg.epochs} | task: {stats['task']:.4f} | val mse: {val_mse:.4f}")
                if val_mse < best_val[name]:
                    best_val[name] = val_mse
                    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_best.pt"))

    print("\n\n=== test results ===")
    for name, model in models.items():
        ckpt = os.path.join(output_dir, f"{name}_best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        y_pred, y_true = evaluate(model, test_loader, device)
        reg = compute_regression_metrics(y_pred, y_true)
        auc = compute_auc_metrics(y_pred, y_true)
        print(f"\n{name}:")
        print_metrics(reg, auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="path to globem dataset root")
    parser.add_argument("--cohorts", nargs="+", default=["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    cfg = ModelConfig()
    if args.epochs:
        cfg.epochs = args.epochs

    set_global_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    cohort_dirs = build_cohort_dirs(args.data_root, args.cohorts)
    run(cfg, cohort_dirs, args.output_dir, device)


if __name__ == "__main__":
    main()
