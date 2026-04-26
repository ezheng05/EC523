import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

TARGET_NAMES = ["depression", "anxiety", "stress"]
# clinical cutoffs: cesd>=10, stais>=40, pss>=14
DEFAULT_THRESHOLDS = [10.0, 40.0, 14.0]


def compute_regression_metrics(y_pred, y_true):
    """returns dict of per-target rmse, mae, pearson r, r2"""
    results = {}
    for i, name in enumerate(TARGET_NAMES):
        p = y_pred[:, i].numpy()
        t = y_true[:, i].numpy()
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        # pearson r
        if p.std() > 0 and t.std() > 0:
            r = float(np.corrcoef(p, t)[0, 1])
        else:
            r = 0.0
        # r2
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        results[name] = {"rmse": rmse, "mae": mae, "pearson_r": r, "r2": r2}
    return results


def compute_auc_metrics(y_pred, y_true, thresholds=None):
    """returns dict of per-target auc-roc and auc-pr using thresholded binary labels"""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    results = {}
    for i, (name, thresh) in enumerate(zip(TARGET_NAMES, thresholds)):
        p = y_pred[:, i].numpy()
        t_bin = (y_true[:, i].numpy() >= thresh).astype(int)
        if t_bin.sum() == 0 or t_bin.sum() == len(t_bin):
            # can't compute auc with single class
            results[name] = {"auc_roc": float("nan"), "auc_pr": float("nan")}
            continue
        results[name] = {
            "auc_roc": float(roc_auc_score(t_bin, p)),
            "auc_pr": float(average_precision_score(t_bin, p)),
        }
    return results


def print_metrics(reg, auc):
    header = f"{'target':<12} {'rmse':>7} {'mae':>7} {'pearson_r':>10} {'r2':>7} {'auc_roc':>8} {'auc_pr':>7}"
    print(header)
    print("-" * len(header))
    for name in TARGET_NAMES:
        r = reg[name]
        a = auc[name]
        print(
            f"{name:<12} {r['rmse']:>7.3f} {r['mae']:>7.3f} {r['pearson_r']:>10.3f} "
            f"{r['r2']:>7.3f} {a['auc_roc']:>8.3f} {a['auc_pr']:>7.3f}"
        )
