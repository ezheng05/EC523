import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

TARGET_NAMES = [
    "depression",
    "anxiety",
    "stress",
    "loneliness",
    "mindfulness",
    "resilience",
    "erq_reappraisal",
    "erq_suppression",
    "social_support",
]

# clinical thresholds for binary AUC — None means skip AUC for that target
# sources: cesd>=10 (depression screen), stais>=40 (anxiety), pss>=14 (moderate stress),
#          ucla>=25 (loneliness), brs<=3.0 (low resilience, note: lower = worse)
THRESHOLDS = [10.0, 40.0, 14.0, 25.0, None, None, None, None, None]

# for resilience, lower score = worse outcome, so we flip the threshold direction
_LOWER_IS_WORSE = {"resilience"}


def compute_regression_metrics(y_pred, y_true):
    results = {}
    for i, name in enumerate(TARGET_NAMES):
        p = y_pred[:, i].numpy()
        t = y_true[:, i].numpy()
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae  = float(np.mean(np.abs(p - t)))
        r    = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else 0.0
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        results[name] = {"rmse": rmse, "mae": mae, "pearson_r": r, "r2": r2}
    return results


def compute_auc_metrics(y_pred, y_true, thresholds=None):
    if thresholds is None:
        thresholds = THRESHOLDS
    results = {}
    for i, (name, thresh) in enumerate(zip(TARGET_NAMES, thresholds)):
        if thresh is None:
            results[name] = {"auc_roc": float("nan"), "auc_pr": float("nan")}
            continue
        p = y_pred[:, i].numpy()
        t_arr = y_true[:, i].numpy()
        if name in _LOWER_IS_WORSE:
            t_bin = (t_arr <= thresh).astype(int)
        else:
            t_bin = (t_arr >= thresh).astype(int)
        if t_bin.sum() == 0 or t_bin.sum() == len(t_bin):
            results[name] = {"auc_roc": float("nan"), "auc_pr": float("nan")}
            continue
        results[name] = {
            "auc_roc": float(roc_auc_score(t_bin, p)),
            "auc_pr":  float(average_precision_score(t_bin, p)),
        }
    return results


def print_metrics(reg, auc):
    header = f"{'target':<18} {'rmse':>7} {'mae':>7} {'pearson_r':>10} {'r2':>7} {'auc_roc':>8} {'auc_pr':>7}"
    print(header)
    print("-" * len(header))
    for name in TARGET_NAMES:
        r = reg[name]
        a = auc[name]
        auc_roc = f"{a['auc_roc']:>8.3f}" if not np.isnan(a['auc_roc']) else "     n/a"
        auc_pr  = f"{a['auc_pr']:>7.3f}"  if not np.isnan(a['auc_pr'])  else "    n/a"
        print(
            f"{name:<18} {r['rmse']:>7.3f} {r['mae']:>7.3f} {r['pearson_r']:>10.3f} "
            f"{r['r2']:>7.3f} {auc_roc} {auc_pr}"
        )
