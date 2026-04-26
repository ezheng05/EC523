import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLS = ["CESD_10items_POST", "STAIS_POST", "PSS_10items_POST"]
BASELINE_COLS = ["CESD_10items_PRE", "STAIS_PRE", "PSS_10items_PRE"]
IGNORE_COLS = {"pid", "date", "uid"}


def _load_cohort(cohort_dir):
    """load and clean rapids + pre + post for one cohort, return (rapids, survey) or (None, None)"""
    rapids = pd.read_csv(os.path.join(cohort_dir, "FeatureData", "rapids.csv"), low_memory=False)
    post = pd.read_csv(os.path.join(cohort_dir, "SurveyData", "post.csv"))
    pre  = pd.read_csv(os.path.join(cohort_dir, "SurveyData", "pre.csv"))

    missing_post = [c for c in TARGET_COLS   if c not in post.columns]
    missing_pre  = [c for c in BASELINE_COLS if c not in pre.columns]

    if missing_post or missing_pre:
        # print what mental-health-related columns do exist to help diagnose
        mh_keywords = ("cesd", "stai", "pss", "phq", "gad", "dep", "anx", "stress")
        avail_post = [c for c in post.columns if any(k in c.lower() for k in mh_keywords)]
        avail_pre  = [c for c in pre.columns  if any(k in c.lower() for k in mh_keywords)]
        print(f"  skipping {os.path.basename(cohort_dir)}: column mismatch")
        if missing_post:
            print(f"    post.csv missing: {missing_post}")
            print(f"    available mh cols in post: {avail_post}")
        if missing_pre:
            print(f"    pre.csv missing: {missing_pre}")
            print(f"    available mh cols in pre:  {avail_pre}")
        return None, None

    survey = pd.merge(post, pre, on="pid", how="inner")
    survey = survey.dropna(subset=TARGET_COLS + BASELINE_COLS)

    feat_cols = [c for c in rapids.columns if c not in IGNORE_COLS and "Unnamed" not in c]
    rapids[feat_cols] = rapids[feat_cols].apply(pd.to_numeric, errors="coerce")
    rapids = rapids.dropna(axis=1, how="all")

    return rapids, survey


def _get_shared_cols(rapids_list):
    """intersection of feature columns across all cohorts"""
    col_sets = [
        set(c for c in r.columns if c not in IGNORE_COLS and "Unnamed" not in c)
        for r in rapids_list
    ]
    shared = col_sets[0]
    for s in col_sets[1:]:
        shared &= s
    return sorted(shared)


class GLOBEM_MultiTaskDataset(Dataset):
    def __init__(self, cohort_dirs, seq_len=30, feat_cols=None, norm_stats=None):
        """
        cohort_dirs: list of cohort root paths (e.g. ['.../INS-W_1', ...])
        feat_cols: if provided, use this column list (for val/test consistency)
        norm_stats: (means, stds) tuple for val/test normalization
        """
        self.seq_len = seq_len

        rapids_list, survey_list = [], []
        for d in cohort_dirs:
            r, s = _load_cohort(d)
            rapids_list.append(r)
            survey_list.append(s)

        # align features across cohorts
        if feat_cols is None:
            feat_cols = _get_shared_cols(rapids_list)
        self.feat_cols = feat_cols
        self.feat_dim = len(feat_cols)

        # merge all cohorts
        self.rapids = pd.concat([r[["pid", "date"] + feat_cols] for r in rapids_list], ignore_index=True)
        self.survey = pd.concat(survey_list, ignore_index=True)
        self.survey = self.survey[self.survey["pid"].isin(self.rapids["pid"].unique())]
        self.users = self.survey["pid"].unique()

        # normalize (fit on this split only if norm_stats not given)
        if norm_stats is None:
            means = self.rapids[feat_cols].mean()
            stds = self.rapids[feat_cols].std().replace(0, 1)
            self.norm_stats = (means, stds)
        else:
            means, stds = norm_stats
            self.norm_stats = norm_stats

        self.rapids[feat_cols] = (self.rapids[feat_cols] - means) / stds
        self.rapids[feat_cols] = self.rapids[feat_cols].fillna(0.0)

        print(f"dataset ready: {len(self.users)} users, {self.feat_dim} features, {len(cohort_dirs)} cohort(s)")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        row = self.survey[self.survey["pid"] == uid].iloc[0]

        y = torch.tensor(row[TARGET_COLS].values.astype(np.float32))
        ehr = torch.tensor(row[BASELINE_COLS].values.astype(np.float32))

        feats = self.rapids[self.rapids["pid"] == uid].sort_values("date")
        seq = feats[self.feat_cols].tail(self.seq_len).values

        if len(seq) < self.seq_len:
            pad = np.full((self.seq_len - len(seq), self.feat_dim), np.nan)
            seq = np.vstack([pad, seq])

        mask = (~np.isnan(seq)).astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0).astype(np.float32)

        return torch.from_numpy(seq), torch.from_numpy(mask), ehr, y


def make_splits(cohort_dirs, seq_len=30, val_ratio=0.15, test_ratio=0.15, seed=42):
    """build train/val/test datasets — loads each cohort only once"""
    rng = np.random.default_rng(seed)

    # load all cohorts once, skip any missing required columns
    print("loading cohorts...")
    rapids_list, survey_list = [], []
    for d in cohort_dirs:
        print(f"  {os.path.basename(d)}")
        r, s = _load_cohort(d)
        if r is None:
            continue
        rapids_list.append(r)
        survey_list.append(s)

    if not rapids_list:
        raise RuntimeError("no cohorts loaded — check column names above")

    feat_cols = _get_shared_cols(rapids_list)
    print(f"shared features: {len(feat_cols)}")

    # collect user IDs
    all_users, cohort_labels = [], []
    for i, s in enumerate(survey_list):
        uids = s["pid"].unique()
        all_users.extend(uids)
        cohort_labels.extend([i] * len(uids))
    all_users = np.array(all_users)

    # random user-level split
    n = len(all_users)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    train_users = set(all_users[idx[n_test + n_val:]])
    val_users   = set(all_users[idx[n_test:n_test + n_val]])
    test_users  = set(all_users[idx[:n_test]])

    # merge all data once, keeping only needed columns
    all_rapids = pd.concat(
        [r[["pid", "date"] + feat_cols] for r in rapids_list],
        ignore_index=True
    )
    all_survey = pd.concat(survey_list, ignore_index=True)
    valid_pids = set(all_rapids["pid"].unique())
    all_survey = all_survey[all_survey["pid"].isin(valid_pids)]

    # normalize using train split stats only (no leakage)
    train_rapids = all_rapids[all_rapids["pid"].isin(train_users)]
    means = train_rapids[feat_cols].mean()
    stds  = train_rapids[feat_cols].std().replace(0, 1)
    norm_stats = (means, stds)

    all_rapids = all_rapids.copy()
    all_rapids[feat_cols] = (all_rapids[feat_cols] - means) / stds
    all_rapids[feat_cols] = all_rapids[feat_cols].fillna(0.0)

    def _make_subset(user_set):
        ds = GLOBEM_MultiTaskDataset.__new__(GLOBEM_MultiTaskDataset)
        ds.seq_len   = seq_len
        ds.feat_cols = feat_cols
        ds.feat_dim  = len(feat_cols)
        ds.norm_stats = norm_stats
        ds.rapids  = all_rapids[all_rapids["pid"].isin(user_set)]
        ds.survey  = all_survey[all_survey["pid"].isin(user_set)]
        ds.users   = ds.survey["pid"].unique()
        return ds

    train_ds = _make_subset(train_users)
    val_ds   = _make_subset(val_users)
    test_ds  = _make_subset(test_users)

    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_ds, val_ds, test_ds
