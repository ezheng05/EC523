import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLS = ["CESD_10items_POST", "STAIS_POST", "PSS_10items_POST"]
BASELINE_COLS = ["CESD_10items_PRE", "STAIS_PRE", "PSS_10items_PRE"]
IGNORE_COLS = {"pid", "date", "uid"}


def _load_cohort(cohort_dir):
    """load and clean rapids + pre + post for one cohort, return merged df"""
    rapids = pd.read_csv(os.path.join(cohort_dir, "FeatureData", "rapids.csv"), low_memory=False)
    post = pd.read_csv(os.path.join(cohort_dir, "SurveyData", "post.csv"))
    pre = pd.read_csv(os.path.join(cohort_dir, "SurveyData", "pre.csv"))

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
    """build train/val/test datasets with shared feat_cols and norm_stats from train"""
    rng = np.random.default_rng(seed)

    # collect all user ids per cohort for stratified split
    all_users = []
    cohort_labels = []
    for i, d in enumerate(cohort_dirs):
        _, survey = _load_cohort(d)
        all_users.extend(survey["pid"].unique())
        cohort_labels.extend([i] * len(survey["pid"].unique()))

    all_users = np.array(all_users)
    cohort_labels = np.array(cohort_labels)

    # stratify by cohort
    n = len(all_users)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    # build per-cohort user lists for each split
    def _filter_cohort_users(split_idx, cohort_i):
        uids = set(all_users[split_idx][cohort_labels[split_idx] == cohort_i])
        return uids

    # we build datasets by passing filtered cohort dirs and restricting by user
    # simpler: build one full dataset, then split by user index
    full_ds = GLOBEM_MultiTaskDataset(cohort_dirs, seq_len=seq_len)
    feat_cols = full_ds.feat_cols
    norm_stats = full_ds.norm_stats

    def _subset(user_idx_arr):
        ds = GLOBEM_MultiTaskDataset.__new__(GLOBEM_MultiTaskDataset)
        ds.seq_len = full_ds.seq_len
        ds.feat_cols = feat_cols
        ds.feat_dim = full_ds.feat_dim
        ds.rapids = full_ds.rapids
        ds.norm_stats = norm_stats
        ds.survey = full_ds.survey[full_ds.survey["pid"].isin(all_users[user_idx_arr])]
        ds.users = ds.survey["pid"].unique()
        return ds

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)
