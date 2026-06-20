#!/usr/bin/env python3
"""Channel-subset drug-state models: can a few frontal/lateral electrodes work?

Repeats the A / B / C classification for four reduced electrode montages
(left lateral, right lateral, both lateral, and a 4-channel BIS-Quatro-style
frontal strip), so the full 62-channel result can be compared against sparse,
clinically practical montages. Same nested 5x4 GroupKFold design and
permutation test (1000 subject-aware shuffles) as the full-montage models.

Inputs (tracked derived data -- no raw EEG required):
    data/derived/features/features_subset_bandpower.csv   (per-channel bandpower)
    data/derived/features/features_B_wpli_matrices_epoch_sliding.npz
    data/derived/montage_info.json                        (channel order)
Outputs (results/):
    models/{results,predictions,best_params}_rf_channel_subsets.csv
    permutation/perm_summary_channel_subsets.csv
    permutation/null_channel_subsets_{bacc,auc}.npz
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402

c.ensure_dirs()

N_CORES = os.cpu_count() or 8
N_OUTER_JOBS = min(N_CORES, len(c.CHANNEL_SUBSETS) * 3 * c.OUTER_SPLITS)
N_INNER_JOBS = max(1, N_CORES // N_OUTER_JOBS)

# --------------------------------------------------------------------------- #
# Channel metadata + per-subset feature layouts
# --------------------------------------------------------------------------- #
CH_NAMES = json.loads(c.MONTAGE_INFO_PATH.read_text())["ch_names"]


@dataclass
class SubsetMeta:
    name: str
    channels: list
    ch_idx: list = field(init=False)
    triu_r: np.ndarray = field(init=False)
    triu_c: np.ndarray = field(init=False)
    bp_cols: list = field(init=False)

    def __post_init__(self):
        self.ch_idx = [CH_NAMES.index(ch) for ch in self.channels]
        self.triu_r, self.triu_c = np.triu_indices(len(self.channels), k=1)
        self.bp_cols = [f"bp_{band}_{ch}" for band in c.BANDS_HZ for ch in self.channels]


SUBSET_META = {name: SubsetMeta(name, chs) for name, chs in c.CHANNEL_SUBSETS.items()}

# --------------------------------------------------------------------------- #
# Load features (bandpower table + dense wPLI matrices)
# --------------------------------------------------------------------------- #
bp_df = pd.read_csv(c.FEAT_SUBSET_BANDPOWER_PATH)
npz = np.load(c.FEAT_B_MATRICES_PATH)

wpli_arrays: dict = {}
for sm in SUBSET_META.values():
    rows = []
    for _, row in bp_df.iterrows():
        sid, recnum, eidx = str(int(row["subject_id"])), int(row["recording_number"]), int(row["epoch_index_original"])
        edge_vec = []
        for band in c.WPLI_BANDS:
            mat = npz[f"{sid}__rec{recnum}__e{eidx:04d}__{band}"]
            sub = mat[np.ix_(sm.ch_idx, sm.ch_idx)]
            edge_vec.append(sub[sm.triu_r, sm.triu_c])
        rows.append(np.concatenate(edge_vec))
    wpli_arrays[sm.name] = np.array(rows)
    print(f"{sm.name} wPLI: {wpli_arrays[sm.name].shape}")

y = (bp_df["drug"] == "ketamine").astype(int).to_numpy()
groups = bp_df["subject_id"].astype(str).to_numpy()
rec_id = (bp_df["subject_id"].astype(str) + "||" + bp_df["recording_number"].astype(str)).to_numpy()


@dataclass(frozen=True)
class ModelSpec:
    X: np.ndarray
    model_name: str
    subset: str
    feat_type: str
    n_features: int


all_specs = []
for sm in SUBSET_META.values():
    XA = bp_df[sm.bp_cols].to_numpy()
    XB = wpli_arrays[sm.name]
    XC = np.concatenate([XA, XB], axis=1)
    for feat_type, X in [("A", XA), ("B", XB), ("C", XC)]:
        all_specs.append(ModelSpec(X, f"{sm.name}__{feat_type}", sm.name, feat_type, X.shape[1]))

splits = list(GroupKFold(n_splits=c.OUTER_SPLITS).split(all_specs[0].X, y, groups))


# --------------------------------------------------------------------------- #
# Nested CV grid search (one GridSearch per outer task)
# --------------------------------------------------------------------------- #
def fit_eval_one(X, model_name, subset, feat_type, fold, tr, te):
    gte = groups[te]
    gs = GridSearchCV(
        RandomForestClassifier(random_state=c.SEED, class_weight="balanced_subsample", n_jobs=1),
        param_grid=c.PARAM_GRID_RF_SUBSET, cv=GroupKFold(c.INNER_SPLITS),
        scoring="balanced_accuracy", n_jobs=N_INNER_JOBS, refit=True,
    )
    gs.fit(X[tr], y[tr], groups=groups[tr])
    best = gs.best_estimator_
    proba = best.predict_proba(X[te])[:, 1]
    yhat = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y[te], proba))
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y[te], yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
    fold_row = {"model": model_name, "subset": subset, "feat_type": feat_type, "fold": fold,
                "n_test": int(len(te)), "n_test_subjects": int(len(np.unique(gte))),
                "accuracy": float(accuracy_score(y[te], yhat)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], yhat)),
                "roc_auc": auc, "best_score_inner": float(gs.best_score_),
                "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}
    pred_rows = [{"model": model_name, "subset": subset, "feat_type": feat_type, "fold": fold,
                  "subject_id": gte[i], "y_true": int(y[te][i]),
                  "y_proba": float(proba[i]), "y_pred": int(yhat[i])} for i in range(len(te))]
    param_row = {"model": model_name, "subset": subset, "feat_type": feat_type, "fold": fold,
                 "best_score_inner": float(gs.best_score_), **gs.best_params_}
    return fold_row, pred_rows, param_row


tasks = [(s.X, s.model_name, s.subset, s.feat_type, fold, tr, te)
         for s in all_specs for fold, (tr, te) in enumerate(splits, 1)]
print(f"\nLaunching {len(tasks)} outer tasks across {N_OUTER_JOBS} workers ...")
results = Parallel(n_jobs=N_OUTER_JOBS, prefer="processes")(delayed(fit_eval_one)(*t) for t in tasks)

folds_df = pd.DataFrame([r[0] for r in results]).sort_values(["subset", "feat_type", "fold"]).reset_index(drop=True)
preds_df = pd.DataFrame([pr for r in results for pr in r[1]])
params_df = pd.DataFrame([r[2] for r in results]).sort_values(["subset", "feat_type", "fold"]).reset_index(drop=True)

folds_df.to_csv(c.RESULTS_SUBSET_PATH, index=False)
preds_df.to_csv(c.PREDS_SUBSET_PATH, index=False)
params_df.to_csv(c.PARAMS_SUBSET_PATH, index=False)
print(folds_df.groupby(["subset", "feat_type"])[["balanced_accuracy", "roc_auc"]].mean().round(4).to_string())


# --------------------------------------------------------------------------- #
# Permutation null (subject-aware recording-label shuffle; refit saved params)
# --------------------------------------------------------------------------- #
rec_lookup = pd.DataFrame({"rec_id": rec_id, "y": y, "subject": groups})
assert (rec_lookup.groupby("rec_id")["y"].nunique() <= 1).all(), "label not constant within recording"
rec_label = rec_lookup.groupby("rec_id")["y"].first()
rec_subject = rec_lookup.groupby("rec_id")["subject"].first()
rec_ids_unique = rec_label.index.to_numpy()
rec_y = rec_label.to_numpy().astype(int)
rec_subj = rec_subject.to_numpy()
epoch_to_rec_idx = pd.Index(rec_ids_unique).get_indexer(rec_id)
subj_to_rec: dict = {}
for i, s in enumerate(rec_subj):
    subj_to_rec.setdefault(s, []).append(i)
subj_rec_idx_list = [np.asarray(v, int) for v in subj_to_rec.values() if len(v) > 1]

INT_KEYS = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}


def _coerce(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if kind == "int":
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return v
    if kind == "max_features":
        if isinstance(v, str):
            s = v.strip()
            if s in {"sqrt", "log2"}:
                return s
            try:
                f = float(s)
            except ValueError:
                return s
            return int(f) if f.is_integer() else f
        if isinstance(v, float) and float(v).is_integer():
            return int(v)
        return v
    return v


BEST_BY_MODEL_FOLD = {}
for _, prow in params_df.iterrows():
    rfp = {}
    for k in c.PARAM_GRID_RF_SUBSET:
        if k in prow:
            kind = "int" if k in INT_KEYS else ("max_features" if k == "max_features" else None)
            rfp[k] = _coerce(prow[k], kind)
    BEST_BY_MODEL_FOLD[(prow["model"], int(prow["fold"]))] = rfp


def _make_estimator(model_name, fold):
    rf = RandomForestClassifier(random_state=c.SEED, class_weight="balanced_subsample", n_jobs=1)
    rf.set_params(**BEST_BY_MODEL_FOLD[(model_name, fold)])
    return rf


def run_perm_chunk(seed_list):
    out_chunk = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        perm_rec_y = rec_y.copy()
        for idxs in subj_rec_idx_list:
            vals = perm_rec_y[idxs].copy()
            rng.shuffle(vals)
            perm_rec_y[idxs] = vals
        y_perm = perm_rec_y[epoch_to_rec_idx]
        per_bacc, per_auc = {}, {}
        for spec in all_specs:
            fb, fa = [], []
            for fold, (tr, te) in enumerate(splits, 1):
                est = _make_estimator(spec.model_name, fold)
                est.fit(spec.X[tr], y_perm[tr])
                proba = est.predict_proba(spec.X[te])[:, 1]
                yhat = (proba >= 0.5).astype(int)
                fb.append(balanced_accuracy_score(y_perm[te], yhat))
                try:
                    fa.append(roc_auc_score(y_perm[te], proba))
                except Exception:
                    fa.append(np.nan)
            per_bacc[spec.model_name] = float(np.nanmean(fb))
            per_auc[spec.model_name] = float(np.nanmean(fa))
        out_chunk.append({"bacc": per_bacc, "auc": per_auc})
    return out_chunk


seeds = [c.SEED + 1000 + i for i in range(c.N_PERM)]
chunks = [ch.tolist() for ch in np.array_split(np.array(seeds, int), N_CORES) if len(ch)]
print(f"\nRunning {c.N_PERM} permutations across {len(chunks)} workers ...")
chunked = Parallel(n_jobs=N_CORES, prefer="processes", batch_size=1)(
    delayed(run_perm_chunk)(ch) for ch in chunks)
perm_flat = [r for sub in chunked for r in sub]

null_bacc = {s.model_name: np.array([r["bacc"][s.model_name] for r in perm_flat]) for s in all_specs}
null_auc = {s.model_name: np.array([r["auc"][s.model_name] for r in perm_flat]) for s in all_specs}

obs_bacc = folds_df.groupby(["subset", "feat_type"])["balanced_accuracy"].mean()
obs_auc = folds_df.groupby(["subset", "feat_type"])["roc_auc"].mean()

perm_rows = []
for s in all_specs:
    ob, oa = float(obs_bacc.loc[s.subset, s.feat_type]), float(obs_auc.loc[s.subset, s.feat_type])
    nb, na = null_bacc[s.model_name], null_auc[s.model_name]
    nb_m, na_m = float(np.mean(nb)), float(np.nanmean(na))
    p_b = float((1 + np.sum(np.abs(nb - nb_m) >= abs(ob - nb_m))) / (len(nb) + 1))
    p_a = float((1 + np.sum(np.abs(na - na_m) >= abs(oa - na_m))) / (len(na) + 1))
    perm_rows.append({"model": s.model_name, "subset": s.subset, "feat_type": s.feat_type,
                      "obs_bacc": ob, "null_bacc_mean": nb_m, "p_perm_bacc": p_b,
                      "obs_auc": oa, "null_auc_mean": na_m, "p_perm_auc": p_a})
perm_df = pd.DataFrame(perm_rows).sort_values("p_perm_bacc").reset_index(drop=True)
perm_df.to_csv(c.PERM_SUMMARY_SUBSET_PATH, index=False)
np.savez(c.PERM_DIR / "null_channel_subsets_bacc.npz", **null_bacc)
np.savez(c.PERM_DIR / "null_channel_subsets_auc.npz", **null_auc)
print(perm_df.round(4).to_string(index=False))
print("\nSaved channel-subset results + permutation outputs.")
