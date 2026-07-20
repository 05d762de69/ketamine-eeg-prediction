#!/usr/bin/env python3
"""Channel-subset drug-state models: can a few frontal/lateral electrodes work?

Repeats the A / B / C classification for four reduced electrode montages
(left lateral, right lateral, both lateral, and a 4-channel BIS-Quatro-style
frontal strip), so the full 62-channel result can be compared against sparse,
clinically practical montages. As in the full-montage analysis, every feature
set is classified with three estimator families -- random forest, RBF-SVM and
XGBoost -- so a sparse-montage result is not an artefact of one algorithm. The
reduced feature spaces are low dimensional, so no PCA is used (SVM standardises).
Same subject-aware nested CV (outer leave-one-subject-out GroupKFold, inner
GroupKFold(4)) and permutation test (1000 subject-aware shuffles) as script 03.

Set ``KETAMINE_SMOKE=1`` for a fast reduced run (outputs under ``results/_smoke/``).

Inputs (tracked derived data -- no raw EEG required):
    data/derived/features/features_subset_bandpower.csv   (per-channel bandpower)
    data/derived/features/features_B_wpli_matrices_epoch_sliding.npz
    data/derived/montage_info.json                        (channel order)
Outputs (results/):
    models/{results,predictions,best_params}_channel_subsets.csv
    permutation/perm_summary_channel_subsets.csv
    permutation/null_channel_subsets_{bacc,auc}.npz
"""
from __future__ import annotations

import json
import os
import sys

# Pin BLAS / OpenMP to a single thread per process BEFORE numpy / xgboost import.
# This script fans out many single-threaded estimator fits across process workers;
# without this, each worker would additionally spawn one BLAS/OpenMP thread per
# core (XGBoost defaults to all cores via OpenMP even at n_jobs=1), oversubscribing
# the machine (workers x cores threads) and slowing the run by an order of magnitude.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402
from ketamine_eeg import models as km  # noqa: E402
from ketamine_eeg import stats as kstats  # noqa: E402

if os.environ.get("KETAMINE_SMOKE") == "1":
    # config applies smoke mode at import (parent + worker processes); just note it.
    print("[SMOKE] reduced grids / permutations; outputs under results/_smoke/")

c.ensure_dirs()

N_CORES = os.cpu_count() or 8
N_TASKS_HINT = len(c.ESTIMATORS) * len(c.CHANNEL_SUBSETS) * 3 * c.OUTER_SPLITS
N_OUTER_JOBS = min(N_CORES, N_TASKS_HINT)
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
n_subjects = bp_df["subject_id"].nunique()
assert n_subjects >= c.OUTER_SPLITS, (
    f"OUTER_SPLITS={c.OUTER_SPLITS} needs >= {c.OUTER_SPLITS} subjects; found {n_subjects}."
)


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
print(f"Estimators: {list(c.ESTIMATORS)}  outer folds: {c.OUTER_SPLITS} "
      f"({'LOSO' if c.OUTER_SPLITS == n_subjects else 'GroupKFold'})")


# --------------------------------------------------------------------------- #
# Nested CV grid search (one GridSearch per estimator x spec x outer fold)
# --------------------------------------------------------------------------- #
def fit_eval_one(X, estimator, model_name, subset, feat_type, fold, tr, te):
    gte = groups[te]
    est, space = km.build_estimator_and_grid(estimator, model_name, subset=True)
    gs = GridSearchCV(est, param_grid=space, cv=GroupKFold(c.INNER_SPLITS),
                      scoring="balanced_accuracy", n_jobs=N_INNER_JOBS, refit=True)
    gs.fit(X[tr], y[tr], groups=groups[tr])
    best = gs.best_estimator_
    yhat, proba = km.predict_and_score(best, X[te])
    try:
        auc = float(roc_auc_score(y[te], proba))
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y[te], yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
    fold_row = {"estimator": estimator, "model": model_name, "subset": subset,
                "feat_type": feat_type, "fold": fold,
                "n_test": int(len(te)), "n_test_subjects": int(len(np.unique(gte))),
                "accuracy": float(accuracy_score(y[te], yhat)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], yhat)),
                "roc_auc": auc, "best_score_inner": float(gs.best_score_),
                "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}
    pred_rows = [{"estimator": estimator, "model": model_name, "subset": subset,
                  "feat_type": feat_type, "fold": fold, "subject_id": gte[i],
                  "y_true": int(y[te][i]), "y_proba": float(proba[i]),
                  "y_pred": int(yhat[i])} for i in range(len(te))]
    param_row = {"estimator": estimator, "model": model_name, "subset": subset,
                 "feat_type": feat_type, "fold": fold,
                 "best_score_inner": float(gs.best_score_), **gs.best_params_}
    return fold_row, pred_rows, param_row, (estimator, model_name, fold, clone(best))


tasks = [(s.X, est, s.model_name, s.subset, s.feat_type, fold, tr, te)
         for est in c.ESTIMATORS for s in all_specs
         for fold, (tr, te) in enumerate(splits, 1)]
print(f"\nLaunching {len(tasks)} outer tasks across {N_OUTER_JOBS} workers ...")
results = Parallel(n_jobs=N_OUTER_JOBS, prefer="processes")(delayed(fit_eval_one)(*t) for t in tasks)

sort_cols = ["estimator", "subset", "feat_type", "fold"]
folds_df = pd.DataFrame([r[0] for r in results]).sort_values(sort_cols).reset_index(drop=True)
preds_df = pd.DataFrame([pr for r in results for pr in r[1]])
params_df = pd.DataFrame([r[2] for r in results]).sort_values(sort_cols).reset_index(drop=True)
BEST_BY_KEY = {(e, mn, fd): tmpl for (_, _, _, (e, mn, fd, tmpl)) in results}

folds_df.to_csv(c.RESULTS_SUBSET_PATH, index=False)
preds_df.to_csv(c.PREDS_SUBSET_PATH, index=False)
params_df.to_csv(c.PARAMS_SUBSET_PATH, index=False)
print(folds_df.groupby(["estimator", "subset", "feat_type"])[["balanced_accuracy", "roc_auc"]]
      .mean().round(4).to_string())


# --------------------------------------------------------------------------- #
# Permutation null (subject-aware recording-label shuffle; refit winning pipeline)
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

SPEC_BY_NAME = {s.model_name: s for s in all_specs}


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
        for estimator in c.ESTIMATORS:
            for spec in all_specs:
                fb, fa = [], []
                for fold, (tr, te) in enumerate(splits, 1):
                    est = clone(BEST_BY_KEY[(estimator, spec.model_name, fold)])
                    est.fit(spec.X[tr], y_perm[tr])
                    yhat, proba = km.predict_and_score(est, spec.X[te])
                    fb.append(balanced_accuracy_score(y_perm[te], yhat))
                    try:
                        fa.append(roc_auc_score(y_perm[te], proba))
                    except Exception:
                        fa.append(np.nan)
                per_bacc[(estimator, spec.model_name)] = float(np.nanmean(fb))
                per_auc[(estimator, spec.model_name)] = float(np.nanmean(fa))
        out_chunk.append({"bacc": per_bacc, "auc": per_auc})
    return out_chunk


seeds = [c.SEED + 1000 + i for i in range(c.N_PERM)]
chunks = [ch.tolist() for ch in np.array_split(np.array(seeds, int), N_CORES) if len(ch)]
print(f"\nRunning {c.N_PERM} permutations x {len(c.ESTIMATORS)} estimators "
      f"across {len(chunks)} workers ...")
chunked = Parallel(n_jobs=N_CORES, prefer="processes", batch_size=1)(
    delayed(run_perm_chunk)(ch) for ch in chunks)
perm_flat = [r for sub in chunked for r in sub]

keys = [(e, s.model_name) for e in c.ESTIMATORS for s in all_specs]
null_bacc = {k: np.array([r["bacc"][k] for r in perm_flat]) for k in keys}
null_auc = {k: np.array([r["auc"][k] for r in perm_flat]) for k in keys}

obs_bacc = folds_df.groupby(["estimator", "subset", "feat_type"])["balanced_accuracy"].mean()
obs_auc = folds_df.groupby(["estimator", "subset", "feat_type"])["roc_auc"].mean()

perm_rows = []
for e in c.ESTIMATORS:
    for s in all_specs:
        ob = float(obs_bacc.loc[e, s.subset, s.feat_type])
        oa = float(obs_auc.loc[e, s.subset, s.feat_type])
        nb, na = null_bacc[(e, s.model_name)], null_auc[(e, s.model_name)]
        # NaN-filter before centring / p-value so this matches script 03's
        # kstats.two_sided_p semantics exactly (AUC can be NaN if a fold lacks a class).
        nb_m = float(np.mean(nb[~np.isnan(nb)]))
        na_m = float(np.mean(na[~np.isnan(na)]))
        p_b = kstats.two_sided_p(nb, ob)
        p_a = kstats.two_sided_p(na, oa)
        perm_rows.append({"estimator": e, "model": s.model_name, "subset": s.subset,
                          "feat_type": s.feat_type,
                          "obs_bacc": ob, "null_bacc_mean": nb_m, "p_perm_bacc": p_b,
                          "obs_auc": oa, "null_auc_mean": na_m, "p_perm_auc": p_a})
perm_df = pd.DataFrame(perm_rows).sort_values(["estimator", "p_perm_bacc"]).reset_index(drop=True)
perm_df.to_csv(c.PERM_SUMMARY_SUBSET_PATH, index=False)
# npz keys cannot be tuples -> join estimator and model name.
np.savez(c.PERM_DIR / "null_channel_subsets_bacc.npz",
         **{f"{e}__{m}": null_bacc[(e, m)] for (e, m) in keys})
np.savez(c.PERM_DIR / "null_channel_subsets_auc.npz",
         **{f"{e}__{m}": null_auc[(e, m)] for (e, m) in keys})
print(perm_df.round(4).to_string(index=False))
print("\nSaved channel-subset results + permutation outputs.")
