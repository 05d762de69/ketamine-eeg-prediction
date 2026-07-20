#!/usr/bin/env python3
"""Full-montage drug-state classification: nested cross-validation + permutation tests.

Trains three feature-set models to distinguish ketamine from awake eyes-closed
epochs:

    A  bandpower         (24 features: 4 bands x 6 regions)
    B  wPLI connectivity (5673 edges: 3 bands x C(62,2))
    C  A + B combined    (5697 features)

Each feature set is classified with **three estimator families** -- random
forest, RBF-SVM and XGBoost -- so that a null (or positive) result cannot be
attributed to a single algorithm (see ``ketamine_eeg.models``). For the wPLI
sets B and C a shared PCA(100) precedes every classifier, so only the estimator
varies, not the representation.

Every (estimator x model) combination is evaluated with subject-aware nested
cross-validation. There are 10 subjects, so the outer GroupKFold with
``OUTER_SPLITS=10`` is leave-one-subject-out; the inner GroupKFold(4) grid
search is scored by balanced accuracy. Statistical significance is assessed
with a permutation test (1000 subject-aware recording-label shuffles), refitting
the selected pipeline per fold and computing centred two-sided p-values for
balanced accuracy, ROC-AUC and accuracy.

Set ``KETAMINE_SMOKE=1`` to run a fast reduced version (tiny grids, few
permutations, outputs under ``results/_smoke/``) for correctness checks.

Inputs:
    data/derived/features/features_A_bandpower_epochwise.csv
    data/derived/features/features_B_wpli_edges_epoch_sliding.csv
Outputs (results/):
    models/results_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    models/predictions_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    models/best_params_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    permutation/perm_summary_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    permutation/null_{bacc,auc,acc}_<estimator>_<model>.npy

Runs from the tracked derived features -- no raw EEG required.
"""
from __future__ import annotations

import os

# Pin BLAS / OpenMP to a single thread per process BEFORE numpy / xgboost import,
# so the permutation test's process workers (and each GridSearch worker) stay
# single-threaded instead of each spawning one thread per core and oversubscribing
# the machine (XGBoost defaults to all cores via OpenMP even at n_jobs=1).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402
from ketamine_eeg import features as feat  # noqa: E402
from ketamine_eeg import models as km  # noqa: E402
from ketamine_eeg import stats as kstats  # noqa: E402

if os.environ.get("KETAMINE_SMOKE") == "1":
    # config applies smoke mode at import (parent + worker processes); just note it.
    print("[SMOKE] reduced grids / permutations; outputs under results/_smoke/")

c.ensure_dirs()

N_CORES = os.cpu_count() or 8
PIPELINE_CACHE_DIR = c.PROJECT_ROOT / ".cache" / "sklearn_pipeline_models"

# --------------------------------------------------------------------------- #
# Load + build design matrices
# --------------------------------------------------------------------------- #
df = feat.load_and_merge_AB()
X_by_model, y, groups, rec_id, feat_cols = feat.build_design_matrices(df)
XA, XB, XAB = X_by_model[c.MODEL_A], X_by_model[c.MODEL_B], X_by_model[c.MODEL_C]

print(f"n A features: {XA.shape[1]}  n B features: {XB.shape[1]}  n A+B: {XAB.shape[1]}")
assert XA.shape[1] == 24, f"Set A must be 24, got {XA.shape[1]}"
assert XB.shape[1] == 5673, f"Set B must be 5673, got {XB.shape[1]}"
assert XAB.shape[1] == 5697, f"Set C must be 5697, got {XAB.shape[1]}"
n_subjects = df["subject_id"].nunique()
assert n_subjects >= c.OUTER_SPLITS, (
    f"OUTER_SPLITS={c.OUTER_SPLITS} needs >= {c.OUTER_SPLITS} subjects "
    f"(GroupKFold cannot exceed the number of groups); found {n_subjects}."
)
print(f"Epochs: {len(df)}  subjects: {n_subjects}  "
      f"ketamine: {int(y.sum())}  awake: {int((1 - y).sum())}")
print(f"Estimators: {list(c.ESTIMATORS)}  outer folds: {c.OUTER_SPLITS} "
      f"({'LOSO' if c.OUTER_SPLITS == n_subjects else 'GroupKFold'})")

# Group-only splits (label-independent, deterministic; shared by all estimators)
splits = list(GroupKFold(n_splits=c.OUTER_SPLITS).split(XA, y, groups))


# --------------------------------------------------------------------------- #
# STEP 1 -- nested CV grid search (per estimator x model x fold)
# --------------------------------------------------------------------------- #
def _fit_eval_one_task(X, estimator, model_name, fold, tr, te, n_jobs):
    gte = groups[te]
    est, space = km.build_estimator_and_grid(estimator, model_name, subset=False)
    est.memory = str(PIPELINE_CACHE_DIR)   # cache PCA/scaler across grid candidates
    gs = GridSearchCV(est, param_grid=space, cv=GroupKFold(c.INNER_SPLITS),
                      scoring="balanced_accuracy", n_jobs=n_jobs, refit=True)
    gs.fit(X[tr], y[tr], groups=groups[tr])
    best = gs.best_estimator_
    yhat, proba = km.predict_and_score(best, X[te])
    try:
        aucv = float(roc_auc_score(y[te], proba))
    except Exception:
        aucv = np.nan
    cm = confusion_matrix(y[te], yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
    param_row = {"estimator": estimator, "model": model_name, "fold": fold,
                 "n_iter": int(len(gs.cv_results_["params"])),
                 "best_score_inner": float(gs.best_score_), **gs.best_params_}
    if km.uses_pca(model_name, subset=False):
        param_row["pca_n_components"] = int(c.PCA_N_COMPONENTS)
    fold_row = {"estimator": estimator, "model": model_name, "fold": fold,
                "n_test": int(len(te)), "n_test_subjects": int(len(np.unique(gte))),
                "accuracy": float(accuracy_score(y[te], yhat)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], yhat)),
                "roc_auc": aucv, "tn": float(tn), "fp": float(fp),
                "fn": float(fn), "tp": float(tp)}
    pred_rows = [{"estimator": estimator, "model": model_name, "fold": fold,
                  "subject_id": gte[i], "y_true": int(y[te][i]),
                  "y_proba": float(proba[i]), "y_pred": int(yhat[i])}
                 for i in range(len(te))]
    # Unfitted clone of the winning pipeline -> refit on shuffled labels in STEP 2.
    tmpl = clone(best)
    return fold_row, pred_rows, param_row, (estimator, model_name, fold, tmpl)


# Wipe any stale PCA/scaler cache so no cross-run cache hit can corrupt results.
shutil.rmtree(PIPELINE_CACHE_DIR, ignore_errors=True)
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

tasks = [(X_by_model[m], est, m, f, tr, te)
         for est in c.ESTIMATORS for m in X_by_model
         for f, (tr, te) in enumerate(splits, 1)]
print(f"\n[STEP 1] nested CV: {len(tasks)} tasks "
      f"({len(c.ESTIMATORS)} estimators x {len(X_by_model)} models x "
      f"{c.OUTER_SPLITS} folds) across {N_CORES} cores")
t0 = time.time()
results = []
for (X, est, m, f, tr, te) in tasks:
    ts = time.time()
    results.append(_fit_eval_one_task(X, est, m, f, tr, te, N_CORES))
    print(f"  {est:3s} {m} fold {f}: {time.time() - ts:.0f}s", flush=True)
print(f"[STEP 1] done in {time.time() - t0:.0f}s")

sort_cols = ["estimator", "model", "fold"]
folds = pd.DataFrame([r[0] for r in results]).sort_values(sort_cols).reset_index(drop=True)
preds = pd.DataFrame([pr for r in results for pr in r[1]]).sort_values(sort_cols).reset_index(drop=True)
params = pd.DataFrame([r[2] for r in results]).sort_values(sort_cols).reset_index(drop=True)
BEST_BY_KEY = {(e, mn, fd): tmpl for (_, _, _, (e, mn, fd, tmpl)) in results}

folds.to_csv(c.RESULTS_MAIN_PATH, index=False)
preds.to_csv(c.PREDS_MAIN_PATH, index=False)
params.to_csv(c.PARAMS_MAIN_PATH, index=False)

print("\n[STEP 1] per-fold results (mean over folds):")
print(folds.groupby(["estimator", "model"])[["balanced_accuracy", "roc_auc", "accuracy"]]
      .mean().round(4).to_string())


# --------------------------------------------------------------------------- #
# STEP 2 -- permutation test (balanced accuracy + ROC-AUC + accuracy)
# --------------------------------------------------------------------------- #
rec_y, subj_rec_idx_list, epoch_to_rec_idx = kstats.build_recording_shuffle_structures(
    rec_id, y, groups
)


def _run_perm_chunk(seed_list):
    out = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        y_perm = kstats.subject_aware_label_shuffle(rec_y, subj_rec_idx_list,
                                                    epoch_to_rec_idx, rng)
        rec = {}
        for estimator in c.ESTIMATORS:
            for model_name, X in X_by_model.items():
                ba, au, ac = [], [], []
                for fold, (tr, te) in enumerate(splits, 1):
                    est = clone(BEST_BY_KEY[(estimator, model_name, fold)])
                    est.fit(X[tr], y_perm[tr])
                    yhat, proba = km.predict_and_score(est, X[te])
                    ba.append(balanced_accuracy_score(y_perm[te], yhat))
                    ac.append(accuracy_score(y_perm[te], yhat))
                    try:
                        au.append(roc_auc_score(y_perm[te], proba))
                    except Exception:
                        au.append(np.nan)
                rec[(estimator, model_name)] = (
                    float(np.mean(ba)), float(np.nanmean(au)), float(np.mean(ac)))
        out.append(rec)
    return out


seeds = [c.SEED + 1000 + i for i in range(c.N_PERM)]
chunks = [ch.tolist() for ch in np.array_split(np.array(seeds, int), N_CORES) if len(ch)]
print(f"\n[STEP 2] permutation: {c.N_PERM} perms x {len(c.ESTIMATORS)} estimators x "
      f"{len(X_by_model)} models x {c.OUTER_SPLITS} folds, {N_CORES} cores")
t0 = time.time()
chunked = Parallel(n_jobs=N_CORES, prefer="processes", batch_size=1)(
    delayed(_run_perm_chunk)(ch) for ch in chunks)
perm_results = [r for sub in chunked for r in sub]
print(f"[STEP 2] done in {time.time() - t0:.0f}s ({len(perm_results)} perms)")

keys = [(e, m) for e in c.ESTIMATORS for m in X_by_model]
null = {k: {"bacc": np.array([r[k][0] for r in perm_results]),
            "auc": np.array([r[k][1] for r in perm_results]),
            "acc": np.array([r[k][2] for r in perm_results])} for k in keys}


def _obs(estimator, model_name, metric_col):
    g = folds[(folds.estimator == estimator) & (folds.model == model_name)]
    return float(g[metric_col].mean())


rows = []
for (e, m) in keys:
    r = {"estimator": e, "model": m}
    obs = {"bacc": _obs(e, m, "balanced_accuracy"), "auc": _obs(e, m, "roc_auc"),
           "acc": _obs(e, m, "accuracy")}
    for met in ("bacc", "auc", "acc"):
        arr = np.asarray(null[(e, m)][met])
        arr = arr[~np.isnan(arr)]
        r[f"obs_{met}"] = obs[met]
        r[f"null_{met}_mean"] = float(arr.mean())
        r[f"p_perm_{met}"] = kstats.two_sided_p(arr, obs[met])
        r[f"p_ttest_{met}"] = float(stats.ttest_1samp(arr, popmean=obs[met],
                                                       alternative="two-sided").pvalue)
    rows.append(r)
perm_summary = pd.DataFrame(rows)
perm_summary.to_csv(c.PERM_SUMMARY_MAIN_PATH, index=False)
for (e, m) in keys:
    np.save(c.PERM_DIR / f"null_bacc_{e}_{m}.npy", null[(e, m)]["bacc"])
    np.save(c.PERM_DIR / f"null_auc_{e}_{m}.npy", null[(e, m)]["auc"])
    np.save(c.PERM_DIR / f"null_acc_{e}_{m}.npy", null[(e, m)]["acc"])

print("\n[STEP 2] permutation summary (two-sided p):")
print(perm_summary[["estimator", "model", "obs_bacc", "p_perm_bacc", "obs_auc",
                    "p_perm_auc", "obs_acc", "p_perm_acc"]]
      .to_string(index=False, float_format="%.4f"))
print("\nSaved model results, predictions, best params and permutation outputs.")
