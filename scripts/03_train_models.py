#!/usr/bin/env python3
"""Full-montage drug-state classification: nested cross-validation + permutation tests.

Trains three random-forest classifiers to distinguish ketamine from awake
eyes-closed epochs:

    A  bandpower         (24 features: 4 bands x 6 regions)
    B  wPLI connectivity (5673 edges: 3 bands x C(62,2))
    C  A + B combined    (5697 features)

Each model is evaluated with subject-aware nested cross-validation
(5 outer / 4 inner GroupKFold, grouped by subject) and an exhaustive
random-forest grid search scored by balanced accuracy. For B and C a PCA(100)
step precedes the forest. Statistical significance is assessed with a
permutation test (1000 subject-aware recording-label shuffles), refitting the
selected hyper-parameters per fold and computing centred two-sided p-values for
balanced accuracy, ROC-AUC and accuracy.

Inputs:
    data/derived/features/features_A_bandpower_epochwise.csv
    data/derived/features/features_B_wpli_edges_epoch_sliding.csv
Outputs (results/):
    models/results_rf_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    models/predictions_rf_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    models/best_params_rf_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    permutation/perm_summary_rf_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
    permutation/null_{bacc,auc,acc}_<model>.npy

Runs from the tracked derived features -- no raw EEG required.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402
from ketamine_eeg import features as feat  # noqa: E402
from ketamine_eeg import stats as kstats  # noqa: E402

c.ensure_dirs()

N_CORES = __import__("os").cpu_count() or 8
PIPELINE_CACHE_DIR = c.PROJECT_ROOT / ".cache" / "sklearn_pipeline_models"
PARAM_GRID_PCA_RF = {f"rf__{k}": v for k, v in c.PARAM_GRID_RF.items()}

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
print(f"Epochs: {len(df)}  subjects: {df['subject_id'].nunique()}  "
      f"ketamine: {int(y.sum())}  awake: {int((1 - y).sum())}")

# Group-only splits (label-independent, deterministic)
splits = list(GroupKFold(n_splits=c.OUTER_SPLITS).split(XA, y, groups))


# --------------------------------------------------------------------------- #
# STEP 1 -- nested CV grid search
# --------------------------------------------------------------------------- #
def _make_estimator_and_space(model_name: str):
    if model_name in c.USE_PCA_FOR:
        est = Pipeline(
            [
                ("pca", PCA(n_components=c.PCA_N_COMPONENTS, svd_solver="randomized",
                            random_state=c.SEED)),
                ("rf", RandomForestClassifier(random_state=c.SEED,
                                              class_weight="balanced_subsample", n_jobs=1)),
            ],
            memory=str(PIPELINE_CACHE_DIR),
        )
        return est, PARAM_GRID_PCA_RF
    return (RandomForestClassifier(random_state=c.SEED, class_weight="balanced_subsample",
                                   n_jobs=1), c.PARAM_GRID_RF)


def _fit_eval_one_task(X, model_name, fold, tr, te, n_jobs):
    gte = groups[te]
    estimator, space = _make_estimator_and_space(model_name)
    rs = GridSearchCV(estimator, param_grid=space, cv=GroupKFold(c.INNER_SPLITS),
                      scoring="balanced_accuracy", n_jobs=n_jobs, refit=True)
    rs.fit(X[tr], y[tr], groups=groups[tr])
    best = rs.best_estimator_
    proba = best.predict_proba(X[te])[:, 1]
    yhat = (proba >= 0.5).astype(int)
    try:
        aucv = float(roc_auc_score(y[te], proba))
    except Exception:
        aucv = np.nan
    cm = confusion_matrix(y[te], yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
    param_row = {"model": model_name, "fold": fold,
                 "n_iter": int(len(rs.cv_results_["params"])),
                 "best_score_inner": float(rs.best_score_), **rs.best_params_}
    if model_name in c.USE_PCA_FOR:
        param_row["pca_n_components"] = int(c.PCA_N_COMPONENTS)
    fold_row = {"model": model_name, "fold": fold, "n_test": int(len(te)),
                "n_test_subjects": int(len(np.unique(gte))),
                "accuracy": float(accuracy_score(y[te], yhat)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], yhat)),
                "roc_auc": aucv, "tn": float(tn), "fp": float(fp),
                "fn": float(fn), "tp": float(tp)}
    pred_rows = [{"model": model_name, "fold": fold, "subject_id": gte[i],
                  "y_true": int(y[te][i]), "y_proba": float(proba[i]),
                  "y_pred": int(yhat[i])} for i in range(len(te))]
    best_pack = {"params": dict(rs.best_params_), "pca": model_name in c.USE_PCA_FOR}
    return fold_row, pred_rows, param_row, (model_name, fold, best_pack)


# Wipe any stale PCA cache so no cross-run cache hit can corrupt results.
shutil.rmtree(PIPELINE_CACHE_DIR, ignore_errors=True)
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

tasks = [(X_by_model[m], m, f, tr, te)
         for m in X_by_model for f, (tr, te) in enumerate(splits, 1)]
print(f"\n[STEP 1] nested CV: {len(tasks)} tasks across {N_CORES} cores")
t0 = time.time()
results = []
for (X, m, f, tr, te) in tasks:
    ts = time.time()
    results.append(_fit_eval_one_task(X, m, f, tr, te, N_CORES))
    print(f"  {m} fold {f}: {time.time() - ts:.0f}s", flush=True)
print(f"[STEP 1] done in {time.time() - t0:.0f}s")

folds = pd.DataFrame([r[0] for r in results]).sort_values(["model", "fold"]).reset_index(drop=True)
preds = pd.DataFrame([pr for r in results for pr in r[1]]).sort_values(["model", "fold"]).reset_index(drop=True)
params = pd.DataFrame([r[2] for r in results]).sort_values(["model", "fold"]).reset_index(drop=True)
BEST_BY_MODEL_FOLD = {(mn, fd): pack for (_, _, _, (mn, fd, pack)) in results}

folds.to_csv(c.RESULTS_MAIN_PATH, index=False)
preds.to_csv(c.PREDS_MAIN_PATH, index=False)
params.to_csv(c.PARAMS_MAIN_PATH, index=False)

print("\n[STEP 1] per-fold results:")
print(folds[["model", "fold", "balanced_accuracy", "roc_auc", "accuracy"]]
      .to_string(index=False, float_format="%.4f"))


# --------------------------------------------------------------------------- #
# STEP 2 -- permutation test (balanced accuracy + ROC-AUC + accuracy)
# --------------------------------------------------------------------------- #
rec_y, subj_rec_idx_list, epoch_to_rec_idx = kstats.build_recording_shuffle_structures(
    rec_id, y, groups
)


def _make_estimator_cached(model_name, fold):
    pack = BEST_BY_MODEL_FOLD[(model_name, fold)]
    if pack["pca"]:
        est = Pipeline([("pca", PCA(n_components=c.PCA_N_COMPONENTS, svd_solver="randomized",
                                    random_state=c.SEED)),
                        ("rf", RandomForestClassifier(random_state=c.SEED,
                                                      class_weight="balanced_subsample", n_jobs=1))])
    else:
        est = RandomForestClassifier(random_state=c.SEED, class_weight="balanced_subsample", n_jobs=1)
    if pack["params"]:
        est.set_params(**pack["params"])
    return est


def _run_perm_chunk(seed_list):
    out = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        y_perm = kstats.subject_aware_label_shuffle(rec_y, subj_rec_idx_list,
                                                    epoch_to_rec_idx, rng)
        rec = {}
        for model_name, X in X_by_model.items():
            ba, au, ac = [], [], []
            for fold, (tr, te) in enumerate(splits, 1):
                est = _make_estimator_cached(model_name, fold)
                est.fit(X[tr], y_perm[tr])
                proba = est.predict_proba(X[te])[:, 1]
                yhat = (proba >= 0.5).astype(int)
                ba.append(balanced_accuracy_score(y_perm[te], yhat))
                ac.append(accuracy_score(y_perm[te], yhat))
                try:
                    au.append(roc_auc_score(y_perm[te], proba))
                except Exception:
                    au.append(np.nan)
            rec[model_name] = (float(np.mean(ba)), float(np.nanmean(au)), float(np.mean(ac)))
        out.append(rec)
    return out


seeds = [c.SEED + 1000 + i for i in range(c.N_PERM)]
chunks = [ch.tolist() for ch in np.array_split(np.array(seeds, int), N_CORES) if len(ch)]
print(f"\n[STEP 2] permutation: {c.N_PERM} perms x {len(X_by_model)} models x "
      f"{c.OUTER_SPLITS} folds, {N_CORES} cores")
t0 = time.time()
chunked = Parallel(n_jobs=N_CORES, prefer="processes", batch_size=1)(
    delayed(_run_perm_chunk)(ch) for ch in chunks)
perm_results = [r for sub in chunked for r in sub]
print(f"[STEP 2] done in {time.time() - t0:.0f}s ({len(perm_results)} perms)")

null = {m: {"bacc": np.array([r[m][0] for r in perm_results]),
            "auc": np.array([r[m][1] for r in perm_results]),
            "acc": np.array([r[m][2] for r in perm_results])} for m in X_by_model}
obs = {m: {"bacc": float(folds.loc[folds.model == m, "balanced_accuracy"].mean()),
           "auc": float(folds.loc[folds.model == m, "roc_auc"].mean()),
           "acc": float(folds.loc[folds.model == m, "accuracy"].mean())} for m in X_by_model}

rows = []
for m in X_by_model:
    r = {"model": m}
    for met in ("bacc", "auc", "acc"):
        arr = np.asarray(null[m][met])
        arr = arr[~np.isnan(arr)]
        r[f"obs_{met}"] = obs[m][met]
        r[f"null_{met}_mean"] = float(arr.mean())
        r[f"p_perm_{met}"] = kstats.two_sided_p(arr, obs[m][met])
        r[f"p_ttest_{met}"] = float(stats.ttest_1samp(arr, popmean=obs[m][met],
                                                       alternative="two-sided").pvalue)
    rows.append(r)
perm_summary = pd.DataFrame(rows)
perm_summary.to_csv(c.PERM_SUMMARY_MAIN_PATH, index=False)
for m in X_by_model:
    np.save(c.PERM_DIR / f"null_bacc_{m}.npy", null[m]["bacc"])
    np.save(c.PERM_DIR / f"null_auc_{m}.npy", null[m]["auc"])
    np.save(c.PERM_DIR / f"null_acc_{m}.npy", null[m]["acc"])

print("\n[STEP 2] permutation summary (two-sided p):")
print(perm_summary[["model", "obs_bacc", "p_perm_bacc", "obs_auc", "p_perm_auc",
                    "obs_acc", "p_perm_acc"]].to_string(index=False, float_format="%.4f"))
print("\nSaved model results, predictions, best params and permutation outputs.")
