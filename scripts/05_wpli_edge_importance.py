#!/usr/bin/env python3
"""Back-projected wPLI edge importance for the connectivity model (set B).

The connectivity model classifies on PCA components, so feature importance is
not directly interpretable per edge. This script refits PCA + random forest per
outer fold using the selected hyper-parameters, then back-projects the random
forest's PCA-component importances onto the original 5673-edge space via the
squared PCA loadings (squared to avoid sign-flip cancellation across folds) and
averages across folds. The result is a per-edge importance for theta/alpha/beta.

Inputs (tracked derived data -- no raw EEG required):
    data/derived/features/features_A_bandpower_epochwise.csv
    data/derived/features/features_B_wpli_edges_epoch_sliding.csv
    data/derived/montage_info.json                 (channel names)
    results/models/best_params_rf_nested_groupkfold_Aepoch_BwpliEpochSliding.csv
Output:
    results/tables/feat_B_edge_importance_table.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402
from ketamine_eeg import features as feat  # noqa: E402

c.ensure_dirs()

N_CORES = os.cpu_count() or 8
EDGE_TABLE_PATH = c.TABLES_DIR / "feat_B_edge_importance_table.csv"

# --------------------------------------------------------------------------- #
# Design matrix (identical selection to the model-training script)
# --------------------------------------------------------------------------- #
df = feat.load_and_merge_AB()
B_feat_cols = feat.select_feature_cols(df, "B__")
XB = df[B_feat_cols].to_numpy()
y = (df["drug"] == "ketamine").astype(int).to_numpy()
groups = df["subject_id"].astype(str).to_numpy()
bare_feat_names = [col.replace("B__", "", 1) for col in B_feat_cols]
assert len(B_feat_cols) == c.WPLI_BANDS.__len__() * c.N_EDGES_PER_BAND == 5673

CH_NAMES = json.loads(c.MONTAGE_INFO_PATH.read_text())["ch_names"]

# --------------------------------------------------------------------------- #
# Saved hyper-parameters for model B
# --------------------------------------------------------------------------- #
params_df = pd.read_csv(c.PARAMS_MAIN_PATH)
B_params = params_df[params_df["model"] == c.MODEL_B].copy()


def _coerce(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _coerce_max_features(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str) and v.strip() in {"sqrt", "log2"}:
        return v.strip()
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except Exception:
        return v


def build_pipeline_from_row(row: pd.Series) -> Pipeline:
    pca_n = int(row.get("pca_n_components", c.PCA_N_COMPONENTS))
    rf_params = {
        "n_estimators": _coerce(row.get("rf__n_estimators")),
        "max_depth": _coerce(row.get("rf__max_depth")),
        "min_samples_split": _coerce(row.get("rf__min_samples_split")),
        "min_samples_leaf": _coerce(row.get("rf__min_samples_leaf")),
        "max_features": _coerce_max_features(row.get("rf__max_features")),
    }
    rf_params = {k: v for k, v in rf_params.items() if v is not None}
    return Pipeline([
        ("pca", PCA(n_components=pca_n, svd_solver="randomized", random_state=c.SEED)),
        ("rf", RandomForestClassifier(random_state=c.SEED, class_weight="balanced_subsample",
                                      n_jobs=1, **rf_params)),
    ])


# --------------------------------------------------------------------------- #
# Refit per outer fold and collect PCA loadings + RF importances
# --------------------------------------------------------------------------- #
splits = list(GroupKFold(n_splits=c.OUTER_SPLITS).split(XB, y, groups))


def fit_one_fold(fold_idx: int, tr: np.ndarray) -> dict:
    row = B_params[B_params["fold"] == fold_idx].iloc[0]
    pipe = build_pipeline_from_row(row)
    pipe.fit(XB[tr], y[tr])
    pca: PCA = pipe.named_steps["pca"]
    rf: RandomForestClassifier = pipe.named_steps["rf"]
    return {"fold": fold_idx, "components": pca.components_.copy(),
            "rf_importances": rf.feature_importances_.copy()}


fold_results = Parallel(n_jobs=min(N_CORES, c.OUTER_SPLITS), prefer="processes")(
    delayed(fit_one_fold)(fold_idx, tr) for fold_idx, (tr, _) in enumerate(splits, 1))
fold_results.sort(key=lambda d: d["fold"])
print(f"Fitted {len(fold_results)} folds; components {fold_results[0]['components'].shape}")

# Back-project: importance_j = sum_k rf_importance_k * loading_{k,j}^2 ; average over folds
backproj_per_fold = np.stack([(fr["components"] ** 2).T @ fr["rf_importances"]
                              for fr in fold_results])
importance_mean = backproj_per_fold.mean(axis=0)


def band_importance(band: str) -> np.ndarray:
    idxs = [i for i, n in enumerate(bare_feat_names) if n.startswith(band + "_")]
    assert len(idxs) == c.N_EDGES_PER_BAND, f"{band}: {len(idxs)} != {c.N_EDGES_PER_BAND}"
    return importance_mean[idxs]


triu_rows, triu_cols = np.triu_indices(c.N_CHANNELS, k=1)
records = []
for band in c.WPLI_BANDS:
    imp_vec = band_importance(band)
    for edge_idx, (ri, ci) in enumerate(zip(triu_rows, triu_cols)):
        records.append({"band": band, "edge": edge_idx, "ch_a": int(ri), "ch_b": int(ci),
                        "ch_a_name": CH_NAMES[ri], "ch_b_name": CH_NAMES[ci],
                        "importance": float(imp_vec[edge_idx])})
edge_df = pd.DataFrame(records)
edge_df.to_csv(EDGE_TABLE_PATH, index=False)
print(f"Saved {len(edge_df)} edge rows -> {EDGE_TABLE_PATH}")
print("\nTop 10 edges by back-projected importance:")
print(edge_df.nlargest(10, "importance")[["band", "ch_a_name", "ch_b_name", "importance"]]
      .to_string(index=False, float_format="%.3e"))
