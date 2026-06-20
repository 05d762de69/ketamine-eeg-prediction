#!/usr/bin/env python3
"""wPLI subject-fingerprinting analysis.

Tests whether the weak wPLI drug-decoding reflects subject "fingerprints"
dominating a weakly transferable drug effect. Two analyses:

  1. Per-edge variance decomposition (between-subject identity vs within-subject
     drug effect; the drug effect split into a cross-subject-shared,
     transferable component and an idiosyncratic one), for all 5673 edges and
     the classifier-weighted top-20 alpha edges.
  2. Subject-ID classifier (10 classes) with condition-aware splits
     (train awake -> test ketamine and vice versa) for wPLI vs bandpower.

Inputs (tracked derived data -- no raw EEG required):
    data/derived/features/features_A_bandpower_epochwise.csv
    data/derived/features/features_B_wpli_edges_epoch_sliding.csv
    results/tables/feat_B_edge_importance_table.csv
Outputs (results/tables/):
    fingerprint_edge_variance.csv
    fingerprint_top20_alpha_variance.csv
    fingerprint_subject_id_accuracy.csv
    fingerprint_subject_id_confusion_wpli.csv
    fingerprint_subject_id_confusion_bandpower.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402
from ketamine_eeg.fingerprinting import decompose  # noqa: E402

c.ensure_dirs()
OUT = c.TABLES_DIR
IMP_PATH = c.TABLES_DIR / "feat_B_edge_importance_table.csv"

# --------------------------------------------------------------------------- #
# Load + align A/B
# --------------------------------------------------------------------------- #
A = pd.read_csv(c.FEAT_A_PATH)
B = pd.read_csv(c.FEAT_B_EDGES_PATH)
imp = pd.read_csv(IMP_PATH)
assert (A["extract_ok"] == True).all() and (B["extract_ok"] == True).all()  # noqa: E712

key_cols = ["subject_id", "drug", "recording_number", "epoch_index_original",
            "epoch_index_within_clean"]
A = A.sort_values(key_cols).reset_index(drop=True)
B = B.sort_values(key_cols).reset_index(drop=True)
assert A[key_cols].equals(B[key_cols]), "A/B row alignment broken"

bp_cols = [col for col in A.columns if col.startswith("logbp_")]
edge_cols = ([col for col in B.columns if col.startswith("theta_")]
             + [col for col in B.columns if col.startswith("alpha_")]
             + [col for col in B.columns if col.startswith("beta_")])
assert len(edge_cols) == 5673 and len(bp_cols) == 24

subjects = sorted(B["subject_id"].unique())
n_subj = len(subjects)
n_per_band = c.N_EDGES_PER_BAND
print(f"[load] {len(B)} epochs, {n_subj} subjects, {len(edge_cols)} edges, {len(bp_cols)} bandpower")

# Canonical edge -> channel-pair order from the alpha rows of the importance table
alpha_imp = imp[imp["band"] == "alpha"].sort_values("edge").reset_index(drop=True)
assert len(alpha_imp) == n_per_band
top20 = imp[imp["band"] == "alpha"].nlargest(20, "importance").reset_index(drop=True)
top20_label = [f"{a}-{b}" for a, b in zip(top20["ch_a_name"], top20["ch_b_name"])]

# --------------------------------------------------------------------------- #
# Analysis 1 -- per-edge variance decomposition
# --------------------------------------------------------------------------- #
X_edges = B[edge_cols].to_numpy(dtype=np.float64)
dec = decompose(X_edges, B["subject_id"].to_numpy(), B["drug"].to_numpy(), subjects)

band_of_col = np.array(["theta"] * n_per_band + ["alpha"] * n_per_band + ["beta"] * n_per_band)
edge_df = pd.DataFrame({
    "band": band_of_col,
    "edge_within_band": np.tile(np.arange(n_per_band), 3),
    "ch_a": np.tile(alpha_imp["ch_a_name"].to_numpy(), 3),
    "ch_b": np.tile(alpha_imp["ch_b_name"].to_numpy(), 3),
    **dec,
})
edge_df.to_csv(OUT / "fingerprint_edge_variance.csv", index=False)

# Top-20 alpha edges (classifier-weighted)
top20_global_ix = top20["edge"].to_numpy() + n_per_band  # alpha block offset
top20_df = edge_df.iloc[top20_global_ix].copy()
top20_df["classifier_importance"] = top20["importance"].to_numpy()
top20_df["edge_label"] = top20_label
top20_df.reset_index(drop=True).to_csv(OUT / "fingerprint_top20_alpha_variance.csv", index=False)

ratio = dec["ratio_between_over_within"]
print(f"  all edges: median between/within={np.median(ratio):.2f}  "
      f"median drug_transfer={np.median(dec['drug_transfer']):.3f}")
for b in c.WPLI_BANDS:
    m = band_of_col == b
    print(f"  {b:5s}: median ICC(subj)={np.median(dec['icc_subject'][m]):.3f}  "
          f"median transfer={np.median(dec['drug_transfer'][m]):.3f}")

# --------------------------------------------------------------------------- #
# Analysis 2 -- subject-ID classifier (condition-aware)
# --------------------------------------------------------------------------- #
y_subj = B["subject_id"].to_numpy()
drug = B["drug"].to_numpy()


def cond_aware_subject_id(X, name):
    folds, confs = [], np.zeros((n_subj, n_subj), dtype=np.int64)
    for train_cond, test_cond in (("awake", "ketamine"), ("ketamine", "awake")):
        tr, te = (drug == train_cond), (drug == test_cond)
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                     class_weight="balanced", random_state=c.SEED, n_jobs=-1)
        clf.fit(X[tr], y_subj[tr])
        pred = clf.predict(X[te])
        bacc = balanced_accuracy_score(y_subj[te], pred)
        folds.append({"feature_set": name, "train_cond": train_cond, "test_cond": test_cond,
                      "n_train": int(tr.sum()), "n_test": int(te.sum()), "balanced_acc": bacc})
        confs += confusion_matrix(y_subj[te], pred, labels=subjects)
        print(f"  {name:10s} train={train_cond:8s} test={test_cond:8s} bacc={bacc:.3f} (chance=0.100)")
    return folds, confs


wpli_folds, wpli_cm = cond_aware_subject_id(X_edges, "wPLI")
bp_folds, bp_cm = cond_aware_subject_id(A[bp_cols].to_numpy(dtype=np.float64), "bandpower")

pd.DataFrame(wpli_folds + bp_folds).to_csv(OUT / "fingerprint_subject_id_accuracy.csv", index=False)
pd.DataFrame(wpli_cm, index=subjects, columns=subjects).to_csv(OUT / "fingerprint_subject_id_confusion_wpli.csv")
pd.DataFrame(bp_cm, index=subjects, columns=subjects).to_csv(OUT / "fingerprint_subject_id_confusion_bandpower.csv")
print("\nSaved fingerprinting tables to results/tables/.")
