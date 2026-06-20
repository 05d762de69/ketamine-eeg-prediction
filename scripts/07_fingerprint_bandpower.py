#!/usr/bin/env python3
"""Bandpower counterpart of the wPLI variance decomposition.

Applies the same per-feature decomposition (see ketamine_eeg.fingerprinting) to
the 24 bandpower features, and writes a wPLI-vs-bandpower transferability
summary. The decomposition function is first re-run on the wPLI edges and
asserted to reproduce fingerprint_edge_variance.csv exactly, proving both
feature classes are computed on an identical code path.

Inputs (tracked derived data -- no raw EEG required):
    data/derived/features/features_A_bandpower_epochwise.csv
    data/derived/features/features_B_wpli_edges_epoch_sliding.csv
    results/tables/fingerprint_edge_variance.csv   (from 06_fingerprint_wpli.py)
Outputs (results/tables/):
    fingerprint_bandpower_variance.csv
    fingerprint_transferability_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402
from ketamine_eeg.fingerprinting import decompose  # noqa: E402

c.ensure_dirs()
OUT = c.TABLES_DIR
KEY_COLS = ["subject_id", "drug", "recording_number", "epoch_index_original",
            "epoch_index_within_clean"]

A = pd.read_csv(c.FEAT_A_PATH).sort_values(KEY_COLS).reset_index(drop=True)
B = pd.read_csv(c.FEAT_B_EDGES_PATH).sort_values(KEY_COLS).reset_index(drop=True)
assert (A["extract_ok"] == True).all() and (B["extract_ok"] == True).all()  # noqa: E712

bp_cols = [col for col in A.columns if col.startswith("logbp_")]
edge_cols = ([col for col in B.columns if col.startswith("theta_")]
             + [col for col in B.columns if col.startswith("alpha_")]
             + [col for col in B.columns if col.startswith("beta_")])
subjects = sorted(B["subject_id"].unique())
print(f"bandpower features: {len(bp_cols)} (4 bands x 6 regions); subjects: {len(subjects)}")

# --- validate the shared decomposition against the saved wPLI table --------- #
wpli_saved = pd.read_csv(OUT / "fingerprint_edge_variance.csv")
wpli_recomp = decompose(B[edge_cols].to_numpy(np.float64),
                        B["subject_id"].to_numpy(), B["drug"].to_numpy(), subjects)
max_drift = max(float(np.nanmax(np.abs(wpli_recomp[k] - wpli_saved[k].to_numpy())))
                for k in wpli_recomp if k in wpli_saved.columns)
assert max_drift < 1e-8, f"decomposition drifted from the saved wPLI table: {max_drift:.2e}"
print(f"zero-drift check vs saved wPLI table: max abs diff = {max_drift:.2e} (OK)")

# --- decompose the bandpower features --------------------------------------- #
bp = decompose(A[bp_cols].to_numpy(np.float64),
               A["subject_id"].to_numpy(), A["drug"].to_numpy(), subjects)
bp_df = pd.DataFrame({"feature": bp_cols,
                      "band": [col.split("_")[1] for col in bp_cols],
                      "region": [col.split("_")[2] for col in bp_cols]})
for k, v in bp.items():
    bp_df[k] = v
bp_df.to_csv(OUT / "fingerprint_bandpower_variance.csv", index=False)


def _summary(df, label):
    return {"feature_set": label,
            "median_drug_transfer": float(df["drug_transfer"].median()),
            "median_ratio_between_over_within": float(df["ratio_between_over_within"].median()),
            "median_ratio_between_over_shared": float(df["ratio_between_over_shared"].median())}


summary_df = pd.DataFrame([_summary(wpli_saved, "wPLI"), _summary(bp_df, "bandpower")])
summary_df.to_csv(OUT / "fingerprint_transferability_summary.csv", index=False)

print("\nmedian drug_transfer (per band, bandpower):")
for b in c.BANDS_HZ:
    m = bp_df["band"] == b
    print(f"  {b:6s}: {bp_df.loc[m, 'drug_transfer'].median():.4f}")
print("\nwPLI vs bandpower transferability summary:")
print(summary_df.to_string(index=False, float_format="%.4f"))
print("\nSaved bandpower decomposition + transferability summary to results/tables/.")
