"""Shared feature loading / selection for the full-montage A/B/C models.

Both the model-training script and the wPLI edge-importance script build their
design matrices the same way through these helpers, guaranteeing identical
feature selection (and therefore identical splits and feature counts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

KEY_COLS = ["subject_id", "recording_number", "epoch_index_original"]

# Columns that are metadata / quality-control rather than neural features and
# must never enter a model. ``ptp_uv`` (peak-to-peak amplitude) and
# ``n_regions`` are quality-control quantities and are excluded explicitly.
_EXCLUDE_EXACT = {
    "A__extract_ok", "B__extract_ok", "A__sfreq", "B__sfreq",
    "A__n_channels", "B__n_channels", "A__epoch_len_sec", "B__epoch_len_sec",
    "A__n_epochs_before", "B__n_epochs_before", "A__n_epochs_after", "B__n_epochs_after",
    "B__n_windows", "B__win_len_sec", "B__win_step_sec",
}
_EXCLUDE_CONTAINS = [
    "file_path", "extract_error", "drug", "eyes",
    "subject_id", "recording_number", "epoch_index",
    "ptp_uv", "n_regions",
]


def load_and_merge_AB(
    feat_a_path=config.FEAT_A_PATH, feat_b_path=config.FEAT_B_EDGES_PATH
) -> pd.DataFrame:
    """Load feature sets A and B and merge them at the epoch level.

    Columns are prefixed ``A__`` / ``B__`` (except the merge keys) and a single
    ``drug`` label column is attached. Rows are limited to successfully
    extracted epochs.
    """
    A = pd.read_csv(feat_a_path)
    B = pd.read_csv(feat_b_path)
    A_ok = A[A.get("extract_ok", True) == True].copy()  # noqa: E712
    B_ok = B[B.get("extract_ok", True) == True].copy()  # noqa: E712

    A_keep = A_ok.rename(columns={c: f"A__{c}" for c in A_ok.columns if c not in KEY_COLS})
    B_keep = B_ok.drop(columns=["drug"], errors="ignore").rename(
        columns={c: f"B__{c}" for c in B_ok.columns if c not in KEY_COLS and c != "drug"}
    )
    df = A_keep.merge(B_keep, on=KEY_COLS, how="inner")
    df["drug"] = df["A__drug"]
    return df


def select_feature_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return the numeric neural-feature columns for a given ``A__``/``B__`` prefix."""
    cols = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        if c in _EXCLUDE_EXACT:
            continue
        if any(x in c for x in _EXCLUDE_CONTAINS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_design_matrices(df: pd.DataFrame):
    """Build the A / B / C design matrices and labels from a merged frame.

    Returns ``(X_by_model, y, groups, rec_id, feat_cols)`` where ``X_by_model``
    maps the canonical model name to its feature matrix.
    """
    a_cols = select_feature_cols(df, "A__")
    b_cols = select_feature_cols(df, "B__")
    XA = df[a_cols].to_numpy()
    XB = df[b_cols].to_numpy()
    XAB = np.concatenate([XA, XB], axis=1)

    y = (df["drug"] == "ketamine").astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()
    rec_id = (df["subject_id"].astype(str) + "||" + df["recording_number"].astype(str)).to_numpy()

    X_by_model = {config.MODEL_A: XA, config.MODEL_B: XB, config.MODEL_C: XAB}
    return X_by_model, y, groups, rec_id, {"A": a_cols, "B": b_cols}
