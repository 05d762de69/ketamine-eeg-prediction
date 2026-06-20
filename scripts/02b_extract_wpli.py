"""Extract weighted phase-lag index (wPLI) connectivity features (Feature set B).

For every eyes-closed recording in the validated manifest, this stage computes
per-epoch wPLI connectivity for the theta/alpha/beta bands. wPLI requires
multiple observations to estimate the cross-spectrum, so within each 8 s epoch
we cut overlapping sliding windows (1 s windows, 0.5 s step ~ 50% overlap) and
treat those windows as the "epochs" handed to
``mne_connectivity.spectral_connectivity_epochs``. This yields one dense
62x62 wPLI matrix per (recording, epoch, band). Each matrix is vectorised over
its upper triangle (no diagonal) to produce the edge-vector features used for
machine learning.

Inputs:
    - data/derived/manifests/manifest_spontaneous_validated.csv
    - raw .set files referenced by the manifest's ``file_path`` column

Outputs:
    - data/derived/features/features_B_wpli_edges_epoch_sliding.csv
        per-epoch edge-vector features (one row per kept epoch, plus failure rows)
    - data/derived/features/features_B_wpli_edges_epoch_sliding_meta.json
        run metadata (method, bands, window settings, output paths)
    - data/derived/features/features_B_wpli_matrices_epoch_sliding.npz
        dense per-epoch wPLI matrices, keyed by recording/epoch/band

Requires the raw EEG recordings (not distributed; see README); this stage is
computationally heavy.

Run with::

    python scripts/02b_extract_wpli.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed

# Make the project package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c  # noqa: E402

try:
    from mne_connectivity import spectral_connectivity_epochs
    CONNECTIVITY_BACKEND = "mne_connectivity"
except Exception:
    from mne.connectivity import spectral_connectivity_epochs
    CONNECTIVITY_BACKEND = "mne.connectivity"

mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# wPLI is computed for theta/alpha/beta only (c.WPLI_BANDS); band ranges come
# from c.BANDS_HZ. These match the original notebook values:
#   theta (4.0, 8.0), alpha (8.0, 13.0), beta (13.0, 30.0).
CONNECTIVITY_BANDS_HZ = {band: c.BANDS_HZ[band] for band in c.WPLI_BANDS}

REJECT_PTP_UV = c.REJECT_PTP_UV          # 250.0 uV peak-to-peak rejection
EYES_KEEP = c.EYES_KEEP                  # "closed"
CONNECTIVITY_MODE = "multitaper"         # spectral estimation mode (source value)
WPLI_METHOD = "wpli"                     # connectivity method (source value)

# Parallelism: recordings are processed independently, so n_jobs caps at len(df).
# Higher values are accepted but ignored (extra workers stay idle).
N_WORKERS = 20                           # not in config; kept from source

# Sliding window settings inside each epoch (8s epochs).
# Use 1s windows with 50% overlap: 8s -> 15 windows (approx).
# Not in config; kept exactly as in the source notebook.
WIN_LEN_SEC = 1.0
WIN_STEP_SEC = 0.5


# --------------------------------------------------------------------------- #
# Helpers: epoch QC, connectivity conversion, vectorisation, sliding windows
# --------------------------------------------------------------------------- #
def peak_to_peak_uv(epoch_data_volts: np.ndarray) -> float:
    """Largest single-channel peak-to-peak amplitude in microvolts."""
    ptp_per_ch = np.ptp(epoch_data_volts, axis=1)  # volts
    return float(np.max(ptp_per_ch) * 1e6)


def compute_epoch_ptp_uv(epochs: mne.Epochs) -> np.ndarray:
    """Per-epoch peak-to-peak amplitude (uV) across the worst channel."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    out = np.zeros(data.shape[0], dtype=float)
    for e in range(data.shape[0]):
        out[e] = peak_to_peak_uv(data[e])
    return out


def reject_epochs_by_ptp(
    epochs: mne.Epochs, reject_ptp_uv: float
) -> tuple[mne.Epochs, np.ndarray, np.ndarray]:
    """Drop epochs exceeding a peak-to-peak amplitude threshold.

    Returns:
      epochs_clean
      keep_mask (length n_epochs_original)
      ptp_uv_original (length n_epochs_original)
    """
    ptp_uv = compute_epoch_ptp_uv(epochs)
    keep = ptp_uv <= reject_ptp_uv
    return epochs[keep], keep, ptp_uv


def connectivity_to_dense_square(con, n_ch: int) -> np.ndarray:
    """Convert an mne-connectivity result to a symmetric dense wPLI matrix."""
    try:
        W = con.get_data(output="dense")
    except TypeError:
        W = con.get_data()

    W = np.asarray(W)
    W = np.squeeze(W)

    # If still 3D (freqs), average across last dim
    if W.ndim == 3:
        W = W.mean(axis=-1)

    if W.ndim != 2 or W.shape != (n_ch, n_ch):
        raise RuntimeError(f"Unexpected wPLI dense shape: {W.shape}, expected {(n_ch, n_ch)}")

    W = np.maximum(W, 0.0)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def upper_triangle_edges(W: np.ndarray) -> np.ndarray:
    """Vectorise the upper triangle (excluding the diagonal) of a matrix."""
    n = W.shape[0]
    tri = np.triu_indices(n, k=1)
    return W[tri]


def epoch_to_sliding_windows_epochs(
    epoch_data: np.ndarray,
    info: mne.Info,
    sfreq: float,
    win_len_sec: float,
    win_step_sec: float,
) -> mne.EpochsArray:
    """Cut one epoch into overlapping sliding windows.

    epoch_data: shape (n_channels, n_times) for ONE epoch (volts)
    Returns an EpochsArray where each epoch is one sliding window segment.
    Shape will be (n_windows, n_channels, n_win_times)
    """
    n_ch, n_times = epoch_data.shape
    win_len = int(round(win_len_sec * sfreq))
    step = int(round(win_step_sec * sfreq))

    if win_len <= 0 or step <= 0:
        raise ValueError("win_len_sec and win_step_sec must be > 0")

    if win_len > n_times:
        raise RuntimeError(f"Window length ({win_len} samples) > epoch length ({n_times} samples)")

    starts = np.arange(0, n_times - win_len + 1, step, dtype=int)
    n_win = len(starts)

    if n_win < 3:
        # wPLI across too-few windows will be unstable / degenerate
        raise RuntimeError(f"Too few windows ({n_win}) for wPLI. Increase epoch length or decrease win_len/step.")

    data = np.zeros((n_win, n_ch, win_len), dtype=float)
    for i, s in enumerate(starts):
        data[i] = epoch_data[:, s:s + win_len]

    # Use dummy events
    events = np.c_[np.arange(n_win), np.zeros(n_win, dtype=int), np.ones(n_win, dtype=int)]
    return mne.EpochsArray(data, info=info, events=events, tmin=0.0, verbose="ERROR")


# --------------------------------------------------------------------------- #
# Per-recording worker
# --------------------------------------------------------------------------- #
def _process_recording(r_dict: dict) -> tuple[list, dict]:
    """Process one recording: load, QC, compute wPLI for every kept epoch.

    Returns (rows, mats) where rows is a list of feat_row dicts and
    mats is a dict {key -> 2D wPLI matrix}.
    """
    sid = str(r_dict["subject_id"])
    fp = r_dict["file_path"]
    drug = str(r_dict["drug"])
    eyes = str(r_dict["eyes"])
    recnum = int(r_dict["recording_number"])

    rec_key = f"{sid}__rec{recnum}"
    rows_local = []
    mats_local = {}

    try:
        # Manifest file_path may be relative to the repo root; resolve it.
        raw_path = c.resolve_raw_path(fp)
        epochs = mne.io.read_epochs_eeglab(raw_path, verbose="ERROR")
        epochs.load_data()

        sfreq = float(epochs.info["sfreq"])
        n_epochs_before = len(epochs)
        # n_ch comes from the recording itself (expected to equal c.N_CHANNELS = 62).
        n_ch = int(epochs.info["nchan"])
        epoch_len_sec = epochs.get_data().shape[-1] / sfreq

        epochs_clean, keep_mask, ptp_uv_all = reject_epochs_by_ptp(epochs, REJECT_PTP_UV)
        n_epochs_after = len(epochs_clean)

        if n_epochs_after == 0:
            raise RuntimeError("All epochs rejected by peak-to-peak threshold")

        kept_original_idx = np.where(keep_mask)[0].astype(int)
        data_clean = epochs_clean.get_data()  # (n_epochs_after, n_ch, n_times)

        for i_clean in range(n_epochs_after):
            i_orig = int(kept_original_idx[i_clean])
            epoch_data = data_clean[i_clean]  # (n_ch, n_times)

            win_epochs = epoch_to_sliding_windows_epochs(
                epoch_data=epoch_data,
                info=epochs_clean.info,
                sfreq=sfreq,
                win_len_sec=WIN_LEN_SEC,
                win_step_sec=WIN_STEP_SEC,
            )

            feat_row = {
                "subject_id": sid,
                "drug": drug,
                "eyes": eyes,
                "recording_number": recnum,
                "epoch_index_original": i_orig,
                "epoch_index_within_clean": int(i_clean),
                "file_path": fp,
                "sfreq": sfreq,
                "n_channels": n_ch,
                "epoch_len_sec": epoch_len_sec,
                "ptp_uv": float(ptp_uv_all[i_orig]),
                "n_epochs_before": n_epochs_before,
                "n_epochs_after": n_epochs_after,
                "n_windows": float(len(win_epochs)),
                "win_len_sec": float(WIN_LEN_SEC),
                "win_step_sec": float(WIN_STEP_SEC),
                "extract_ok": True,
                "extract_error": "",
            }

            for band, (fmin, fmax) in CONNECTIVITY_BANDS_HZ.items():
                con = spectral_connectivity_epochs(
                    win_epochs,
                    method=WPLI_METHOD,
                    mode=CONNECTIVITY_MODE,
                    sfreq=sfreq,
                    fmin=float(fmin),
                    fmax=float(fmax),
                    faverage=True,
                    n_jobs=1,  # outer joblib already parallelizes across recordings
                    verbose="ERROR",
                )
                W = connectivity_to_dense_square(con, n_ch=n_ch)

                mats_local[f"{rec_key}__e{i_orig:04d}__{band}"] = W

                edges = upper_triangle_edges(W)
                for j, val in enumerate(edges):
                    feat_row[f"{band}_e{j:04d}"] = float(val)

            rows_local.append(feat_row)

    except Exception as e:
        rows_local.append({
            "subject_id": sid,
            "drug": drug,
            "eyes": eyes,
            "recording_number": recnum,
            "epoch_index_original": np.nan,
            "epoch_index_within_clean": np.nan,
            "file_path": fp,
            "sfreq": np.nan,
            "n_channels": np.nan,
            "epoch_len_sec": np.nan,
            "ptp_uv": np.nan,
            "n_epochs_before": np.nan,
            "n_epochs_after": np.nan,
            "n_windows": np.nan,
            "win_len_sec": float(WIN_LEN_SEC),
            "win_step_sec": float(WIN_STEP_SEC),
            "extract_ok": False,
            "extract_error": str(e),
        })

    return rows_local, mats_local


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    c.ensure_dirs()

    print("Connectivity backend:", CONNECTIVITY_BACKEND)
    print("Manifest:", c.MANIFEST_PATH)
    print("Edges CSV:", c.FEAT_B_EDGES_PATH)
    print("Matrices NPZ:", c.FEAT_B_MATRICES_PATH)
    print("Sliding window:", WIN_LEN_SEC, "sec, step", WIN_STEP_SEC, "sec")

    # ------------------------------------------------------------------ #
    # Load manifest and filter eyes-closed
    # ------------------------------------------------------------------ #
    manifest = pd.read_csv(c.MANIFEST_PATH)

    required_cols = ["subject_id", "file_path", "eyes", "drug", "recording_number"]
    missing = [col for col in required_cols if col not in manifest.columns]
    assert len(missing) == 0, f"Manifest missing columns: {missing}"

    df = manifest[manifest["eyes"] == EYES_KEEP].copy()
    df = df.sort_values(["subject_id", "recording_number"]).reset_index(drop=True)

    print("Recordings in EC-only subset:", len(df))
    print("Subjects:", df["subject_id"].nunique())
    print(df.groupby("drug").size().rename("n_recordings"))

    # ------------------------------------------------------------------ #
    # Compute wPLI per epoch via sliding windows (parallel across recordings)
    # ------------------------------------------------------------------ #
    # Convert to plain dicts so pandas Series machinery doesn't get pickled.
    recording_dicts = [r.to_dict() for _, r in df.iterrows()]
    n_jobs = min(N_WORKERS, len(recording_dicts))
    print(f"Dispatching {len(recording_dicts)} recordings across {n_jobs} workers...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_recording)(rd) for rd in recording_dicts
    )

    rows = []
    mat_store = {}
    for rows_local, mats_local in results:
        rows.extend(rows_local)
        mat_store.update(mats_local)

    features_B_edges_epoch = pd.DataFrame(rows)

    print("Failure rows:", int((~features_B_edges_epoch["extract_ok"]).sum()))
    print("Total rows (epochs + failure rows):", len(features_B_edges_epoch))

    # ------------------------------------------------------------------ #
    # Save features + matrices + meta
    # ------------------------------------------------------------------ #
    features_B_edges_epoch.to_csv(c.FEAT_B_EDGES_PATH, index=False)
    print("Saved edges CSV:", c.FEAT_B_EDGES_PATH)
    print("Shape:", features_B_edges_epoch.shape)

    np.savez_compressed(
        c.FEAT_B_MATRICES_PATH,
        **{k: v.astype(np.float32) for k, v in mat_store.items()},
    )
    print("Saved matrices NPZ:", c.FEAT_B_MATRICES_PATH)
    print("Matrices stored:", len(mat_store))

    # Store output paths relative to the repository root (never absolute / author paths).
    meta = {
        "connectivity_backend": CONNECTIVITY_BACKEND,
        "method": WPLI_METHOD,
        "mode": CONNECTIVITY_MODE,
        "bands_hz": CONNECTIVITY_BANDS_HZ,
        "reject_ptp_uv": REJECT_PTP_UV,
        "eyes_keep": EYES_KEEP,
        "unit_of_analysis": "epoch (wPLI computed across sliding windows within each epoch)",
        "edge_vectorization": "upper triangle, no diagonal",
        "win_len_sec": WIN_LEN_SEC,
        "win_step_sec": WIN_STEP_SEC,
        "edges_csv": str(c.FEAT_B_EDGES_PATH.relative_to(c.PROJECT_ROOT)),
        "matrices_npz": str(c.FEAT_B_MATRICES_PATH.relative_to(c.PROJECT_ROOT)),
    }
    with open(c.FEAT_B_EDGES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta:", c.FEAT_B_EDGES_META_PATH)

    # ------------------------------------------------------------------ #
    # Diagnostic
    # ------------------------------------------------------------------ #
    ok = features_B_edges_epoch[features_B_edges_epoch["extract_ok"] == True].copy()  # noqa: E712

    edge_cols = [
        col for col in ok.columns
        if col.startswith("theta_e") or col.startswith("alpha_e") or col.startswith("beta_e")
    ]
    print("Edge feature columns:", len(edge_cols))

    # Sample a few edge columns to check variance quickly
    sample_cols = edge_cols[:10] if len(edge_cols) >= 10 else edge_cols
    if len(sample_cols) > 0:
        print(ok[sample_cols].describe().loc[["mean", "std", "min", "max"]])
    else:
        print("No edge columns found (unexpected).")


if __name__ == "__main__":
    main()
