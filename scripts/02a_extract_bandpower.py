"""Extract per-epoch log-bandpower features (Feature set A) from eyes-closed EEG.

For every eyes-closed recording listed in the validated manifest this script:

  1. loads the epoched ``.set`` recording,
  2. rejects epochs whose peak-to-peak amplitude exceeds a threshold,
  3. computes a Welch PSD over each kept epoch,
  4. averages power within the delta/theta/alpha/beta bands over the whole head
     and per scalp region, applies a natural-log transform, and
  5. writes one row per kept epoch (plus recording-level metadata).

Inputs:
    - data/derived/manifests/manifest_spontaneous_validated.csv
    - raw .set files referenced by the manifest's ``file_path`` column

Outputs:
    - data/derived/features/features_A_bandpower_epochwise.csv
    - data/derived/features/features_A_bandpower_epochwise_meta.json

Requires the raw EEG recordings (not distributed; see README).

Run with::

    python scripts/02a_extract_bandpower.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

# Make the in-repo package importable and pull paths / constants from config.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c

# Keep MNE quiet; per-call verbosity is still set on the heavy operations below.
mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
# Helpers for QC and epoch-level feature extraction
# --------------------------------------------------------------------------- #
def pick_region_indices(ch_names: list[str]) -> dict[str, list[int]]:
    """Group channel indices into scalp regions by 10-20-style name prefix.

    Works for standard names like Fp1, F3, Cz, Pz, O1, T7, etc. Empty regions
    are dropped to avoid NaNs downstream.
    """
    regions = {"frontal": [], "central": [], "parietal": [], "occipital": [], "temporal": []}

    for i, ch in enumerate(ch_names):
        name = ch.upper()

        # Exclude non-EEG channels if present
        if any(x in name for x in ["EOG", "ECG", "EMG", "AUX", "TRIG", "STI"]):
            continue

        if name.startswith(("FP", "AF", "F")):
            regions["frontal"].append(i)
        elif name.startswith(("C",)):
            regions["central"].append(i)
        elif name.startswith(("P",)):
            regions["parietal"].append(i)
        elif name.startswith(("O",)):
            regions["occipital"].append(i)
        elif name.startswith(("T",)):
            regions["temporal"].append(i)

    # Remove empty regions to avoid NaNs later
    regions = {k: v for k, v in regions.items() if len(v) > 0}
    return regions


def peak_to_peak_uv(epoch_data_volts: np.ndarray) -> float:
    """Max peak-to-peak amplitude across channels for one epoch, in microvolts.

    ``epoch_data_volts`` has shape (n_channels, n_times) and is in volts.
    """
    ptp_per_ch = np.ptp(epoch_data_volts, axis=1)  # volts
    return float(np.max(ptp_per_ch) * 1e6)


def compute_epoch_ptp_uv(epochs: mne.Epochs) -> np.ndarray:
    """Per-epoch peak-to-peak amplitude (max across channels), in microvolts.

    Returns an array of shape (n_epochs,).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times), volts
    ptps = np.zeros(data.shape[0], dtype=float)
    for e in range(data.shape[0]):
        ptps[e] = peak_to_peak_uv(data[e])
    return ptps


def bandpower_epochwise(
    epochs: mne.Epochs,
    bands_hz: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Compute per-epoch, per-channel bandpower.

    Returns a dict mapping band name -> array of shape (n_epochs, n_channels).
    Uses a Welch PSD over the full epoch duration (n_fft = n_times).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times), volts
    sfreq = float(epochs.info["sfreq"])
    n_times = data.shape[-1]

    # NOTE: the Welch fmin/fmax span the requested bands (1-30 Hz here), derived
    # from ``bands_hz`` rather than c.PSD_FMIN/c.PSD_FMAX (1.0/40.0). The narrower
    # band-derived range is preserved exactly to keep numerical behaviour identical
    # to the original notebook.
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=min(b[0] for b in bands_hz.values()),
        fmax=max(b[1] for b in bands_hz.values()),
        n_fft=n_times,
        n_overlap=0,
        verbose="ERROR",
    )

    out: dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in bands_hz.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(idx) == 0:
            raise RuntimeError(f"No frequency bins for band {band_name}")

        bp = np.mean(psds[..., idx], axis=-1)  # (n_epochs, n_channels)
        out[band_name] = bp

    return out


def summarize_bandpower_features_epochwise(
    epochs: mne.Epochs,
    bands_hz: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Build epoch-level features (one row per epoch).

    Produces, per band:
      - log bandpower averaged across all channels (global), and
      - log bandpower averaged across the channels in each region.
    """
    regions = pick_region_indices(epochs.ch_names)
    bp = bandpower_epochwise(epochs, bands_hz)

    n_epochs = len(epochs)
    out = pd.DataFrame({"epoch_index_within_clean": np.arange(n_epochs, dtype=int)})

    # For each band: compute per-epoch channel-mean log-power
    for band, bp_ec in bp.items():  # (n_epochs, n_channels)
        bp_ec = np.maximum(bp_ec, 1e-20)

        # Global per epoch: mean across channels
        out[f"logbp_{band}_global"] = np.log(bp_ec).mean(axis=1)

        # Regions per epoch: mean across channels in region
        for region, idxs in regions.items():
            out[f"logbp_{band}_{region}"] = np.log(bp_ec[:, idxs]).mean(axis=1)

    # Region availability metadata
    out["n_regions"] = float(len(regions))
    return out


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    c.ensure_dirs()

    print("Manifest:", c.MANIFEST_PATH)
    print("Output:", c.FEAT_A_PATH)

    # ------------------------------------------------------------------ #
    # 1. Load manifest and select eyes-closed-only rows
    # ------------------------------------------------------------------ #
    manifest = pd.read_csv(c.MANIFEST_PATH)

    required_cols = ["subject_id", "file_path", "eyes", "drug", "recording_number"]
    missing = [col for col in required_cols if col not in manifest.columns]
    assert len(missing) == 0, f"Manifest missing columns: {missing}"

    df = manifest[manifest["eyes"] == c.EYES_KEEP].copy()
    df = df.sort_values(["subject_id", "recording_number"]).reset_index(drop=True)

    print("Rows (recordings) in EC-only subset:", len(df))
    print("Subjects in EC-only subset:", df["subject_id"].nunique())
    print("Counts by drug:")
    print(df.groupby("drug").size().rename("n_recordings"))

    # ------------------------------------------------------------------ #
    # 2. Extract Feature Set A per epoch for all EC-only recordings
    #    For each file: load epochs, reject by peak-to-peak amplitude,
    #    compute bandpower for kept epochs, store one row per kept epoch
    #    plus the metadata needed for later analysis and debugging.
    # ------------------------------------------------------------------ #
    rows = []

    for _, r in df.iterrows():
        sid = str(r["subject_id"])
        fp = r["file_path"]
        drug = r["drug"]
        eyes = r["eyes"]
        recnum = int(r["recording_number"])

        try:
            # Manifest paths are relative to the repo root; resolve to absolute.
            raw_path = c.resolve_raw_path(fp)
            epochs = mne.io.read_epochs_eeglab(str(raw_path), verbose="ERROR")
            epochs.load_data()  # need data in memory for QC and PSD

            sfreq = float(epochs.info["sfreq"])
            n_epochs_before = len(epochs)
            n_channels = int(epochs.info["nchan"])
            epoch_len_sec = epochs.get_data().shape[-1] / sfreq

            # Compute per-epoch ptp and keep mask
            ptp_uv = compute_epoch_ptp_uv(epochs)  # (n_epochs_before,)
            keep_mask = ptp_uv <= c.REJECT_PTP_UV

            # If everything got rejected, fail fast for this file
            if int(keep_mask.sum()) == 0:
                raise RuntimeError("All epochs rejected by peak-to-peak threshold")

            # Keep only good epochs
            good_epoch_indices = np.where(keep_mask)[0].astype(int)  # indices in ORIGINAL epochs
            epochs_clean = epochs[keep_mask]
            n_epochs_after = len(epochs_clean)

            # Epoch-level bandpower features for kept epochs
            feat_df = summarize_bandpower_features_epochwise(epochs_clean, c.BANDS_HZ)

            # Attach original epoch indices + per-epoch ptp
            feat_df["epoch_index_original"] = good_epoch_indices
            feat_df["ptp_uv"] = ptp_uv[good_epoch_indices]

            # Attach recording-level metadata (repeated for each epoch row)
            feat_df["subject_id"] = sid
            feat_df["drug"] = drug
            feat_df["eyes"] = eyes
            feat_df["recording_number"] = recnum
            feat_df["file_path"] = fp
            feat_df["sfreq"] = sfreq
            feat_df["n_channels"] = n_channels
            feat_df["epoch_len_sec"] = epoch_len_sec
            feat_df["n_epochs_before"] = n_epochs_before
            feat_df["n_epochs_after"] = n_epochs_after
            feat_df["extract_ok"] = True
            feat_df["extract_error"] = ""

            rows.append(feat_df)

        except Exception as e:
            # On failure, add a single row indicating error at recording-level
            rows.append(pd.DataFrame([{
                "subject_id": sid,
                "drug": drug,
                "eyes": eyes,
                "recording_number": recnum,
                "file_path": fp,
                "sfreq": np.nan,
                "n_channels": np.nan,
                "epoch_len_sec": np.nan,
                "n_epochs_before": np.nan,
                "n_epochs_after": np.nan,
                "epoch_index_original": np.nan,
                "epoch_index_within_clean": np.nan,
                "ptp_uv": np.nan,
                "extract_ok": False,
                "extract_error": str(e),
            }]))

    features_A_epoch = pd.concat(rows, ignore_index=True)

    print(
        "Extraction failures (recordings):",
        features_A_epoch.loc[~features_A_epoch["extract_ok"], "file_path"].nunique(),
    )
    print("Rows total (epochs + failure rows):", len(features_A_epoch))

    # ------------------------------------------------------------------ #
    # 3. QC summary for epoch-level Feature Set A
    # ------------------------------------------------------------------ #
    ok = features_A_epoch[features_A_epoch["extract_ok"]].copy()

    print("Epoch-rows ok:", len(ok))
    print("Subjects ok:", ok["subject_id"].nunique())
    print(
        "Recordings ok:",
        ok[["subject_id", "recording_number", "file_path"]].drop_duplicates().shape[0],
    )

    print("Epoch rows per drug:")
    print(ok.groupby("drug").size().rename("n_epochs"))

    print("Epoch ptp summary (kept epochs):")
    print(ok["ptp_uv"].describe())

    print("Sampling rates:")
    print(ok.groupby("sfreq").size().rename("n_epochs"))

    print("Channel counts:")
    print(ok.groupby("n_channels").size().rename("n_epochs"))

    # Rejection rate per recording (computed from repeated metadata)
    rec_qc = (
        ok[["subject_id", "drug", "recording_number", "file_path", "n_epochs_before", "n_epochs_after"]]
        .drop_duplicates()
        .copy()
    )
    rec_qc["epoch_drop_frac"] = 1.0 - (rec_qc["n_epochs_after"] / rec_qc["n_epochs_before"])
    print("Epoch drop fraction per recording summary:")
    print(rec_qc["epoch_drop_frac"].describe())

    # ------------------------------------------------------------------ #
    # 4. Save epoch-level Feature Set A
    # ------------------------------------------------------------------ #
    # Put identifiers/labels first for readability
    id_cols = ["subject_id", "drug", "eyes", "recording_number", "epoch_index_original", "epoch_index_within_clean"]
    meta_cols = ["sfreq", "n_channels", "epoch_len_sec", "ptp_uv", "n_epochs_before", "n_epochs_after", "file_path", "extract_ok", "extract_error"]
    feat_cols = [col for col in features_A_epoch.columns if col.startswith("logbp_") or col == "n_regions"]

    ordered = [col for col in id_cols + meta_cols + feat_cols if col in features_A_epoch.columns] + \
              [col for col in features_A_epoch.columns if col not in (id_cols + meta_cols + feat_cols)]

    features_A_epoch = features_A_epoch[ordered]
    features_A_epoch.to_csv(c.FEAT_A_PATH, index=False)

    print("Saved:", c.FEAT_A_PATH)
    print("Shape:", features_A_epoch.shape)

    # Save the band definitions used for traceability. File paths are stored
    # relative to the repository root so the metadata is machine-independent.
    meta = {
        "bands_hz": c.BANDS_HZ,
        "reject_ptp_uv": c.REJECT_PTP_UV,
        "eyes_keep": c.EYES_KEEP,
        "feature_file": str(c.FEAT_A_PATH.relative_to(c.PROJECT_ROOT)),
        "unit_of_analysis": "epoch (one row per kept epoch)",
    }
    with open(c.FEAT_A_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved meta:", c.FEAT_A_META_PATH)


if __name__ == "__main__":
    main()
