"""Build the spontaneous-EEG recording manifest.

Discovers the raw EEGLAB ``.set`` recordings, parses subject / date / recording
metadata from each filename with a strict regex, assigns drug labels by the
recording-order rule (recordings 1-2 = awake, 3-4 = ketamine), validates the
per-subject structure (4 files, 2 eyes-closed + 2 eyes-open, both drug states
present for eyes-closed), and writes the full and validated manifests. An
optional structural exploration step (sampling rate / channel / duration /
window-count checks) runs at the end when MNE and the raw recordings are
available.

Inputs:  raw EEG ``.set`` files under data/raw/spontaneous/
Outputs: data/derived/manifests/manifest_spontaneous_all.csv and
         data/derived/manifests/manifest_spontaneous_validated.csv
Note:    Requires the raw EEG recordings (not distributed; see README).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project package importable, then pull paths/constants from it so the
# script contains no hard-coded, machine-specific paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ketamine_eeg import config as c

# Optional: only needed for the light structural exploration at the end.
try:
    import mne
except Exception:
    mne = None

if mne is not None:
    mne.set_log_level("WARNING")


# ----------------------------
# Expected structure
# ----------------------------
EXPECTED_FILES_PER_SUBJECT = 4
EXPECTED_EC_PER_SUBJECT = 2
EXPECTED_EO_PER_SUBJECT = 2

# Manifest output paths (the validated one is the canonical project artifact).
MANIFEST_ALL_PATH = c.MANIFEST_DIR / "manifest_spontaneous_all.csv"
MANIFEST_VALIDATED_PATH = c.MANIFEST_PATH

# Manifest ``file_path``/``parent_dir`` are stored relative to the repository
# root so the manifest is portable across machines.
PARENT_DIR_REL = "data/raw/spontaneous"


# ============================================
# Section 2. File discovery (spontaneous only)
# ============================================

def discover_spontaneous_set_files(spont_path: Path) -> pd.DataFrame:
    """
    Discover EEGLAB .set files in the spontaneous folder.
    Ignores .fdt files by design.

    The ``file_path``/``parent_dir`` columns are stored relative to the
    repository root (e.g. ``data/raw/spontaneous/<filename>``).
    """
    if not spont_path.exists():
        raise FileNotFoundError(f"Spontaneous path not found: {spont_path}")

    set_files = sorted(spont_path.glob("*.set"))

    rows = []
    for p in set_files:
        rows.append({
            "file_path": f"{PARENT_DIR_REL}/{p.name}",
            "file_name": p.name,
            "parent_dir": PARENT_DIR_REL,
        })

    return pd.DataFrame(rows)


# ============================================
# Section 3. Strict filename parsing
# and initial manifest creation (spontaneous)
# ============================================

# Expected filename pattern (normalized):
# <subject>_<YYYYMMDD>_<4digitRec>_<eyesOpen|eyesClosed>_afterICA.set

FNAME_REGEX = re.compile(
    r"^(?P<subject>\d+)_"
    r"(?P<date>\d{8})_"
    r"(?P<rec>\d{4})_"
    r"(?P<eyes>eyesOpen|eyesClosed)_"
    r"afterICA\.set$",
    flags=re.IGNORECASE
)


def parse_from_filename(file_name: str) -> dict:
    """
    Strict parser for normalized spontaneous EEG filenames.

    Returns a dict with parsed fields and a parse_ok flag.
    Files that do not match exactly are marked parse_ok=False
    and should be excluded from the validated manifest.
    """
    m = FNAME_REGEX.match(file_name)
    if not m:
        return {
            "subject_id": "unknown",
            "date_str": None,
            "recording_number": None,
            "eyes": "unknown",
            "parse_ok": False,
            "parse_notes": "regex_failed",
        }

    eyes_raw = m.group("eyes").lower()
    eyes = "open" if eyes_raw == "eyesopen" else "closed"

    return {
        "subject_id": m.group("subject"),
        "date_str": m.group("date"),
        "recording_number": int(m.group("rec")),
        "eyes": eyes,
        "parse_ok": True,
        "parse_notes": "",
    }


def build_initial_manifest(df_files: pd.DataFrame) -> pd.DataFrame:
    """
    Build the initial manifest by parsing filenames only.
    No drug labels, no QC yet.
    """
    rows = []
    for _, r in df_files.iterrows():
        parsed = parse_from_filename(r["file_name"])
        rows.append({
            **parsed,
            "file_path": r["file_path"],
            "file_name": r["file_name"],
            "parent_dir": r["parent_dir"],
        })
    return pd.DataFrame(rows)


# ============================================
# Section 4. Drug assignment (order rule)
# and basic per-subject validation
# ============================================

# ----------------------------
# 4.1 Assign drug label by recording order
# Rule: first two recordings = awake, last two = ketamine
# ----------------------------

def assign_drug_by_order_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign drug labels within each subject based on recording_number order.
    Assumes exactly 4 recordings per subject in the spontaneous dataset.
    """
    df = df.copy()
    df["drug"] = "unknown"
    df["drug_source"] = "order_rule"
    df["drug_order_confidence"] = True

    for sid, g in df.groupby("subject_id", sort=False):
        # Sort strictly by recording number
        g_sorted = g.sort_values("recording_number")
        idx = g_sorted.index.tolist()

        if len(idx) != EXPECTED_FILES_PER_SUBJECT:
            # Leave drug as unknown for this subject
            df.loc[idx, "drug_order_confidence"] = False
            continue

        # Apply rule
        df.loc[idx[:2], "drug"] = "awake"
        df.loc[idx[2:], "drug"] = "ketamine"

    return df


# ----------------------------
# 4.2 Per-subject validation for minimal EC-only analysis
# ----------------------------

def validate_subject_group(g: pd.DataFrame) -> dict:
    notes = []

    n_files = len(g)
    if n_files != EXPECTED_FILES_PER_SUBJECT:
        notes.append(f"expected_{EXPECTED_FILES_PER_SUBJECT}_files_got_{n_files}")

    n_ec = (g["eyes"] == "closed").sum()
    n_eo = (g["eyes"] == "open").sum()

    if n_ec != EXPECTED_EC_PER_SUBJECT or n_eo != EXPECTED_EO_PER_SUBJECT:
        notes.append(f"eyes_counts_ec_{n_ec}_eo_{n_eo}")

    has_awake_ec = ((g["eyes"] == "closed") & (g["drug"] == "awake")).any()
    has_ket_ec = ((g["eyes"] == "closed") & (g["drug"] == "ketamine")).any()

    if not (has_awake_ec and has_ket_ec):
        notes.append("missing_ec_for_one_or_both_drug_states")

    if not g["drug_order_confidence"].all():
        notes.append("drug_order_confidence_false")

    return {
        "subject_id": g["subject_id"].iloc[0],
        "passes_basic_checks": len(notes) == 0,
        "check_notes": ";".join(notes),
        "n_files": n_files,
        "n_ec": int(n_ec),
        "n_eo": int(n_eo),
        "has_awake_ec": bool(has_awake_ec),
        "has_ket_ec": bool(has_ket_ec),
    }


def build_qc_report(manifest: pd.DataFrame) -> pd.DataFrame:
    """Run the per-subject validation over the whole manifest."""
    qc_rows = []
    for sid, g in manifest.groupby("subject_id", sort=False):
        qc_rows.append(validate_subject_group(g))
    return pd.DataFrame(qc_rows)


# ============================================
# Section 5. Data exploration (structural / technical only)
# Spontaneous EEG -- validated subjects
# ============================================

def hash_channel_names(ch_names) -> str:
    joined = "|".join(ch_names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def estimate_n_windows(duration_sec, window_len, step_sec) -> int:
    if duration_sec < window_len:
        return 0
    return int(np.floor((duration_sec - window_len) / step_sec) + 1)


def explore_structure(manifest_validated: pd.DataFrame) -> pd.DataFrame:
    """
    Load one eyes-closed file per validated subject and extract structural
    metadata (sampling rate, channel layout, duration, window-count estimate).

    Handles both continuous Raw and epoched Epochs. Requires MNE and the raw
    recordings; raises if MNE is unavailable.
    """
    assert mne is not None, "mne is required for exploration (read_raw_eeglab)"

    # ------------------------------------------------
    # 5.1 Select ONE file per subject (EC preferred)
    # ------------------------------------------------
    # We explore EC only, because that is the minimal analysis subset.
    manifest_ec = manifest_validated[manifest_validated["eyes"] == "closed"].copy()

    # Take the first EC file per subject (deterministic).
    explore_df = (
        manifest_ec
        .sort_values(["subject_id", "recording_number"])
        .groupby("subject_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    print("Subjects explored:", explore_df["subject_id"].nunique())
    print("Files explored:", len(explore_df))

    # ------------------------------------------------
    # 5.2 Load files and extract structural metadata
    # Handles both continuous Raw and epoched Epochs
    # ------------------------------------------------
    rows = []
    for _, r in explore_df.iterrows():
        # Manifest paths are repo-relative; resolve to an absolute path to load.
        fp = str(c.resolve_raw_path(r["file_path"]))
        sid = r["subject_id"]

        try:
            # Try continuous first
            raw = mne.io.read_raw_eeglab(fp, preload=False, verbose="ERROR")

            sfreq = float(raw.info["sfreq"])
            n_channels = int(raw.info["nchan"])
            duration_sec = raw.n_times / sfreq
            ch_hash = hash_channel_names(raw.ch_names)

            rows.append({
                "subject_id": sid,
                "file_path": r["file_path"],
                "data_type": "raw",
                "sfreq": sfreq,
                "n_channels": n_channels,
                "duration_sec": duration_sec,
                "n_epochs": None,
                "epoch_len_sec": None,
                "channel_hash": ch_hash,
                "load_ok": True,
            })

        except Exception as e_raw:
            # Fallback to epoched data
            try:
                epochs = mne.io.read_epochs_eeglab(fp, verbose="ERROR")

                sfreq = float(epochs.info["sfreq"])
                n_channels = int(epochs.info["nchan"])
                n_epochs = int(len(epochs))
                n_times = int(epochs.get_data().shape[-1])
                epoch_len_sec = n_times / sfreq
                duration_sec = n_epochs * epoch_len_sec  # total samples across epochs
                ch_hash = hash_channel_names(epochs.ch_names)

                rows.append({
                    "subject_id": sid,
                    "file_path": r["file_path"],
                    "data_type": "epochs",
                    "sfreq": sfreq,
                    "n_channels": n_channels,
                    "duration_sec": duration_sec,
                    "n_epochs": n_epochs,
                    "epoch_len_sec": epoch_len_sec,
                    "channel_hash": ch_hash,
                    "load_ok": True,
                    "load_error_raw": str(e_raw),
                })

            except Exception as e_epochs:
                rows.append({
                    "subject_id": sid,
                    "file_path": r["file_path"],
                    "data_type": "unreadable",
                    "sfreq": None,
                    "n_channels": None,
                    "duration_sec": None,
                    "n_epochs": None,
                    "epoch_len_sec": None,
                    "channel_hash": None,
                    "load_ok": False,
                    "load_error_raw": str(e_raw),
                    "load_error_epochs": str(e_epochs),
                })

    exploration = pd.DataFrame(rows)

    print("Load failures:", (~exploration["load_ok"]).sum())
    print("Data types observed:")
    print(exploration.groupby("data_type").size().rename("n_subjects"))

    # ------------------------------------------------
    # 5.3 Sampling rate exploration
    # ------------------------------------------------
    print("Sampling rates observed:")
    print(exploration.groupby("sfreq").size().rename("n_subjects"))

    # Hard check
    if exploration["sfreq"].nunique() > 1:
        print("WARNING: multiple sampling rates detected")

    # ------------------------------------------------
    # 5.4 Channel consistency exploration
    # ------------------------------------------------
    print("Channel counts observed:")
    print(exploration.groupby("n_channels").size().rename("n_subjects"))

    print("Unique channel layouts (hashes):")
    print(exploration.groupby("channel_hash").size().rename("n_subjects"))

    if exploration["channel_hash"].nunique() > 1:
        print("WARNING: channel sets differ across subjects")

    # ------------------------------------------------
    # 5.5 Recording duration exploration
    # ------------------------------------------------
    exploration["duration_min"] = exploration["duration_sec"] / 60.0

    print("Recording duration (minutes):")
    print(exploration["duration_min"].describe())

    # Flag very short recordings
    short_thresh_min = 2.0
    n_short = (exploration["duration_min"] < short_thresh_min).sum()
    print(f"Recordings shorter than {short_thresh_min} min:", n_short)

    # ------------------------------------------------
    # 5.6 Window-count simulation (no feature computation)
    # ------------------------------------------------
    window_len_sec = 10.0
    overlap_frac = 0.5
    step_sec = window_len_sec * (1 - overlap_frac)

    exploration["n_windows_10s"] = exploration["duration_sec"].apply(
        lambda d: estimate_n_windows(d, window_len_sec, step_sec)
    )

    print("Estimated number of 10s windows per recording:")
    print(exploration["n_windows_10s"].describe())

    print("Subjects with < 10 windows:",
          (exploration["n_windows_10s"] < 10).sum())

    return exploration


# ============================================
# Pipeline
# ============================================

def main() -> None:
    c.ensure_dirs()

    print("RAW_DIR:", c.RAW_DIR)
    print("MANIFEST_DIR:", c.MANIFEST_DIR)

    # --- Section 2. File discovery -------------------------------------------
    df_files = discover_spontaneous_set_files(c.RAW_DIR)
    print("Total spontaneous .set files found:", len(df_files))
    print("Example files:")
    print(df_files.head(10))

    # --- Section 3. Strict filename parsing ----------------------------------
    manifest = build_initial_manifest(df_files)

    print("Files before parsing filter:", len(df_files))
    print("Files matching strict pattern:", manifest["parse_ok"].sum())

    # Drop files that failed strict parsing.
    manifest = manifest[manifest["parse_ok"]].reset_index(drop=True)

    print("Rows in manifest after filtering:", len(manifest))
    print("Unique subjects:", manifest["subject_id"].nunique())

    # --- Section 4.1 Drug assignment by recording order ----------------------
    manifest = assign_drug_by_order_rule(manifest)

    # --- Section 4.2 Per-subject validation ----------------------------------
    qc_report = build_qc_report(manifest)

    print("Total subjects:", qc_report["subject_id"].nunique())
    print("Subjects passing basic checks:", qc_report["passes_basic_checks"].sum())

    # Show failures explicitly.
    failures = qc_report[~qc_report["passes_basic_checks"]]
    if len(failures):
        print("Subjects failing basic checks:")
        print(failures.head(20))

    # --- Section 4.3 Create validated manifest -------------------------------
    manifest = manifest.merge(
        qc_report[["subject_id", "passes_basic_checks", "check_notes"]],
        on="subject_id",
        how="left",
    )

    manifest_validated = manifest[manifest["passes_basic_checks"]].copy()

    print("Rows in validated manifest:", len(manifest_validated))
    print("Validated subjects:", manifest_validated["subject_id"].nunique())

    # Final sanity check: EC-only rows that will be used downstream.
    manifest_validated_ec = manifest_validated[manifest_validated["eyes"] == "closed"]
    print("EC-only rows:", len(manifest_validated_ec))

    # --- Section 4.4 Save manifest artifacts (CSV) ---------------------------
    manifest.to_csv(MANIFEST_ALL_PATH, index=False)
    manifest_validated.to_csv(MANIFEST_VALIDATED_PATH, index=False)

    print("Saved manifest files:")
    print(" -", MANIFEST_ALL_PATH)
    print(" -", MANIFEST_VALIDATED_PATH)

    print("Summary:")
    print("  total files:", len(manifest))
    print("  validated files:", len(manifest_validated))
    print("  validated subjects:", manifest_validated["subject_id"].nunique())

    # --- Section 5. Structural exploration (optional) ------------------------
    # Only run if MNE is available and the raw recordings are present.
    if mne is not None and len(manifest_validated):
        try:
            explore_structure(manifest_validated)
        except Exception as e:
            print("Skipping structural exploration:", e)


if __name__ == "__main__":
    main()
