#!/usr/bin/env python
# coding: utf-8
"""
Merged PSD figures from 05_psd_visualization.ipynb.

Produces two synthesis figures:
  * psd_combined.png      — two stacked panels (grand-average spectrum on top,
                            log-power ratio with cluster overlay below) sharing
                            one frequency axis.
  * psd_single_panel.png  — single axes folding both messages together: the two
                            grand-average spectra (direction of effect = which
                            curve is on top) with significant frequency ranges
                            marked directly. Band boundaries are drawn as light
                            dashed verticals (no fill) so the grey significance
                            shading is the only filled element and reads clearly.

The PSD computation (reads every eyes-closed recording) is cached to an
.npz so re-renders are fast; delete the cache to recompute from raw.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

# ----------------------------------------------------------------------
# Config — mirrors 05_psd_visualization.ipynb
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "derived" / "manifests" / "manifest_spontaneous_validated.csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CACHE_PATH = PROJECT_ROOT / "data" / "derived" / "features" / "psd_grandavg_cache.npz"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EYES_KEEP = "closed"
FMIN, FMAX = 1.0, 40.0

BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}
COLORS = {"awake": "steelblue", "ketamine": "firebrick"}
BAND_COLORS = {"delta": "#e8e8e8", "theta": "#d8eaf7", "alpha": "#d4f0d4", "beta": "#fce8d4"}


# ----------------------------------------------------------------------
# Compute (or load cached) per-recording PSDs
# ----------------------------------------------------------------------
def compute_psds():
    import mne
    mne.set_log_level("WARNING")

    manifest = pd.read_csv(MANIFEST_PATH)
    df = manifest[manifest["eyes"] == EYES_KEEP].copy()
    df = df.sort_values(["subject_id", "recording_number"]).reset_index(drop=True)

    psds = {"awake": [], "ketamine": []}
    freqs = None
    for _, row in df.iterrows():
        epochs = mne.io.read_epochs_eeglab(row["file_path"], verbose="ERROR")
        epochs.load_data()
        spectrum = epochs.compute_psd(
            method="welch",
            fmin=FMIN, fmax=FMAX,
            n_fft=int(epochs.info["sfreq"] * 2),
            n_overlap=int(epochs.info["sfreq"]),
            verbose="ERROR",
        )
        psd_mean = spectrum.get_data().mean(axis=0)   # (n_channels, n_freqs)
        if freqs is None:
            freqs = spectrum.freqs
        psds[row["drug"]].append(psd_mean)

    awake = np.array(psds["awake"])
    ketamine = np.array(psds["ketamine"])
    return freqs, awake, ketamine


if CACHE_PATH.exists():
    print("Loading cached PSDs from", CACHE_PATH)
    cache = np.load(CACHE_PATH)
    freqs, awake, ketamine = cache["freqs"], cache["awake"], cache["ketamine"]
else:
    print("Computing PSDs from raw recordings (this reads every eyes-closed file)…")
    freqs, awake, ketamine = compute_psds()
    np.savez_compressed(CACHE_PATH, freqs=freqs, awake=awake, ketamine=ketamine)
    print("Cached to", CACHE_PATH)

print(f"Awake {awake.shape} | Ketamine {ketamine.shape} | {len(freqs)} freq bins")


# ----------------------------------------------------------------------
# Stats — cluster-based permutation test (paired, within-subject)
# ----------------------------------------------------------------------
from mne.stats import permutation_cluster_1samp_test

awake_lp = np.log(awake.mean(axis=1))        # (n_subj, n_freqs)
ketamine_lp = np.log(ketamine.mean(axis=1))
diff_lp = ketamine_lp - awake_lp
n_subj = diff_lp.shape[0]

T_OBS, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
    diff_lp,
    n_permutations=10_000,
    threshold=2.262,   # t-crit, df = n_subj - 1 = 9, p=0.05 two-tailed
    tail=0,
    out_type="mask",
    seed=0,
    verbose=False,
)
print(f"{len(clusters)} cluster(s); significant: "
      f"{[round(p, 4) for p in cluster_pv if p < 0.05]}")

# Significant cluster frequency ranges
sig_ranges = []
for cmask, p in zip(clusters, cluster_pv):
    if p < 0.05:
        idx = np.arange(*cmask[0].indices(len(freqs))) if isinstance(cmask, tuple) \
            else np.where(cmask)[0]
        sig_ranges.append((freqs[idx].min(), freqs[idx].max()))


# ----------------------------------------------------------------------
# Figure 1 — two stacked panels sharing the frequency axis
# ----------------------------------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.07},
)

for ax, alpha in ((ax_top, 0.15), (ax_bot, 0.12)):
    for band, (flo, fhi) in BANDS_HZ.items():
        ax.axvspan(flo, fhi, alpha=alpha, color=BAND_COLORS[band], zorder=0)

for band, (flo, fhi) in BANDS_HZ.items():
    ax_top.annotate(band, xy=((flo + fhi) / 2, 0.975), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=9, color="0.45")

for drug, color in COLORS.items():
    data = awake if drug == "awake" else ketamine
    per_subj = data.mean(axis=1)
    grand = per_subj.mean(axis=0)
    sem = per_subj.std(axis=0) / np.sqrt(len(per_subj))
    ax_top.semilogy(freqs, grand, color=color, linewidth=2, zorder=3)
    ax_top.fill_between(freqs, np.maximum(grand - sem, 1e-30), grand + sem,
                        alpha=0.25, color=color, zorder=2)

ax_top.set_ylabel("Power spectral density (a.u.)", fontsize=12)
ax_top.set_xlim(FMIN, FMAX)
cond_handles = [
    Line2D([0], [0], color=COLORS["awake"], lw=2, label="Awake"),
    Line2D([0], [0], color=COLORS["ketamine"], lw=2, label="Ketamine"),
]
ax_top.legend(handles=cond_handles, loc="upper right", fontsize=10, framealpha=0.9)
ax_top.text(-0.085, 1.0, "A", transform=ax_top.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")

ax_bot.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
grand_ratio = diff_lp.mean(axis=0)
sem_ratio = diff_lp.std(axis=0) / np.sqrt(n_subj)
ax_bot.plot(freqs, grand_ratio, color="purple", linewidth=2, zorder=3)
ax_bot.fill_between(freqs, grand_ratio - sem_ratio, grand_ratio + sem_ratio,
                    alpha=0.25, color="purple", zorder=2)

ylim = ax_bot.get_ylim()
bar_y = ylim[1] - 0.06 * (ylim[1] - ylim[0])
drew_sig = False
for cmask, p in zip(clusters, cluster_pv):
    if p < 0.05:
        idx = np.arange(*cmask[0].indices(len(freqs))) if isinstance(cmask, tuple) \
            else np.where(cmask)[0]
        ax_bot.plot(freqs[idx], np.full(len(idx), bar_y),
                    color="black", linewidth=4, solid_capstyle="butt", zorder=4)
        drew_sig = True
ax_bot.set_ylim(ylim)
if drew_sig:
    sig_handle = [Line2D([0], [0], color="black", lw=4,
                         label="significant cluster (p < 0.05)")]
    ax_bot.legend(handles=sig_handle, loc="lower right", fontsize=9, framealpha=0.9)

ax_bot.set_ylabel("Log power ratio\n(ketamine − awake)", fontsize=11)
ax_bot.set_xlabel("Frequency (Hz)", fontsize=12)
ax_bot.set_xlim(FMIN, FMAX)
ax_bot.text(-0.085, 1.0, "B", transform=ax_bot.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right")

fig.align_ylabels([ax_top, ax_bot])

out = FIGURE_DIR / "psd_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("Saved:", out)
plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2 — single-panel synthesis: both spectra + significance
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))

# Band boundaries as light dashed verticals + labels (NO fill) so that the
# grey significance shading is the only filled element and stays distinct.
band_edges = sorted({e for lohi in BANDS_HZ.values() for e in lohi})
for edge in band_edges:
    if FMIN < edge < FMAX:
        ax.axvline(edge, color="0.75", linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
for band, (flo, fhi) in BANDS_HZ.items():
    ax.annotate(band, xy=((flo + fhi) / 2, 0.015), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=9, color="0.45")

# Grand-average spectra with SEM
for drug, color in COLORS.items():
    data = awake if drug == "awake" else ketamine
    per_subj = data.mean(axis=1)
    grand = per_subj.mean(axis=0)
    sem = per_subj.std(axis=0) / np.sqrt(len(per_subj))
    ax.semilogy(freqs, grand, color=color, linewidth=2.2, zorder=3)
    ax.fill_between(freqs, np.maximum(grand - sem, 1e-30), grand + sem,
                    alpha=0.25, color=color, zorder=2)

# Significant ranges: grey vertical shading + a bar near the top
trans = blended_transform_factory(ax.transData, ax.transAxes)
for flo, fhi in sig_ranges:
    ax.axvspan(flo, fhi, color="0.4", alpha=0.13, zorder=1)
    ax.plot([flo, fhi], [0.97, 0.97], transform=trans,
            color="black", linewidth=5, solid_capstyle="butt", zorder=5)

ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("Power spectral density (a.u.)", fontsize=12)
ax.set_xlim(FMIN, FMAX)

handles = [
    Line2D([0], [0], color=COLORS["awake"], lw=2.2, label="Awake"),
    Line2D([0], [0], color=COLORS["ketamine"], lw=2.2, label="Ketamine"),
    Patch(facecolor="0.4", alpha=0.25, label="ketamine ≠ awake\n(cluster p < 0.05)"),
]
ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
fig.tight_layout()

out = FIGURE_DIR / "psd_single_panel.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("Saved:", out)
plt.close(fig)
