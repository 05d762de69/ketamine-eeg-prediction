"""Submission-style matplotlib configuration for the manuscript figures.

Usage::

    from ketamine_eeg.plotting import apply_style, fig_width, save_fig
    apply_style()
    fig, ax = plt.subplots(figsize=(fig_width("single"), 4))
    ...
    save_fig(fig, "psd_grand_average")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# Toggle: include figure titles or not.
#   False -> manuscript-ready (no titles; descriptive text lives in the caption);
#            figures are saved with a ``_notitle`` suffix.
#   True  -> labelled versions for slides / internal use.
TITLES_FOR_MANUSCRIPT = False

# Output directory for figures (kept as a module attribute so callers can
# override it if needed).
FIGURE_DIR = config.FIGURE_DIR
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# PLOS column widths in inches (single column 5.2", full width 7.5").
_FIG_WIDTHS = {"single": 5.2, "double": 7.5}


def fig_width(kind: str = "single") -> float:
    """Return a PLOS-compliant figure width in inches."""
    return _FIG_WIDTHS[kind]


def apply_style() -> None:
    """Apply a consistent style for all manuscript figures."""
    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 12,
            # Lines and markers
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Layout
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Avoid Type 3 fonts (required by many journals incl. PLOS)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def maybe_title(ax_or_fig, title: str, **kwargs) -> None:
    """Set a title only when ``TITLES_FOR_MANUSCRIPT`` is True."""
    if not TITLES_FOR_MANUSCRIPT:
        return
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(title, **kwargs)
    else:
        ax_or_fig.suptitle(title, **kwargs)


def save_fig(fig, name: str, also_pdf: bool = False) -> Path:
    """Save a figure, appending ``_notitle`` when titles are disabled."""
    suffix = "" if TITLES_FOR_MANUSCRIPT else "_notitle"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{name}{suffix}.png"
    fig.savefig(png_path)
    if also_pdf:
        fig.savefig(FIGURE_DIR / f"{name}{suffix}.pdf")
    return png_path
