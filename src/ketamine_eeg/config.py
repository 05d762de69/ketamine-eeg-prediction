"""Central configuration: paths, analysis constants, and small path helpers.

Every script and notebook imports paths and constants from here so that the
whole project has a single source of truth and contains no hard-coded
machine-specific paths.

Directory layout (relative to the repository root)::

    data/
        raw/                     # EEG recordings -- NOT distributed (see README)
        derived/
            manifests/           # recording manifest (validated)
            features/            # extracted features + caches (tracked)
            montage_info.json    # channel names + 2D sensor positions
    results/
        figures/                 # publication figures
        models/                  # cross-validated model outputs (per fold)
        permutation/             # permutation-test null distributions + summaries
        tables/                  # result tables + statistical-test outputs
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# src/ketamine_eeg/config.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "spontaneous"
DERIVED_DIR = DATA_DIR / "derived"

MANIFEST_DIR = DERIVED_DIR / "manifests"
FEATURES_DIR = DERIVED_DIR / "features"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
PERM_DIR = RESULTS_DIR / "permutation"
TABLES_DIR = RESULTS_DIR / "tables"

# --- canonical files ------------------------------------------------------- #
MANIFEST_PATH = MANIFEST_DIR / "manifest_spontaneous_validated.csv"
MONTAGE_INFO_PATH = DERIVED_DIR / "montage_info.json"

FEAT_A_PATH = FEATURES_DIR / "features_A_bandpower_epochwise.csv"
FEAT_A_META_PATH = FEATURES_DIR / "features_A_bandpower_epochwise_meta.json"
FEAT_B_EDGES_PATH = FEATURES_DIR / "features_B_wpli_edges_epoch_sliding.csv"
FEAT_B_EDGES_META_PATH = FEATURES_DIR / "features_B_wpli_edges_epoch_sliding_meta.json"
FEAT_B_MATRICES_PATH = FEATURES_DIR / "features_B_wpli_matrices_epoch_sliding.npz"
FEAT_SUBSET_BANDPOWER_PATH = FEATURES_DIR / "features_subset_bandpower.csv"
PSD_CACHE_PATH = FEATURES_DIR / "psd_grandavg_cache.npz"

# Main A/B/C model outputs. The result tables carry an ``estimator`` column
# (rf/svm/xgb), so the stem is estimator-agnostic (no longer prefixed "rf_").
MODEL_STEM = "nested_groupkfold_Aepoch_BwpliEpochSliding"
RESULTS_MAIN_PATH = MODELS_DIR / f"results_{MODEL_STEM}.csv"
PREDS_MAIN_PATH = MODELS_DIR / f"predictions_{MODEL_STEM}.csv"
PARAMS_MAIN_PATH = MODELS_DIR / f"best_params_{MODEL_STEM}.csv"
PERM_SUMMARY_MAIN_PATH = PERM_DIR / f"perm_summary_{MODEL_STEM}.csv"

# Channel-subset model outputs (also carry an ``estimator`` column).
RESULTS_SUBSET_PATH = MODELS_DIR / "results_channel_subsets.csv"
PREDS_SUBSET_PATH = MODELS_DIR / "predictions_channel_subsets.csv"
PARAMS_SUBSET_PATH = MODELS_DIR / "best_params_channel_subsets.csv"
PERM_SUMMARY_SUBSET_PATH = PERM_DIR / "perm_summary_channel_subsets.csv"


def ensure_dirs() -> None:
    """Create the standard output directories if they do not yet exist."""
    for d in (FEATURES_DIR, MANIFEST_DIR, FIGURE_DIR, MODELS_DIR, PERM_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_raw_path(file_path: str | Path) -> Path:
    """Resolve a manifest ``file_path`` to an absolute path.

    Manifest paths are stored relative to the repository root (e.g.
    ``data/raw/spontaneous/210_..._afterICA.set``). Absolute paths are
    returned unchanged so the project also works if a manifest still holds
    machine-specific absolute paths.
    """
    p = Path(file_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --------------------------------------------------------------------------- #
# Analysis constants (single source of truth across scripts / notebooks)
# --------------------------------------------------------------------------- #
SEED = 0

# Frequency bands (Hz). Connectivity (wPLI) is computed for theta/alpha/beta only.
BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
WPLI_BANDS = ["theta", "alpha", "beta"]

PSD_FMIN, PSD_FMAX = 1.0, 40.0
EYES_KEEP = "closed"
REJECT_PTP_UV = 250.0

N_CHANNELS = 62
N_EDGES_PER_BAND = N_CHANNELS * (N_CHANNELS - 1) // 2  # 1891

# Cross-validation.
# There are exactly 10 subjects, so an outer subject-grouped GroupKFold with
# OUTER_SPLITS=10 is leave-one-subject-out (each fold holds out one subject's
# awake + ketamine recordings). This is the literal "10-fold" for this cohort;
# GroupKFold cannot exceed the number of groups without leaking subjects.
OUTER_SPLITS = 10
INNER_SPLITS = 4
N_PERM = 1000
PCA_N_COMPONENTS = 100

# Model names used throughout the project
MODEL_A = "A_bandpower_epoch"
MODEL_B = "B_wpli_edges_epoch_sliding"
MODEL_C = "C_AplusB_epoch_sliding"
USE_PCA_FOR = {MODEL_B, MODEL_C}
FULL_MODEL_MAP = {"A": MODEL_A, "B": MODEL_B, "C": MODEL_C}

# Classifier families compared side by side so that a null result (or a positive
# one) cannot be attributed to a single algorithm. See ``ketamine_eeg.models``
# for how each is built (SVM is standardised; the wPLI sets B/C get a shared
# PCA(100) for every estimator so only the classifier varies).
ESTIMATORS = ("rf", "svm", "xgb")
ESTIMATOR_LABELS = {"rf": "Random forest", "svm": "SVM (RBF)", "xgb": "XGBoost"}

# Random-forest hyper-parameter grid for the full-montage A/B/C models
PARAM_GRID_RF = {
    "n_estimators": [200, 400, 600, 1000],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 4, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": ["sqrt", 0.3, 0.8],
}
# Smaller grid used for the channel-subset sweep (60 outer tasks)
PARAM_GRID_RF_SUBSET = {
    "n_estimators": [200, 400, 800],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 6, 10],
    "min_samples_leaf": [2, 4, 5],
    "max_features": ["sqrt", 0.3, 0.8],
}

# RBF-SVM grid (features are standardised; class_weight="balanced").
PARAM_GRID_SVM = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 1e-2, 1e-3],
}
PARAM_GRID_SVM_SUBSET = PARAM_GRID_SVM

# XGBoost (gradient-boosted trees) grid for the full-montage models.
PARAM_GRID_XGB = {
    "n_estimators": [300, 600],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
# Smaller XGBoost grid for the channel-subset sweep.
PARAM_GRID_XGB_SUBSET = {
    "n_estimators": [200, 400],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
}

# Convenience lookup: (estimator, is_subset) -> hyper-parameter grid.
PARAM_GRIDS = {
    ("rf", False): PARAM_GRID_RF, ("rf", True): PARAM_GRID_RF_SUBSET,
    ("svm", False): PARAM_GRID_SVM, ("svm", True): PARAM_GRID_SVM_SUBSET,
    ("xgb", False): PARAM_GRID_XGB, ("xgb", True): PARAM_GRID_XGB_SUBSET,
}

# Channel subsets (modern 10-20 names; T3=T7, T4=T8, T5=P7, T6=P8).
# "bis_quatro" approximates the 4-electrode BIS Quatro frontal strip.
CHANNEL_SUBSETS = {
    "left_lateral": ["Fp1", "F7", "T7", "P7", "O1"],
    "right_lateral": ["Fp2", "F8", "T8", "P8", "O2"],
    "both_lateral": ["Fp1", "F7", "T7", "P7", "O1", "Fp2", "F8", "T8", "P8", "O2"],
    "bis_quatro": ["Fpz", "Fp1", "AF3", "F7"],
}
SUBSET_N_CH = {"left_lateral": 5, "right_lateral": 5, "both_lateral": 10, "bis_quatro": 4}

# Plotting colours (manuscript-consistent)
COND_COLORS = {"awake": "steelblue", "ketamine": "firebrick"}


def enable_smoke_mode() -> None:
    """Shrink grids / permutations and redirect outputs to ``results/_smoke/``.

    Used by ``scripts/03`` and ``scripts/04`` when the environment variable
    ``KETAMINE_SMOKE=1`` is set, so the full multi-estimator LOSO pipeline can
    be exercised end-to-end for correctness in seconds instead of hours. It
    keeps the real OUTER_SPLITS (LOSO) and all three estimators — only the
    hyper-parameter grids, permutation count and output directories change, so
    it never overwrites the published results.
    """
    global INNER_SPLITS, N_PERM
    global PARAM_GRID_RF, PARAM_GRID_RF_SUBSET
    global PARAM_GRID_SVM, PARAM_GRID_SVM_SUBSET, PARAM_GRID_XGB, PARAM_GRID_XGB_SUBSET
    global PARAM_GRIDS, MODELS_DIR, PERM_DIR, TABLES_DIR
    global RESULTS_MAIN_PATH, PREDS_MAIN_PATH, PARAMS_MAIN_PATH, PERM_SUMMARY_MAIN_PATH
    global RESULTS_SUBSET_PATH, PREDS_SUBSET_PATH, PARAMS_SUBSET_PATH, PERM_SUMMARY_SUBSET_PATH

    INNER_SPLITS = 3
    N_PERM = 12
    PARAM_GRID_RF = {"n_estimators": [200], "max_depth": [10], "max_features": ["sqrt"]}
    PARAM_GRID_RF_SUBSET = PARAM_GRID_RF
    PARAM_GRID_SVM = {"C": [1.0, 10.0], "gamma": ["scale"]}
    PARAM_GRID_SVM_SUBSET = PARAM_GRID_SVM
    PARAM_GRID_XGB = {"n_estimators": [200], "max_depth": [3], "learning_rate": [0.1]}
    PARAM_GRID_XGB_SUBSET = PARAM_GRID_XGB
    PARAM_GRIDS = {
        ("rf", False): PARAM_GRID_RF, ("rf", True): PARAM_GRID_RF_SUBSET,
        ("svm", False): PARAM_GRID_SVM, ("svm", True): PARAM_GRID_SVM_SUBSET,
        ("xgb", False): PARAM_GRID_XGB, ("xgb", True): PARAM_GRID_XGB_SUBSET,
    }

    smoke = RESULTS_DIR / "_smoke"
    MODELS_DIR = smoke / "models"
    PERM_DIR = smoke / "permutation"
    TABLES_DIR = smoke / "tables"
    for d in (MODELS_DIR, PERM_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    RESULTS_MAIN_PATH = MODELS_DIR / f"results_{MODEL_STEM}.csv"
    PREDS_MAIN_PATH = MODELS_DIR / f"predictions_{MODEL_STEM}.csv"
    PARAMS_MAIN_PATH = MODELS_DIR / f"best_params_{MODEL_STEM}.csv"
    PERM_SUMMARY_MAIN_PATH = PERM_DIR / f"perm_summary_{MODEL_STEM}.csv"
    RESULTS_SUBSET_PATH = MODELS_DIR / "results_channel_subsets.csv"
    PREDS_SUBSET_PATH = MODELS_DIR / "predictions_channel_subsets.csv"
    PARAMS_SUBSET_PATH = MODELS_DIR / "best_params_channel_subsets.csv"
    PERM_SUMMARY_SUBSET_PATH = PERM_DIR / "perm_summary_channel_subsets.csv"


# Apply smoke mode at *import* time when KETAMINE_SMOKE=1, so it also takes
# effect in the joblib/loky worker processes (which re-import this module fresh
# and never run the launching script's __main__). Without this the workers would
# silently use the full grids and the "smoke" run would take as long as a real one.
if os.environ.get("KETAMINE_SMOKE") == "1":
    enable_smoke_mode()
