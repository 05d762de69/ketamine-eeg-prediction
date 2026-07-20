# Data dictionary

Description of every tracked derived-data and results file. Raw EEG lives under
`data/raw/` and is **not** distributed (see the README). Subject identifiers are
anonymised integer codes.

## `data/derived/manifests/`

| File | Contents |
|------|----------|
| `manifest_spontaneous_validated.csv` | One row per raw recording that passed validation. Columns: `subject_id`, `date_str`, `recording_number`, `eyes` (open/closed), `file_path` (relative to the repo root), `file_name`, `parent_dir`, `drug` (awake/ketamine), `drug_source`, `drug_order_confidence`, `passes_basic_checks`, `check_notes`. |
| `manifest_spontaneous_all.csv` | The full unfiltered discovery manifest (superset of the validated one). |

## `data/derived/features/`

| File | Contents |
|------|----------|
| `features_A_bandpower_epochwise.csv` | Feature set **A**. One row per kept eyes-closed epoch. `logbp_<band>_<region>` columns give natural-log bandpower for delta/theta/alpha/beta × {global, frontal, central, parietal, occipital, temporal} (24 neural features). Plus keys (`subject_id`, `recording_number`, `epoch_index_*`), `drug`, and recording-level QC metadata (`ptp_uv`, `n_*`) which are **excluded** from modelling. |
| `features_A_bandpower_epochwise_meta.json` | Band definitions, rejection threshold, eyes kept, unit of analysis. |
| `features_B_wpli_edges_epoch_sliding.csv` | Feature set **B**. One row per epoch; `<band>_e####` columns give wPLI for each of the 1891 channel pairs in theta/alpha/beta (5673 edges). Same keys / metadata convention as A. |
| `features_B_wpli_edges_epoch_sliding_meta.json` | wPLI parameters (bands, window length/step, connectivity method) and relative output paths. |
| `features_B_wpli_matrices_epoch_sliding.npz` | Dense 62×62 wPLI matrices, one per `(subject, recording, epoch, band)` under key `"{sid}__rec{n}__e{idx:04d}__{band}"`. Used to build reduced-montage feature matrices. |
| `features_subset_bandpower.csv` | Per-channel log bandpower for the 12 electrodes used by the channel-subset analysis (`bp_<band>_<channel>`). Lets `scripts/04` run without raw EEG. |
| `psd_grandavg_cache.npz` | Cached per-recording Welch PSDs: `freqs` (n_freqs,), `awake` and `ketamine` arrays of shape (n_subjects, n_channels, n_freqs). Lets the PSD figures + stats reproduce without raw EEG. |

## `data/derived/montage_info.json`

Channel names, sampling rate, and 3-D sensor positions for the 62-channel
montage. Used for channel indexing and (optional) topographic plots without
loading raw recordings.

## `results/models/`

Per-fold cross-validation outputs. `results_*` = per-fold metrics
(`balanced_accuracy`, `roc_auc`, `accuracy`, confusion counts); `predictions_*`
= per-epoch out-of-fold predictions; `best_params_*` = selected hyper-parameters.
Every row carries an `estimator` column (`rf` / `svm` / `xgb`) and there are 10
outer folds (leave-one-subject-out); `best_params_*` columns are prefixed by the
pipeline step (`rf__*`, `svc__*`, `xgb__*`, plus `pca_n_components` for B/C).

| Stem | Models |
|------|--------|
| `*_nested_groupkfold_Aepoch_BwpliEpochSliding.csv` | Full-montage A / B / C × {rf, svm, xgb} (`scripts/03`). |
| `*_channel_subsets.csv` | Reduced montages × feature sets × {rf, svm, xgb} (`scripts/04`). |

## `results/permutation/`

| File | Contents |
|------|----------|
| `perm_summary_nested_groupkfold_Aepoch_BwpliEpochSliding.csv` | Observed vs null mean and two-sided permutation / t-test p-values for balanced accuracy, ROC-AUC and accuracy (full-montage A/B/C), one row per `estimator` × model. |
| `perm_summary_channel_subsets.csv` | Observed vs null mean and two-sided permutation p-values for balanced accuracy and ROC-AUC (no accuracy / t-test columns), per `estimator` × reduced montage × feature set. |
| `null_{bacc,auc,acc}_<estimator>_<model>.npy` | Permutation null distributions (full montage), one file per estimator × model. |
| `null_channel_subsets_{bacc,auc}.npz` | Permutation null distributions (subsets); array keys are `"<estimator>__<model>"`. |

## `results/tables/`

| File | Contents |
|------|----------|
| `main_model_performance.csv` | Headline A/B/C performance (mean ± SD) with permutation p-values (random-forest estimator). |
| `model_performance_by_estimator.csv` | Full-montage A/B/C performance (mean ± SD, permutation p) for each estimator (rf/svm/xgb). |
| `subset_performance_by_estimator.csv` | Channel-subset performance (mean ± SD, permutation p) for each estimator × subset × feature set. |
| `sensitivity_specificity_by_estimator.csv` | Per-fold-averaged sensitivity (ketamine detection, TP/(TP+FN)) and specificity (awake correct, TN/(TN+FP)), ± SD, for each estimator × montage (`full_62ch` + 4 subsets) × feature set; `(sensitivity+specificity)/2` = balanced accuracy. |
| `psd_band_wilcoxon.csv` | Per-band Wilcoxon signed-rank tests (ketamine − awake log power), Holm-corrected. |
| `feat_B_edge_importance_table.csv` | Back-projected random-forest importance for every wPLI edge (band, channel pair, importance). |
| `fingerprint_edge_variance.csv` | Per-edge variance decomposition (between-subject, drug shared/idiosyncratic, residual, ICC, transferability, ratios). |
| `fingerprint_bandpower_variance.csv` | Same decomposition for the 24 bandpower features. |
| `fingerprint_top20_alpha_variance.csv` | Decomposition for the top-20 classifier-weighted alpha edges. |
| `fingerprint_transferability_summary.csv` | wPLI vs bandpower transferability medians. |
| `fingerprint_subject_id_accuracy.csv` | Condition-aware subject-ID classifier balanced accuracy. |
| `fingerprint_subject_id_confusion_{wpli,bandpower}.csv` | Subject-ID confusion matrices. |
