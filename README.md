# Predicting ketamine vs. awake state from resting EEG

Machine-learning analysis of resting-state EEG recorded before and after
sub-anaesthetic **ketamine**. The project asks whether the drug state can be
decoded from two complementary feature families — **spectral power**
(bandpower) and **functional connectivity** (weighted phase-lag index, wPLI) —
and how few electrodes are needed to do it. A subject-fingerprinting analysis
explains *why* connectivity decodes poorly: the connectivity drug effect is
largely idiosyncratic, whereas the power drug effect is shared across subjects.

> **Status / placeholders.** Author, ORCID and affiliation are set in
> [`CITATION.cff`](CITATION.cff); add the preprint/DOI there once available.

## Key findings

- **Power decodes the drug state; connectivity does not.** Full-montage
  bandpower (set A) separates ketamine from awake well above chance, while wPLI
  connectivity (set B) sits at chance; combining them (set C) does not beat
  power alone. Exact numbers, with permutation p-values, are in
  [`results/tables/main_model_performance.csv`](results/tables).
- **A sparse montage suffices.** A handful of lateral / frontal electrodes
  recovers most of the full-montage power performance
  ([`results/figures/channel_subsets_heatmap.png`](results/figures)).
- **Why connectivity fails — fingerprinting.** Per-edge variance decomposition
  shows wPLI is dominated by stable between-subject "fingerprints" with a
  drug effect that is mostly idiosyncratic (low transferability), whereas the
  bandpower drug effect is far more transferable across subjects.

## Repository layout

```
.
├── src/ketamine_eeg/        # installable package: shared config + helpers
│   ├── config.py            #   paths, bands, channel subsets, CV constants
│   ├── plotting.py          #   manuscript matplotlib style
│   ├── features.py          #   feature loading / selection for A/B/C models
│   ├── fingerprinting.py    #   between-subject vs drug-effect decomposition
│   └── stats.py             #   permutation-test helpers
├── scripts/                 # numbered, runnable pipeline stages
│   ├── 01_build_manifest.py            # raw EEG -> recording manifest      [needs raw]
│   ├── 02a_extract_bandpower.py        # -> feature set A                   [needs raw]
│   ├── 02b_extract_wpli.py             # -> feature set B (wPLI)            [needs raw]
│   ├── 03_train_models.py              # A/B/C nested CV + permutation tests
│   ├── 04_channel_subset_models.py     # reduced-montage models + perm tests
│   ├── 05_wpli_edge_importance.py      # back-projected edge importance
│   ├── 06_fingerprint_wpli.py          # variance decomposition + subject ID
│   └── 07_fingerprint_bandpower.py     # bandpower decomposition + summary
├── notebooks/               # demo notebooks: reproduce every figure + table + test
│   ├── 01_psd_spectra_and_clusters.ipynb
│   ├── 02_drug_prediction_results.ipynb
│   ├── 03_wpli_edge_importance.ipynb
│   ├── 04_fingerprinting.ipynb
│   └── 00_wpli_sanity_check.ipynb       # wPLI implementation validation (synthetic)
├── data/
│   ├── raw/                 # raw EEG -- NOT distributed (see Data availability)
│   └── derived/             # tracked: manifests, features, caches, montage
├── results/
│   ├── figures/             # publication figures
│   ├── models/              # per-fold cross-validation outputs
│   ├── permutation/         # permutation-test null distributions + summaries
│   └── tables/              # result tables + statistical-test outputs
├── docs/data_dictionary.md  # description of every derived / result file
├── pyproject.toml · requirements.txt · Makefile · LICENSE · CITATION.cff
```

## Installation

Python ≥ 3.10. The pinned versions the published results were produced with are
in [`requirements.txt`](requirements.txt).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the `ketamine_eeg` package
# or simply:  make install
```

## Data availability

The raw EEG (10 subjects, eyes-closed/open recordings before and after
ketamine) is **human-subject data and is not distributed in this repository**.
It is available from the authors on reasonable request, subject to the relevant
data-use agreement.

Everything needed to reproduce the published results **without the raw data**
is tracked under `data/derived/` (the extracted feature tables, the dense wPLI
matrices, the cached per-recording PSDs, and the channel montage). Subject
identifiers are anonymised integer codes.

## Reproducing the results

Two tiers, mirroring the data availability:

| Tier | Command | Needs raw EEG? | Produces |
|------|---------|----------------|----------|
| **1 — everything published** | `make reproduce` | No | all models, tables, statistical tests and figures from the tracked derived data |
| **2 — from raw** | `make features` then `make reproduce` | Yes | re-extracts features (`scripts/01`, `02a`, `02b`) and then runs tier 1 |

`make reproduce` runs `scripts/03`–`07` (models, subsets, edge importance,
fingerprinting) and then executes the four demo notebooks in place. Individual
stages are also available: `make models`, `make subsets`, `make importance`,
`make fingerprint`, `make figures`. See `make help`.

### What reproduces each figure / table / statistical test

| Output | Produced by |
|--------|-------------|
| **Fig** PSD grand average + significant clusters (`psd_combined.png`) | `notebooks/01_psd_spectra_and_clusters.ipynb` |
| **Fig** PSD per subject (`psd_per_subject.png`) | `notebooks/01` |
| **Fig** channel-subset heatmap (`channel_subsets_heatmap.png`) | `notebooks/02_drug_prediction_results.ipynb` |
| **Fig** channel-subset per-fold strip (`channel_subsets_per_fold.png`) | `notebooks/02` |
| **Fig** wPLI edge-importance heatmap (`feat_B_edge_importance_heatmap.png`) | `notebooks/03_wpli_edge_importance.ipynb` |
| **Fig** fingerprint subject-ID confusion (`fingerprint_subject_id_confusion.png`) | `notebooks/04_fingerprinting.ipynb` |
| **Fig** fingerprint transferability (`fingerprint_transferability.png`) | `notebooks/04` |
| **Fig** fingerprint variance decomposition (`fingerprint_variance_decomposition.png`) | `notebooks/04` |
| **Table** full-montage A/B/C performance (`main_model_performance.csv`) | `notebooks/02` ← `scripts/03` |
| **Table** wPLI edge importance (`feat_B_edge_importance_table.csv`) | `scripts/05` |
| **Table** fingerprint variance / transferability / subject-ID accuracy (`fingerprint_*.csv`) | `scripts/06`, `scripts/07` |
| **Test** PSD spatio-spectral cluster permutation test | `notebooks/01` |
| **Test** per-band Wilcoxon signed-rank (Holm-corrected, `psd_band_wilcoxon.csv`) | `notebooks/01` |
| **Test** A/B/C label-permutation tests (bacc / AUC / acc) | `scripts/03` → `results/permutation/` |
| **Test** channel-subset label-permutation tests | `scripts/04` → `results/permutation/` |
| **Test** subject ICC + drug-transferability indices | `scripts/06`, `scripts/07` |

## Methods at a glance

- **Features.** *A* — log bandpower (delta/theta/alpha/beta) over the whole head
  and 5 scalp regions, per epoch (24 features). *B* — wPLI connectivity for
  theta/alpha/beta over all 62-channel pairs, computed with sliding windows
  within each epoch (5673 edges). *C* — A and B concatenated.
- **Classification.** Random forest with subject-aware **nested** cross-
  validation (5 outer / 4 inner GroupKFold, grouped by subject) and an
  exhaustive hyper-parameter grid scored by balanced accuracy. Connectivity
  features are reduced with PCA(100) inside the cross-validation.
- **Significance.** Label permutation with **subject-aware**, recording-level
  shuffles (1000 permutations), refitting the selected hyper-parameters per
  fold; centred two-sided p-values for balanced accuracy, ROC-AUC and accuracy.
- **Fingerprinting.** Per-feature variance decomposition into between-subject
  identity, a cross-subject-shared (transferable) drug effect, an idiosyncratic
  drug effect and residual epoch variance, plus a condition-aware subject-ID
  classifier.

Constants (bands, CV folds, seeds, hyper-parameter grids, channel subsets) live
in [`src/ketamine_eeg/config.py`](src/ketamine_eeg/config.py) as a single
source of truth.

## Citation

If you use this code or its results, please cite the repository via
[`CITATION.cff`](CITATION.cff) (and the associated paper once available).

## License

Released under the [MIT License](LICENSE).
