# Reproduction pipeline for the ketamine EEG prediction project.
#
# Two tiers (see README):
#   * `make reproduce` regenerates every model result, table, statistical test
#     and figure from the *tracked derived data* -- no raw EEG required.
#   * `make features` re-extracts features from the raw EEG and therefore
#     requires the raw recordings under data/raw/ (available on request).

PY ?= python3
NB = jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.timeout=1800

.PHONY: help install features models subsets importance fingerprint figures reproduce clean

help:
	@echo "make install      - install pinned dependencies + the ketamine_eeg package"
	@echo "make reproduce    - regenerate all results + figures from tracked derived data"
	@echo "make features     - re-extract features from raw EEG (requires data/raw/)"
	@echo "  -- individual stages --"
	@echo "make models       - full-montage A/B/C nested-CV models + permutation tests"
	@echo "make subsets      - channel-subset models + permutation tests"
	@echo "make importance   - wPLI back-projected edge-importance table"
	@echo "make fingerprint  - subject-fingerprinting variance decomposition + subject-ID"
	@echo "make figures      - execute the demo notebooks (figures + tables + stats)"
	@echo "make clean        - remove generated results/figures (keeps derived features)"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# --- Tier 2: feature extraction from raw EEG (needs data/raw/) --------------
features:
	$(PY) scripts/01_build_manifest.py
	$(PY) scripts/02a_extract_bandpower.py
	$(PY) scripts/02b_extract_wpli.py

# --- Tier 1: everything below runs from tracked derived data ----------------
models:
	$(PY) scripts/03_train_models.py

subsets:
	$(PY) scripts/04_channel_subset_models.py

importance:
	$(PY) scripts/05_wpli_edge_importance.py

fingerprint:
	$(PY) scripts/06_fingerprint_wpli.py
	$(PY) scripts/07_fingerprint_bandpower.py

figures:
	$(NB) notebooks/01_psd_spectra_and_clusters.ipynb
	$(NB) notebooks/02_drug_prediction_results.ipynb
	$(NB) notebooks/03_wpli_edge_importance.ipynb
	$(NB) notebooks/04_fingerprinting.ipynb

reproduce: models subsets importance fingerprint figures
	@echo "Done. See results/ for tables, models, permutation tests and figures."

clean:
	rm -f results/figures/*.png results/figures/*.pdf
