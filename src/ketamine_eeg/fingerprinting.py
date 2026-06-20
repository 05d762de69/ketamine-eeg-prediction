"""Per-feature variance decomposition used by the fingerprinting analysis.

The decomposition separates, for each feature (a wPLI edge or a bandpower
feature), the between-subject identity variance from the within-subject drug
effect, and splits the drug effect into a cross-subject-shared component
(transferable, learnable by a leave-subjects-out classifier) and an
idiosyncratic subject x drug component.

Both the wPLI and the bandpower fingerprinting scripts call ``decompose`` so
the two feature classes are computed under numerically identical definitions.
"""
from __future__ import annotations

import numpy as np

COND_LABELS = ["awake", "ketamine"]
EPS = 1e-12


def decompose(X, subj_ix, drug_ix, subjects, cond_labels=COND_LABELS, eps=EPS):
    """Per-feature variance decomposition (between-subject vs drug effect).

    Parameters
    ----------
    X : (n_epochs, n_features) array
        Feature matrix.
    subj_ix, drug_ix : (n_epochs,) arrays
        Per-epoch subject and drug labels aligned to the rows of ``X``.
    subjects : sequence
        Ordered unique subject identifiers.

    Returns
    -------
    dict of (n_features,) arrays with the variance components, the
    transferability index and the between/within ratios.
    """
    n_subj = len(subjects)
    n_feat = X.shape[1]

    # Per-subject, per-condition mean per feature -> (n_subj, 2, n_feat)
    subj_cond_mean = np.full((n_subj, 2, n_feat), np.nan, dtype=np.float64)
    for i, s in enumerate(subjects):
        for j, c in enumerate(cond_labels):
            mask = (subj_ix == s) & (drug_ix == c)
            subj_cond_mean[i, j, :] = X[mask].mean(axis=0)

    # Per-subject overall mean (conditions weighted equally so unequal
    # awake/ketamine epoch counts do not bias the subject mean).
    subj_mean = subj_cond_mean.mean(axis=1)  # (n_subj, n_feat)

    # Per-subject signed drug effect (awake - ketamine).
    drug_diff_subj = subj_cond_mean[:, 0, :] - subj_cond_mean[:, 1, :]

    between_var = subj_mean.var(axis=0, ddof=1)
    within_drug_msd = (drug_diff_subj ** 2).mean(axis=0)
    within_drug_std = np.sqrt(within_drug_msd)
    between_std = np.sqrt(between_var)

    # Shared vs idiosyncratic decomposition of the drug effect.
    drug_diff_mean = drug_diff_subj.mean(axis=0)
    drug_diff_var = drug_diff_subj.var(axis=0, ddof=1)
    drug_shared_sq = drug_diff_mean ** 2
    drug_idio_sq = drug_diff_var
    drug_transfer = drug_shared_sq / (drug_shared_sq + drug_idio_sq + eps)

    # Pooled within-cell (subject x condition) residual epoch-level variance.
    resid_var_acc = np.zeros(n_feat, dtype=np.float64)
    resid_n = 0
    for i, s in enumerate(subjects):
        for j, c in enumerate(cond_labels):
            mask = (subj_ix == s) & (drug_ix == c)
            x = X[mask]
            if x.shape[0] >= 2:
                resid_var_acc += x.var(axis=0, ddof=1) * (x.shape[0] - 1)
                resid_n += (x.shape[0] - 1)
    resid_var = resid_var_acc / resid_n

    # ICC(1)-style subject reliability (drug as fixed effect, balanced +/-1 coding).
    drug_var_contrib = 0.25 * within_drug_msd
    icc_subject = between_var / (between_var + drug_var_contrib + resid_var)

    ratio = between_var / (within_drug_msd + eps)
    ratio_shared = between_var / (drug_shared_sq + eps)

    return {
        "between_subj_var": between_var,
        "between_subj_std": between_std,
        "within_drug_msd": within_drug_msd,
        "within_drug_std": within_drug_std,
        "drug_shared_sq": drug_shared_sq,
        "drug_idio_sq": drug_idio_sq,
        "drug_transfer": drug_transfer,
        "residual_var": resid_var,
        "icc_subject": icc_subject,
        "ratio_between_over_within": ratio,
        "log10_ratio": np.log10(ratio),
        "ratio_between_over_shared": ratio_shared,
        "log10_ratio_shared": np.log10(ratio_shared),
    }
