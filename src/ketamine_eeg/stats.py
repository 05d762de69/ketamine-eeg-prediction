"""Permutation-test helpers shared by the modelling scripts."""
from __future__ import annotations

import numpy as np


def two_sided_p(null_values, obs) -> float:
    """Centred two-sided permutation p-value.

    ``p = (1 + #{|null - mu| >= |obs - mu|}) / (n + 1)`` where ``mu`` is the
    mean of the (NaN-filtered) null distribution. The +1 in numerator and
    denominator prevents p = 0 for extreme observed values and keeps the test
    conservative.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[~np.isnan(null_values)]
    mu = null_values.mean()
    return float((1 + np.sum(np.abs(null_values - mu) >= abs(obs - mu))) / (len(null_values) + 1))


def subject_aware_label_shuffle(rec_y, subj_rec_idx_list, epoch_to_rec_idx, rng):
    """Return an epoch-level label vector with recording labels shuffled
    *within* each subject (preserving the group structure used by GroupKFold).

    Parameters
    ----------
    rec_y : (n_recordings,) int array
        The true label of each recording.
    subj_rec_idx_list : list of int arrays
        For each subject with >1 recording, the indices (into ``rec_y``) of its
        recordings.
    epoch_to_rec_idx : (n_epochs,) int array
        Maps each epoch to its recording index.
    rng : numpy Generator
    """
    perm_rec_y = rec_y.copy()
    for idxs in subj_rec_idx_list:
        vals = perm_rec_y[idxs].copy()
        rng.shuffle(vals)
        perm_rec_y[idxs] = vals
    return perm_rec_y[epoch_to_rec_idx]


def build_recording_shuffle_structures(rec_id, y, groups):
    """Build the structures needed for subject-aware recording-label shuffles.

    Returns ``(rec_y, subj_rec_idx_list, epoch_to_rec_idx)``.
    """
    import pandas as pd

    rec_df = pd.DataFrame({"rec_id": rec_id, "y": y, "subject": groups})
    assert (rec_df.groupby("rec_id")["y"].nunique() <= 1).all(), \
        "label not constant within recording"
    rec_label = rec_df.groupby("rec_id")["y"].first()
    rec_subject = rec_df.groupby("rec_id")["subject"].first()
    rec_ids_unique = rec_label.index.to_numpy()
    rec_y = rec_label.to_numpy().astype(int)
    rec_subj = rec_subject.to_numpy()
    epoch_to_rec_idx = pd.Index(rec_ids_unique).get_indexer(rec_id)
    assert (epoch_to_rec_idx >= 0).all()
    subj_to_rec: dict = {}
    for i, s in enumerate(rec_subj):
        subj_to_rec.setdefault(s, []).append(i)
    subj_rec_idx_list = [np.asarray(v, int) for v in subj_to_rec.values() if len(v) > 1]
    return rec_y, subj_rec_idx_list, epoch_to_rec_idx
