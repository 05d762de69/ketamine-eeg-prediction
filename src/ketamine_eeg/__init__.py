"""Ketamine EEG drug-state prediction: shared library code.

Submodules
----------
config         : paths and analysis constants (single source of truth)
plotting       : manuscript matplotlib style + figure saving helpers
features       : feature loading / selection for the A/B/C models
fingerprinting : per-feature between-subject vs drug-effect variance decomposition
stats          : permutation-test helpers
"""
from __future__ import annotations

from . import config, features, fingerprinting, plotting, stats

__all__ = ["config", "features", "fingerprinting", "plotting", "stats"]
__version__ = "1.0.0"
