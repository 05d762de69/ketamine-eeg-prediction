"""Classifier factory shared by the drug-state modelling scripts.

Three classifier families are compared so that a null (or positive) result
cannot be attributed to a single algorithm:

    rf   RandomForest                (bagged trees, ``class_weight`` balanced)
    svm  RBF-kernel SVM              (standardised features, ``class_weight`` balanced)
    xgb  XGBoost                     (gradient-boosted trees)

Design choices that keep the comparison fair:

* **The representation is held constant across estimators.** For the
  high-dimensional wPLI feature sets (B and C) in the *full-montage* analysis a
  ``PCA(100)`` step precedes every classifier, exactly as the original
  random-forest pipeline did, so only the classifier changes -- not the input
  space. The low-dimensional bandpower set (A) and all channel-subset feature
  sets are fed to the classifiers directly (no PCA).
* **SVM always standardises** its inputs (``StandardScaler``), which the tree
  models do not need.
* Every classifier is wrapped in a ``Pipeline`` (even when it is the only step)
  so the selected hyper-parameters are reported with a uniform ``<step>__``
  prefix (``rf__``, ``svc__``, ``xgb__``).

The two public helpers are :func:`build_estimator_and_grid` (for nested-CV grid
search) and :func:`predict_and_score` (a uniform predict/score interface: the
SVM is fitted without Platt scaling and scored via ``decision_function`` for
speed, which is monotonic and therefore correct for ROC-AUC).
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config

# Final-step name inside the pipeline for each estimator (drives the ``__`` prefix).
STEP_NAME = {"rf": "rf", "svm": "svc", "xgb": "xgb"}


def _make_classifier(estimator: str):
    """Return an unfitted classifier for ``estimator`` (rf/svm/xgb)."""
    if estimator == "rf":
        return RandomForestClassifier(
            random_state=config.SEED, class_weight="balanced_subsample", n_jobs=1
        )
    if estimator == "svm":
        # probability=False -> no internal Platt-scaling CV; AUC uses decision_function.
        return SVC(
            kernel="rbf", class_weight="balanced", probability=False,
            random_state=config.SEED,
        )
    if estimator == "xgb":
        # Classes are near-balanced (see README); balanced accuracy is the CV
        # scoring metric, so no explicit re-weighting is applied.
        return XGBClassifier(
            random_state=config.SEED, n_jobs=1, tree_method="hist",
            eval_metric="logloss", verbosity=0,
        )
    raise ValueError(f"unknown estimator {estimator!r}; expected one of {config.ESTIMATORS}")


def uses_pca(model_name: str, *, subset: bool) -> bool:
    """Whether this (model, analysis) combination gets the shared PCA(100) step."""
    return (not subset) and (model_name in config.USE_PCA_FOR)


def build_estimator_and_grid(estimator: str, model_name: str, *, subset: bool = False):
    """Build the pipeline and its prefixed hyper-parameter grid.

    Parameters
    ----------
    estimator : {"rf", "svm", "xgb"}
    model_name : str
        Canonical feature-set model name (``config.MODEL_A/B/C`` for the full
        montage, or ``"<subset>__<A/B/C>"`` for the channel-subset sweep).
    subset : bool
        True for the channel-subset analysis (no PCA; low dimensional).

    Returns
    -------
    (sklearn Pipeline, dict)
        The estimator and a grid whose keys are prefixed with the classifier
        step name (e.g. ``"rf__n_estimators"``, ``"svc__C"``).
    """
    step = STEP_NAME[estimator]
    steps = []
    if estimator == "svm":
        steps.append(("scaler", StandardScaler()))
    if uses_pca(model_name, subset=subset):
        steps.append(("pca", PCA(n_components=config.PCA_N_COMPONENTS,
                                 svd_solver="randomized", random_state=config.SEED)))
    steps.append((step, _make_classifier(estimator)))

    grid = config.PARAM_GRIDS[(estimator, subset)]
    prefixed = {f"{step}__{k}": v for k, v in grid.items()}
    return Pipeline(steps), prefixed


def predict_and_score(fitted, X):
    """Return ``(y_pred, score)`` for a fitted pipeline.

    ``score`` is the positive-class probability where available (RF, XGB) and
    the SVM decision function otherwise -- both are valid, monotonic inputs to
    ROC-AUC. ``y_pred`` uses the estimator's native decision rule.
    """
    y_pred = fitted.predict(X)
    if hasattr(fitted, "predict_proba"):
        score = fitted.predict_proba(X)[:, 1]
    else:
        score = fitted.decision_function(X)
    return y_pred, score
