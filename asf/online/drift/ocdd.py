"""
Feature drift detection with a one-class classifier (OCDD).

The One-Class Drift Detector of Gözüaçık and Can (Artificial Intelligence
Review 54, 2021), Algorithm 1: a one-class SVM trained on the reference
labels each new instance typical or outlier; once the window of the last
``window_size`` labels is full, drift is signalled when its outlier share
reaches ``threshold``. On drift the SVM is retrained on the newest
``threshold * window_size`` instances; otherwise the oldest one leaves.

Differences from the paper:

* ``baseline="cross_validated"`` (off by default): an SVM calls about ``nu``
  of typical new data outliers (more with few instances and many features),
  so the paper's settings (``nu=0.5``, ``threshold=0.3``) signal whenever the
  window is full. The baseline estimates that share by cross-validation and
  signals only if the window exceeds it by ``threshold``.
* Features are median-imputed and standardised (refitted at every training).
* The arriving instance joins the window before the check.
* After a drift the kept instances are relabelled by the new SVM (as in
  Fig. 3); old labels would start the next window with the drift's outliers.
* The detector re-references itself, so ``reset`` does nothing.
* ``score`` (per-instance unfamiliarity) is not in the paper.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


class OneClassDriftDetector(AbstractDriftDetector):
    """
    Covariate drift detector: outlier share of a one-class SVM (OCDD).

    Checks at every instance once ``window_size`` have arrived after ``fit``
    (features only). After a drift it retrains on the newest ``n_keep_``
    instances by itself, whether or not the selector responds. Defaults are
    the paper's; the SVM is deterministic, so there is no ``random_state``.

    Parameters
    ----------
    threshold : float, default=0.3
        ``rho`` in (0, 1]: the outlier share (above the baseline) that
        signals drift, and the share of the window kept after it
        (``threshold * window_size >= 2``).
    nu, kernel, gamma
        Of the one-class SVM.
    baseline : {"none", "cross_validated"}, default="none"
        ``"cross_validated"`` subtracts ``baseline_share_``, the outlier share
        of an SVM on data it was not trained on (``n_folds`` consecutive
        folds; with fewer than ``2 * n_folds`` instances the in-sample share).

    Attributes: ``n_keep_``, ``baseline_share_``, ``last_outlier_share_`` and
    ``n_retrains_`` (self-retrainings after a drift).
    """

    def __init__(
        self,
        window_size: int = 250,
        threshold: float = 0.3,
        nu: float = 0.5,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        baseline: Literal["none", "cross_validated"] = "none",
        n_folds: int = 5,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1].")
        if not 0 < nu <= 1:
            raise ValueError("nu must be in (0, 1].")
        if baseline not in ("none", "cross_validated"):
            raise ValueError(f"Unknown baseline: {baseline!r}")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2.")
        n_keep = math.floor(threshold * window_size + 0.5)
        if n_keep < 2:
            raise ValueError("threshold * window_size must be at least 2.")
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = gamma
        self.baseline = baseline
        self.n_folds = int(n_folds)

        self._model: Pipeline | None = None
        self._rows: deque[np.ndarray] = deque()
        self._flags: deque[bool] = deque()
        self.feature_names_: list[str] = []
        self.n_keep_ = min(n_keep, self.window_size)
        self.baseline_share_ = 0.0
        self.last_outlier_share_: float | None = None
        self._last_score: float | None = None
        self.n_retrains_ = 0

    # ------------------------------------------------------------------ fit
    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "OneClassDriftDetector":
        """Train the one-class SVM on ``features`` and empty the window."""
        self._train(self._reference_matrix(features))
        self._rows.clear()
        self._flags.clear()
        self.last_outlier_share_ = None
        self._last_score = None
        self.n_retrains_ = 0
        return self

    def _train(self, X: np.ndarray) -> None:
        """(Re)train the one-class model on ``X`` and set the baseline share."""
        self._model = self._new_model().fit(X)
        self.baseline_share_ = self._expected_outlier_share(X)

    def _new_model(self) -> Pipeline:
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma),
        )

    def _expected_outlier_share(self, X: np.ndarray) -> float:
        """Share of typical instances the model is expected to call outliers."""
        if self.baseline == "none":
            return 0.0
        if len(X) < 2 * self.n_folds:
            return float(self._outliers(X).mean())
        flags = np.empty(len(X), dtype=bool)
        for train, test in KFold(self.n_folds).split(X):
            model = self._new_model().fit(X[train])
            flags[test] = model.predict(X[test]) == -1
        return float(flags.mean())

    def _outliers(self, X: np.ndarray) -> np.ndarray:
        """Boolean outlier label of each row under the current model."""
        assert self._model is not None
        return self._model.predict(X) == -1

    # --------------------------------------------------------------- update
    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        """
        Consume one or more rows; drift if any completed check drifted.
        ``score`` = outlier share minus baseline share at the last check.
        """
        drift = False
        for row in self._matrix(features):
            self._rows.append(row)
            self._flags.append(bool(self._outliers(row[None, :])[0]))
            if len(self._rows) == self.window_size:
                drift |= self._check()
        return DriftSignal(drift=drift, score=self._last_score)

    def _check(self) -> bool:
        """Compare the window's outlier share with the threshold and move on."""
        n_outliers = sum(self._flags)
        self.last_outlier_share_ = n_outliers / self.window_size
        self._last_score = self.last_outlier_share_ - self.baseline_share_
        excess = n_outliers - self.baseline_share_ * self.window_size
        if excess < self.threshold * self.window_size - 1e-9:
            self._rows.popleft()
            self._flags.popleft()
            return False
        kept = np.vstack(list(self._rows)[-self.n_keep_ :])
        self._train(kept)
        self._rows = deque(kept)
        self._flags = deque(bool(flag) for flag in self._outliers(kept))
        self.n_retrains_ += 1
        return True

    # ---------------------------------------------------------------- score
    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Negated SVM decision value: positive outside the boundary (scale
        changes with every retraining)."""
        X = self._matrix(features)
        assert self._model is not None
        return -np.asarray(self._model.decision_function(X), dtype=float)

    def reset(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> None:
        """Do nothing: refitting on the selector's buffer would mix data from
        before and after the drift; the detector already retrained itself."""
