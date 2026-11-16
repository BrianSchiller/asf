"""
Feature drift detection with a one-class classifier (OCDD).

This implements the One-Class Drift Detector of Gözüaçık and Can ("Concept
learning using one-class classifiers for implicit drift detection in evolving
data streams", Artificial Intelligence Review 54, 2021), Algorithm 1:

1. a one-class SVM is trained on the reference data and labels every new
   instance as typical or outlier,
2. the labels of the most recent ``window_size`` instances are kept; once the
   window is full, the share of outliers in it is compared with ``threshold``
   after every instance,
3. on drift only the newest ``threshold * window_size`` instances are kept
   and the one-class SVM is retrained on them; without drift the oldest
   instance leaves the window.

Differences from the paper:

* ``baseline="cross_validated"`` (off by default) is not in the paper. A
  one-class SVM calls a part of perfectly typical new data outliers: about
  ``nu`` of it, and more when it was trained on few instances with many
  features. With the paper's settings (``nu=0.5``, ``threshold=0.3``) the
  detector therefore signals whenever its window is full, drift or not. With
  the baseline, that expected share is estimated by cross-validation at every
  training and drift is signalled if the window exceeds it by ``threshold``.
* Features are median-imputed and standardised before they reach the SVM
  (the paper does not preprocess); both steps are refitted with every
  training. This is the numeric branch of ASF's default preprocessor
  (:func:`~asf.preprocessing.sklearn_preprocessor.get_default_preprocessor`),
  applied to arrays because the detector is called once per instance.
* The incoming instance is added to the window before the check. As printed,
  Algorithm 1 does not store the instance that arrives on a check.
* After a drift the kept instances are labelled again by the retrained SVM,
  as drawn in Fig. 3 of the paper. Keeping their old labels would start the
  next window with the outliers that caused the drift.
* The detector adapts itself when it signals, so
  :meth:`~OneClassDriftDetector.reset` (called by a selector after it rebuilt
  its model) does nothing.
* :meth:`~OneClassDriftDetector.score` (per-instance unfamiliarity) is not
  part of the paper.
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

    The detector consumes one instance at a time. The first check happens
    once ``window_size`` instances have arrived after :meth:`fit`; from then
    on it checks at every instance, until a drift empties most of the window.
    Only features are used; ``performance`` is ignored.

    The reference is the data given to :meth:`fit` (the selector passes its
    buffer). After a drift the detector retrains on the newest ``n_keep_``
    window instances by itself, whether or not the selector responds.

    Parameters
    ----------
    window_size : int, default=250
        Number of recent instances ``w`` whose outlier labels are kept
        (paper: 250).
    threshold : float, default=0.3
        ``rho`` of the paper, in (0, 1]: drift is signalled if the share of
        outliers in the window is at least ``threshold`` (plus the baseline,
        see below). It is also the share of the window that is kept, and
        retrained on, after a drift (paper: 0.3).
        ``threshold * window_size`` must be at least 2.
    nu : float, default=0.5
        ``nu`` of the one-class SVM, in (0, 1]: an upper bound on the share
        of training instances outside the boundary (paper: 0.5).
    kernel : str, default="rbf"
        Kernel of the one-class SVM.
    gamma : str or float, default="scale"
        Kernel coefficient of the one-class SVM.
    baseline : {"none", "cross_validated"}, default="none"
        ``"none"``: the paper's rule, ``outlier share >= threshold``.
        ``"cross_validated"``: signal if
        ``outlier share - baseline_share_ >= threshold``, where
        ``baseline_share_`` is the share of training instances a one-class
        SVM calls outliers when it was not trained on them (``n_folds``-fold
        cross-validation on consecutive folds, repeated at every training).
        With fewer than ``2 * n_folds`` training instances the share on the
        training data itself is used.
    n_folds : int, default=5
        Folds for ``baseline="cross_validated"``.

    Attributes
    ----------
    feature_names_ : list[str]
        Feature names seen in ``fit``.
    n_keep_ : int
        Instances kept after a drift, ``threshold * window_size`` rounded.
    baseline_share_ : float
        Outlier share expected without drift (0 for ``baseline="none"``).
    last_outlier_share_ : float or None
        Share of outliers in the window at the last check.
    n_retrains_ : int
        Number of times the detector retrained itself after a drift.

    Notes
    -----
    The one-class SVM is deterministic, so the detector has no
    ``random_state``.
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
        if len(features) < 2:
            raise ValueError("At least two reference instances are required.")
        if features.shape[1] == 0:
            raise ValueError("At least one feature is required.")
        self.feature_names_ = [str(c) for c in features.columns]
        self._train(features.to_numpy(float))
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
        Consume new instance(s) and report whether the latest check found drift.

        Accepts one or several rows; if several rows complete several checks,
        the signal reports drift if any of them did. The signal's ``score``
        is the outlier share of the window at the last check minus the
        baseline share at that check (drift if at least ``threshold``).
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
        """
        Per-instance distance outside the one-class boundary.

        The negated decision value of the current one-class SVM: positive for
        instances it calls outliers, negative for typical ones, larger the
        further outside. The scale depends on the kernel and changes with
        every retraining.
        """
        X = self._matrix(features)
        assert self._model is not None
        return -np.asarray(self._model.decision_function(X), dtype=float)

    def reset(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> None:
        """
        Do nothing: the detector re-references itself when it signals.

        A selector calls this after it rebuilt its model. Refitting on its
        buffer would mix instances from before and after the drift, whereas
        the detector has already retrained on the newest window instances.
        """

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before update()/score().")
        if list(features.columns) == self.feature_names_:
            return features.to_numpy(float)
        missing = [
            c for c in self.feature_names_ if c not in map(str, features.columns)
        ]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]}")
        return features.rename(columns=str)[self.feature_names_].to_numpy(float)
