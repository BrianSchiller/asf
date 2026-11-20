"""
Feature drift detection via PCA reconstruction error.

A stream version of NannyML's ``DataReconstructionDriftCalculator``, as used
by Rook et al. (FOGA 2025): median-impute, standardise and fit a PCA on the
reference; an instance's error is the distance to its reconstruction; a chunk
whose mean error leaves ``mean +- 3 std`` of the reference chunk means is an
alarm. A shift *within* the principal subspace goes unnoticed.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


def _projection(n_components: int | float | str, random_state: int | None) -> Pipeline:
    """Median imputer -> standard scaler -> PCA."""
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        PCA(n_components=n_components, random_state=random_state),
    )


def _reconstruction_errors(projection: Pipeline, X: np.ndarray) -> np.ndarray:
    """Euclidean distance of each standardised row to its PCA reconstruction."""
    Z = projection[:-1].transform(X)
    pca = projection[-1]
    return np.linalg.norm(Z - pca.inverse_transform(pca.transform(Z)), axis=1)


class PCAReconstructionDetector(AbstractDriftDetector):
    """
    Stream drift detector on PCA reconstruction error (NannyML-style).

    Every ``stride`` instances (default ``chunk_size``: non-overlapping, as in
    NannyML) the mean error of the last ``chunk_size`` instances is compared
    with ``mean +- std_multiplier * std`` of reference chunk means, so a
    signal comes up to ``stride`` instances late. The reference chunk means
    are bootstrapped (``n_bootstrap`` draws) because a reference period can
    be short; NannyML splits it into consecutive chunks instead.

    Parameters
    ----------
    n_components : int, float or str, default=0.65
        For ``sklearn.decomposition.PCA`` (NannyML's default: 65% variance).
    two_sided : bool, default=True
        Also alert on a significantly *lower* error (as NannyML).
    reference_errors : {"in_sample", "cross_validated"}, default="in_sample"
        ``"in_sample"`` (NannyML) underestimates the error on new data, which
        can cause false alarms with few instances and many features;
        ``"cross_validated"`` uses ``n_folds`` out-of-fold errors.
    random_state : int or None, default=None
        Seed for PCA, folds and bootstrap.

    Attributes: the alert band ``lower_threshold_`` / ``upper_threshold_``,
    ``reference_error_``, ``n_components_`` and ``last_chunk_error_``.
    """

    def __init__(
        self,
        n_components: int | float | str = 0.65,
        chunk_size: int = 50,
        stride: int | None = None,
        std_multiplier: float = 3.0,
        two_sided: bool = True,
        reference_errors: Literal["in_sample", "cross_validated"] = "in_sample",
        n_folds: int = 5,
        n_bootstrap: int = 500,
        random_state: int | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")
        if stride is not None and not 1 <= stride <= chunk_size:
            raise ValueError("stride must be in [1, chunk_size].")
        if reference_errors not in ("in_sample", "cross_validated"):
            raise ValueError(f"Unknown reference_errors: {reference_errors!r}")
        self.n_components = n_components
        self.chunk_size = int(chunk_size)
        self.stride = int(stride) if stride is not None else self.chunk_size
        self.std_multiplier = float(std_multiplier)
        self.two_sided = two_sided
        self.reference_errors = reference_errors
        self.n_folds = int(n_folds)
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = random_state

        self._projection: Pipeline | None = None
        self.feature_names_: list[str] = []
        self._window: deque[float] = deque(maxlen=self.chunk_size)
        self._since_check = 0
        self.lower_threshold_ = float("nan")
        self.upper_threshold_ = float("nan")
        self.reference_error_ = float("nan")
        self.n_components_ = 0
        self.last_chunk_error_: float | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "PCAReconstructionDetector":
        """Fit PCA and alert thresholds on reference data; clears the chunk state."""
        X = self._reference_matrix(features)
        self._projection = _projection(self.n_components, self.random_state).fit(X)
        self.n_components_ = int(self._projection[-1].n_components_)

        if self.reference_errors == "cross_validated" and len(X) >= 2 * self.n_folds:
            errors = self._out_of_fold_errors(X)
        else:
            errors = _reconstruction_errors(self._projection, X)
        self._set_thresholds(errors)

        self._window.clear()
        self._since_check = 0
        self.last_chunk_error_ = None
        return self

    def _out_of_fold_errors(self, X: np.ndarray) -> np.ndarray:
        errors = np.empty(len(X))
        splitter = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for train, test in splitter.split(X):
            proj = _projection(self.n_components, self.random_state).fit(X[train])
            errors[test] = _reconstruction_errors(proj, X[test])
        return errors

    def _set_thresholds(self, errors: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        draws = rng.integers(0, len(errors), size=(self.n_bootstrap, self.chunk_size))
        chunk_means = errors[draws].mean(axis=1)
        center, spread = float(chunk_means.mean()), float(chunk_means.std(ddof=1))
        self.reference_error_ = center
        self.upper_threshold_ = center + self.std_multiplier * spread
        self.lower_threshold_ = max(center - self.std_multiplier * spread, 0.0)

    # --------------------------------------------------------------- update
    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        """Consume one or more rows; drift if any completed check drifted."""
        errors = self.score(features).tolist()
        drift = self._strided_checks(errors, self._window, self._check)
        return DriftSignal(drift=drift, score=self.last_chunk_error_)

    def _check(self) -> bool:
        """Is the mean error of the window outside the alert band?"""
        self.last_chunk_error_ = chunk_error = float(np.mean(self._window))
        if chunk_error > self.upper_threshold_:
            return True
        return self.two_sided and chunk_error < self.lower_threshold_

    # ---------------------------------------------------------------- score
    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Per-instance reconstruction error (standardised space)."""
        X = self._matrix(features)
        assert self._projection is not None
        return _reconstruction_errors(self._projection, X)
