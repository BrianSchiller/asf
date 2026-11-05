"""
Feature drift detection via PCA reconstruction error.

This re-implements, for streams, the multivariate detector of NannyML
(``DataReconstructionDriftCalculator``) that Rook et al. (FOGA 2025) use:

1. median-impute, standardise and fit a PCA on reference data
   (``n_components=0.65``, i.e. as many components as explain 65% of the variance),
2. per instance, the reconstruction error is the Euclidean distance between the
   standardised instance and its PCA reconstruction,
3. errors are averaged over chunks, and a chunk is a drift alarm if its mean
   leaves ``mean +- 3 std`` of the chunk means observed on the reference data.

Caveat inherited from the method: a shift *within* the retained principal
subspace does not change the reconstruction error and goes unnoticed; only
changes that move data off the subspace (or change its spread around it) do.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


class _Projection:
    """Median imputer -> standard scaler -> PCA, with per-row reconstruction error."""

    def __init__(
        self, n_components: int | float | str, random_state: int | None
    ) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X: np.ndarray) -> "_Projection":
        self.pca.fit(self.scaler.fit_transform(self.imputer.fit_transform(X)))
        return self

    def errors(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(self.imputer.transform(X))
        recon = self.pca.inverse_transform(self.pca.transform(Z))
        return np.linalg.norm(Z - recon, axis=1)


class PCAReconstructionDetector(AbstractDriftDetector):
    """
    Stream drift detector based on PCA reconstruction error (NannyML-style).

    The detector consumes one instance at a time. It keeps the reconstruction
    errors of the most recent ``chunk_size`` instances and, every ``stride``
    instances (default ``chunk_size``: non-overlapping chunks, as in NannyML),
    compares their mean with the alert thresholds. A signal is therefore
    delayed by up to ``stride`` instances.

    The thresholds are ``mean +- std_multiplier * std`` of chunk-mean errors on the
    reference data. Since a reference period can be short, the reference chunk
    means are obtained by bootstrapping: ``n_bootstrap`` chunks of ``chunk_size``
    errors are drawn with replacement from the reference errors (NannyML instead
    splits the reference data into consecutive chunks).

    Parameters
    ----------
    n_components : int, float or str, default=0.65
        Passed to ``sklearn.decomposition.PCA`` (a float in (0, 1) keeps that
        share of the variance). 0.65 is NannyML's default.
    chunk_size : int, default=50
        Number of instances per chunk.
    stride : int or None, default=None
        Evaluate every ``stride`` instances (``None`` = ``chunk_size``).
        Values below ``chunk_size`` give overlapping, more responsive but
        correlated checks.
    std_multiplier : float, default=3.0
        Width of the alert band in reference standard deviations (NannyML: 3).
    two_sided : bool, default=True
        Also alert when the error is significantly *lower* than the reference
        (NannyML does). Set to ``False`` to alert on increases only.
    reference_errors : {"in_sample", "cross_validated"}, default="in_sample"
        How the reference errors that determine the thresholds are computed.
        ``"in_sample"`` evaluates the fitted PCA on its own training data
        (NannyML). That underestimates the error on new data, especially for few
        reference instances and many features, and can cause false alarms.
        ``"cross_validated"`` uses out-of-fold errors, which is more honest.
    n_folds : int, default=5
        Folds for ``reference_errors="cross_validated"``.
    n_bootstrap : int, default=500
        Number of bootstrap chunks used to estimate the reference spread.
    random_state : int or None, default=None
        Seed for PCA, folds and bootstrap.

    Attributes
    ----------
    lower_threshold_, upper_threshold_ : float
        Alert band for chunk-mean reconstruction errors (after ``fit``).
    reference_error_ : float
        Mean reference chunk error.
    n_components_ : int
        Number of retained principal components.
    last_chunk_error_ : float or None
        Most recent evaluated chunk-mean error.
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

        self._projection: _Projection | None = None
        self._feature_names: list[str] = []
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
        if len(features) < 2:
            raise ValueError("At least two reference instances are required.")
        self._feature_names = [str(c) for c in features.columns]
        X = features.to_numpy(float)
        self._projection = _Projection(self.n_components, self.random_state).fit(X)
        self.n_components_ = int(self._projection.pca.n_components_)

        if self.reference_errors == "cross_validated" and len(X) >= 2 * self.n_folds:
            errors = self._out_of_fold_errors(X)
        else:
            errors = self._projection.errors(X)
        self._set_thresholds(errors)

        self._window.clear()
        self._since_check = 0
        self.last_chunk_error_ = None
        return self

    def _out_of_fold_errors(self, X: np.ndarray) -> np.ndarray:
        errors = np.empty(len(X))
        splitter = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for train, test in splitter.split(X):
            proj = _Projection(self.n_components, self.random_state).fit(X[train])
            errors[test] = proj.errors(X[test])
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
        """
        Consume new instance(s) and report whether the latest chunk drifted.

        Accepts one or several rows; if several rows complete several checks,
        the signal reports drift if any of them did.
        """
        errors = self.score(features)
        drift = False
        for e in errors:
            self._window.append(float(e))
            self._since_check += 1
            if (
                len(self._window) == self.chunk_size
                and self._since_check >= self.stride
            ):
                self._since_check = 0
                chunk_error = float(np.mean(self._window))
                self.last_chunk_error_ = chunk_error
                drift |= self._outside_band(chunk_error)
        return DriftSignal(drift=drift, score=self.last_chunk_error_)

    def _outside_band(self, chunk_error: float) -> bool:
        if chunk_error > self.upper_threshold_:
            return True
        return self.two_sided and chunk_error < self.lower_threshold_

    # ---------------------------------------------------------------- score
    def score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Per-instance reconstruction error (Euclidean distance in standardised
        space). Higher means less familiar; it is not compared with a threshold.
        """
        if self._projection is None:
            raise RuntimeError("Call fit() before update()/score().")
        missing = [
            c for c in self._feature_names if c not in map(str, features.columns)
        ]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]}")
        X = features.rename(columns=str)[self._feature_names].to_numpy(float)
        return self._projection.errors(X)
