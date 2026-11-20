"""Turning instance features into the vectors a bandit works with.

* ``z``, the *feature vector* (imputed, standardised), is what similarity-
  based components see (nearest neighbours, partitioners, confidence).
* ``x``, the posterior *model input*: ``[z, 1]`` (contextual) or ``[1]``.

Keeping them apart lets a context-free posterior still use the features to
decide which past observations are relevant (e.g. DTS-OC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ContextBuilder:
    """
    Impute and standardise features with warm-start statistics (on a cold
    start there are none, so features are only imputed with 0).
    """

    def __init__(self, standardize: bool = True) -> None:
        self.standardize = bool(standardize)
        self.feature_names: list[str] = []
        self.mean_: np.ndarray = np.zeros(0)
        self.scale_: np.ndarray = np.ones(0)

    @property
    def dim(self) -> int:
        """Length of ``z``."""
        return len(self.feature_names)

    def fit(self, features: pd.DataFrame) -> "ContextBuilder":
        """Learn imputation values and scaling from (warm-start) features."""
        self.feature_names = [str(c) for c in features.columns]
        values = features.to_numpy(dtype=float)
        d = values.shape[1]
        if len(values) == 0:
            self.mean_, self.scale_ = np.zeros(d), np.ones(d)
            return self
        with np.errstate(all="ignore"):  # all-NaN columns give NaN + warnings
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        self.scale_ = std if self.standardize else np.ones(d)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Feature vectors ``z`` as an array of shape ``(n, dim)``."""
        values = features.reindex(columns=self.feature_names).to_numpy(dtype=float)
        values = np.where(np.isnan(values), self.mean_, values)
        if self.standardize:
            values = (values - self.mean_) / self.scale_
        return values


def model_input(z: np.ndarray, contextual: bool) -> np.ndarray:
    """
    Feature vectors ``z`` ``(n, d)`` -> posterior inputs ``[z, 1]``, or
    ``[1]`` (context-free, turning a linear posterior into a plain mean).
    """
    z = np.atleast_2d(z)
    ones = np.ones((z.shape[0], 1))
    return np.hstack([z, ones]) if contextual else ones
