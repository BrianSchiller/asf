"""Turning instance features into the vectors a bandit works with.

Two vectors are derived from every instance:

* ``z`` -- the *feature vector*: NaNs imputed, features standardised. It is what
  similarity-based components see (nearest-neighbour windows, partitioners,
  confidence models).
* ``x`` -- the *model input* of the arm posteriors: ``[z, 1]`` (features plus a
  bias term) for contextual bandits, ``[1]`` for context-free ones.

Keeping both apart lets a context-free posterior (plain TS) still use the
features to decide *which* past observations are relevant (e.g. DTS-OC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ContextBuilder:
    """
    Impute and standardise features with statistics of the warm-start data.

    Parameters
    ----------
    standardize : bool, default=True
        Scale every feature to zero mean and unit variance. Without warm-start
        data (cold start) no statistics exist and features are only imputed.
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
    Map feature vectors ``z`` (shape ``(n, d)``) to posterior inputs ``x``.

    Contextual: ``[z, 1]`` (shape ``(n, d + 1)``). Context-free: ``[1]``
    (shape ``(n, 1)``), which turns a linear posterior into a plain mean.
    """
    z = np.atleast_2d(z)
    ones = np.ones((z.shape[0], 1))
    return np.hstack([z, ones]) if contextual else ones
