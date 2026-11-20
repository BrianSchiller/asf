"""Confidence models: how much to trust an observation, judged by its features.

A model scores feature vectors in ``[0, 1]`` (near 1 = like the current
distribution). The bandit uses it fixed at arrival (``confidence=``) or
re-estimated at every decision (``ConfidenceWeighting``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ConfidenceModel(ABC):
    """Abstract confidence model over feature vectors ``z``."""

    @abstractmethod
    def fit(self, Z: np.ndarray) -> None:
        """Initialise from warm-start feature vectors (may be empty)."""

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """Incorporate one new instance."""

    @abstractmethod
    def score(self, Z: np.ndarray) -> np.ndarray:
        """Confidence weights in ``[0, 1]`` for ``Z`` of shape ``(n, d)``."""


class GaussianConfidence(ConfidenceModel):
    """
    Diagonal Gaussian of the recent features (exponential ``forgetting``);
    score ``exp(-0.5 * mean_j((z_j - mu_j)^2 / var_j))``, 1 at the centre.
    """

    def __init__(self, forgetting: float = 0.99) -> None:
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1].")
        self.forgetting = float(forgetting)
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, Z: np.ndarray) -> None:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[0] == 0:
            self.mean_ = self.var_ = None
            return
        self.mean_ = Z.mean(axis=0)
        self.var_ = np.maximum(Z.var(axis=0), 1e-6)

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        if self.mean_ is None or self.var_ is None:
            self.mean_, self.var_ = z.copy(), np.ones_like(z)
            return
        a = 1.0 - self.forgetting
        diff = z - self.mean_
        self.mean_ = self.mean_ + a * diff
        self.var_ = np.maximum((1 - a) * (self.var_ + a * diff**2), 1e-6)

    def score(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if self.mean_ is None or self.var_ is None:
            return np.ones(Z.shape[0])
        sq = (Z - self.mean_) ** 2 / self.var_
        return np.exp(-0.5 * sq.mean(axis=1))
