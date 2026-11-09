"""Confidence models: how much to trust an observation, judged by its features.

A confidence model scores feature vectors with a weight in ``[0, 1]``: close
to 1 for instances that look like the current distribution, close to 0 for
outliers. The bandit uses it in one of two ways (or both):

* fixed at arrival -- ``AbstractContextualBandit(confidence=model)`` scores
  each instance once, when it is observed, and stores the weight;
* re-estimated -- ``weighting=ConfidenceWeighting(model)`` re-scores the
  whole history at every decision.

:class:`GaussianConfidence` is a simple stand-in; a flow-based density model
(FC-TS) implements the same three methods.
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
    Diagonal Gaussian density of the recent feature distribution.

    Running mean and variance with exponential forgetting; the score is
    ``exp(-0.5 * mean_j((z_j - mu_j)^2 / var_j))``, i.e. 1 at the centre and
    decreasing with the average squared z-score.

    Parameters
    ----------
    forgetting : float, default=0.99
        Weight kept by the old estimate per update (1.0 = no forgetting).
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
