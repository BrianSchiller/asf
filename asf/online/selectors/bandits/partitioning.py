"""Partitioners: route an instance to its own group of arm models.

A partitioner splits feature space into cells (clusters, tree leaves, ...).
Every cell has its own posterior per arm, and only the cell's observations
are used for it. The default is a single cell (one global model per arm).

Cell keys are assigned when an observation arrives and stored with it, so
both computation modes see the same data. Online clustering (DTS-OC) or a
dyadic partitioning tree implement :class:`Partitioner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable

import numpy as np
from sklearn.cluster import KMeans


class Partitioner(ABC):
    """Abstract partitioner over feature vectors ``z``."""

    def fit(self, Z: np.ndarray) -> None:
        """Initialise from warm-start feature vectors (default: nothing)."""

    def update(self, z: np.ndarray) -> None:
        """React to a newly observed instance (default: nothing)."""

    @abstractmethod
    def assign(self, z: np.ndarray) -> Hashable:
        """Key of the cell that ``z`` belongs to."""


class SinglePartition(Partitioner):
    """Everything in one cell: a single global model per arm."""

    def assign(self, z: np.ndarray) -> Hashable:
        return 0


class OnlineKMeans(Partitioner):
    """
    Online k-means with discounted centroids (DTS-OC, Dzhoha & Rozora 2023).

    An instance belongs to its nearest centroid. After it is observed, only
    that centroid moves: ``m = gamma * m + z``, ``w = gamma * w + 1``,
    ``c = m / w``, so centroids follow covariate drift.

    Initialisation: k-means on the warm start if it has at least
    ``n_clusters`` instances; otherwise the first instances (warm start, then
    stream) become the centroids, one each, until ``n_clusters`` exist.
    Instances identical to an existing centroid do not open a new one.

    Parameters
    ----------
    n_clusters : int
        Number of clusters ``K``.
    gamma : float, default=0.95
        Discount of the centroid statistics, in ``(0, 1]``.
    fuzziness : float, default=1.3
        Fuzziness exponent ``l > 1`` of :meth:`membership`.
    random_state : int or None, default=None
        Seed of the warm-start k-means.
    """

    EPS = 1e-12

    def __init__(
        self,
        n_clusters: int,
        gamma: float = 0.95,
        fuzziness: float = 1.3,
        random_state: int | None = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        if fuzziness <= 1:
            raise ValueError("fuzziness must be > 1.")
        self.n_clusters = int(n_clusters)
        self.gamma = float(gamma)
        self.fuzziness = float(fuzziness)
        self.random_state = random_state
        self.fit(np.zeros((0, 0)))

    @property
    def centroids(self) -> np.ndarray:
        """Current centroids, shape ``(n_existing, d)``."""
        return np.array(self._c)

    def fit(self, Z: np.ndarray) -> None:
        self._c: list[np.ndarray] = []  # centroids
        self._m: list[np.ndarray] = []  # discounted feature sums
        self._w: list[float] = []  # discounted weights
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if len(Z) >= self.n_clusters and Z.shape[1] > 0:
            km = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(Z)
            for y, c in enumerate(km.cluster_centers_):
                w = float(np.sum(labels == y))
                self._c.append(c.copy())
                self._m.append(c * w)
                self._w.append(w)
        else:
            for z in Z:  # too few instances: each one seeds a centroid
                self.update(z)

    def distances(self, z: np.ndarray) -> np.ndarray:
        """Euclidean distance of ``z`` to every existing centroid."""
        if not self._c:
            return np.zeros(0)
        return np.linalg.norm(np.array(self._c) - np.asarray(z)[None, :], axis=1)

    def assign(self, z: np.ndarray) -> Hashable:
        d = self.distances(z)
        if len(d) < self.n_clusters and (len(d) == 0 or d.min() > 0):
            return len(d)  # will become a new centroid when observed
        return int(np.argmin(d))

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        y = self.assign(z)
        if y == len(self._c):
            self._c.append(z.copy())
            self._m.append(z.copy())
            self._w.append(1.0)
            return
        self._m[y] = self.gamma * self._m[y] + z
        self._w[y] = self.gamma * self._w[y] + 1.0
        self._c[y] = self._m[y] / self._w[y]

    def membership(self, z: np.ndarray) -> float:
        """
        Fuzzy penalty ``phi`` of ``z`` for its assigned cluster.

        ``phi = 1 / sum_y (d_assigned / d_y) ** (2 / (l - 1))``: 1 when ``z``
        clearly belongs to its cluster, ``1 / K`` when it is equally far from
        all ``K`` centroids. 1 for a new cluster or a zero distance.
        """
        d = self.distances(z)
        y = self.assign(z)
        if y == len(d) or d[y] == 0:
            return 1.0
        d = np.maximum(d, self.EPS)
        return float(1.0 / np.sum((d[y] / d) ** (2.0 / (self.fuzziness - 1.0))))
