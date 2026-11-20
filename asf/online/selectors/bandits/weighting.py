"""Which past observations count, and how much (the non-stationarity knob).

A :class:`Weighting` multiplies the weight stored with each observation (e.g.
a confidence score) given the instance decided on now. Weightings combine
with ``*``, e.g. ``ExponentialDiscount(0.99) * NearestNeighbours(50)``.

Observations carry the clock ``s`` at arrival (warm start first); a decision
at clock ``t`` sees ``s < t``, the newest with age ``t - 1 - s = 0``.
Weightings that decay geometrically with age are *incremental*: the bandit
can keep running posteriors and ``decay`` them instead of refitting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from asf.online.selectors.bandits.confidence import ConfidenceModel


class Weighting(ABC):
    """Abstract weighting of past observations."""

    #: True if the weights can be maintained with ``Posterior.decay`` alone.
    incremental: bool = False
    #: Incremental mode: decay only the posteriors being updated (per pull)
    #: instead of all posteriors every step.
    decay_on_update: bool = False

    @property
    def decay_factor(self) -> float:
        """Per-step decay applied in incremental mode (1.0 = no forgetting)."""
        return 1.0

    def fit(self, Z: np.ndarray) -> None:
        """Initialise from warm-start feature vectors (default: nothing)."""

    def update(self, z: np.ndarray, t: int) -> None:
        """React to a newly observed instance (default: nothing)."""

    @abstractmethod
    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        """
        Non-negative weights of stored observations (arrival ``steps``,
        feature vectors ``Z``) for deciding on ``z`` at clock ``t``.
        """

    def __mul__(self, other: "Weighting") -> "ProductWeighting":
        left = self.parts if isinstance(self, ProductWeighting) else [self]
        right = other.parts if isinstance(other, ProductWeighting) else [other]
        return ProductWeighting(*left, *right)


class Uniform(Weighting):
    """Every observation counts fully (stationary bandit)."""

    incremental = True

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        return np.ones(len(steps))


class ExponentialDiscount(Weighting):
    """
    Weight ``gamma ** age`` (discounted UCB / TS / LinUCB); memory roughly
    ``1 / (1 - gamma)`` observations.
    """

    incremental = True

    def __init__(self, gamma: float) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        self.gamma = float(gamma)

    @property
    def decay_factor(self) -> float:
        return self.gamma

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        return self.gamma ** (t - 1 - np.asarray(steps, dtype=float))


class PullDiscount(ExponentialDiscount):
    """
    Weight ``gamma ** k`` with ``k`` the newer observations of the same arm in
    the same cell (per pull, as in DTS-OC): an unplayed arm keeps its
    posterior, unlike :class:`ExponentialDiscount`.
    """

    decay_on_update = True

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        n = len(steps)
        newer = n - 1 - np.argsort(np.argsort(steps, kind="stable"), kind="stable")
        return self.gamma ** newer.astype(float)


class SlidingWindow(Weighting):
    """
    Only the last ``size`` global steps count (SW-UCB / SW-TS).
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("size must be >= 1.")
        self.size = int(size)

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        return ((t - 1 - np.asarray(steps)) < self.size).astype(float)


class LastPulls(Weighting):
    """
    Only the last ``n`` observations *of each arm* count, however old (the
    "hot trace" of f-dsw TS).
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1.")
        self.n = int(n)

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        out = np.zeros(len(steps))
        out[np.argsort(steps, kind="stable")[-self.n :]] = 1.0
        return out


class NearestNeighbours(Weighting):
    """
    Only the ``k`` observations nearest to the current instance count
    (Euclidean on ``z``): a feature-space instead of a time window. Chosen
    among all of the arm's observations, before other weights apply.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = int(k)

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        n = len(steps)
        out = np.zeros(n)
        if n <= self.k:
            out[:] = 1.0
            return out
        dist = np.linalg.norm(np.asarray(Z) - np.asarray(z)[None, :], axis=1)
        out[np.argpartition(dist, self.k - 1)[: self.k]] = 1.0
        return out


class ConfidenceWeighting(Weighting):
    """
    Weight each observation by a confidence model's *current* score: the
    history is re-scored at every decision as the model (updated with every
    instance through this weighting) learns. Compare the bandit's
    ``confidence`` argument, which fixes the weight at arrival.
    """

    def __init__(self, model: ConfidenceModel) -> None:
        self.model = model

    def fit(self, Z: np.ndarray) -> None:
        self.model.fit(Z)

    def update(self, z: np.ndarray, t: int) -> None:
        self.model.update(z)

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        if len(steps) == 0:
            return np.ones(0)
        return np.asarray(self.model.score(np.asarray(Z)), dtype=float)


class ProductWeighting(Weighting):
    """Product of several weightings (build it with ``a * b``)."""

    def __init__(self, *parts: Weighting) -> None:
        self.parts = list(parts)

    @property
    def incremental(self) -> bool:  # type: ignore[override]
        scopes = {p.decay_on_update for p in self.parts if p.decay_factor != 1.0}
        return all(p.incremental for p in self.parts) and len(scopes) <= 1

    @property
    def decay_on_update(self) -> bool:  # type: ignore[override]
        return any(p.decay_on_update for p in self.parts)

    @property
    def decay_factor(self) -> float:
        return float(np.prod([p.decay_factor for p in self.parts]))

    def fit(self, Z: np.ndarray) -> None:
        for p in self.parts:
            p.fit(Z)

    def update(self, z: np.ndarray, t: int) -> None:
        for p in self.parts:
            p.update(z, t)

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        out = np.ones(len(steps))
        for p in self.parts:
            out = out * p.weights(z, t, steps, Z)
        return out
