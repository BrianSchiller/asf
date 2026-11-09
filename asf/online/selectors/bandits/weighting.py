"""Which past observations count, and how much (the non-stationarity knob).

A :class:`Weighting` returns a multiplicative weight for every stored
observation of an arm, given the instance that is being decided on now. The
final weight of an observation is::

    stored weight (fixed at arrival, e.g. a confidence score)
        x  weighting.weights(z_query, t, steps, Z)

Weightings combine with ``*``, e.g. ``ExponentialDiscount(0.99) *
NearestNeighbours(50)`` (a discounted nearest-neighbour window, as in DTS-OC).

Time convention: observations carry the clock value ``s`` at which they
arrived (0, 1, 2, ...; warm-start rows first). A decision at clock ``t`` sees
observations with ``s < t``; the newest has age ``t - 1 - s = 0``.

Weightings that depend only on age and decay geometrically are *incremental*:
the bandit can then keep running sums and call ``Posterior.decay`` once per
step instead of refitting from the history (see ``decay_factor``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from asf.online.selectors.bandits.confidence import ConfidenceModel


class Weighting(ABC):
    """Abstract weighting of past observations."""

    #: True if the weights can be maintained with ``Posterior.decay`` alone.
    incremental: bool = False

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
        Weights of stored observations for a decision at clock ``t``.

        Parameters
        ----------
        z : np.ndarray
            Feature vector of the instance being decided on, shape ``(d,)``.
        t : int
            Current clock value.
        steps : np.ndarray
            Arrival clock of each stored observation, shape ``(n,)``.
        Z : np.ndarray
            Feature vectors of the stored observations, shape ``(n, d)``.

        Returns
        -------
        np.ndarray
            Non-negative weights, shape ``(n,)``.
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
    Weight ``gamma ** age`` (discounted UCB / TS / LinUCB).

    Parameters
    ----------
    gamma : float
        Discount per step, in ``(0, 1]``. The effective memory is roughly
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


class SlidingWindow(Weighting):
    """
    Only the observations of the last ``size`` steps count (SW-UCB / SW-TS).

    The window is measured in global steps (instances), not per arm.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("size must be >= 1.")
        self.size = int(size)

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        return ((t - 1 - np.asarray(steps)) < self.size).astype(float)


class NearestNeighbours(Weighting):
    """
    Only the ``k`` stored observations closest to the current instance count.

    Distances are Euclidean on the (standardised) feature vectors ``z``. This
    replaces a *time* window by a *feature-space* window: an arm's posterior is
    built from how it did on similar instances.

    When combined with other weightings, the ``k`` neighbours are chosen among
    all stored observations of the arm, before the other weights apply.

    Parameters
    ----------
    k : int
        Neighbourhood size (per arm).
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
    Weight each observation by a confidence model's *current* score.

    In contrast to a confidence weight fixed at arrival (the bandit's
    ``confidence`` argument), the whole history is re-scored at every
    decision, so observations are re-weighted as the model learns -- e.g. a
    flow model whose density estimate follows the drifting distribution.

    Parameters
    ----------
    model : ConfidenceModel
        Scores feature vectors; it is fitted on the warm start and updated with
        every streamed instance through this weighting.
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
        return all(p.incremental for p in self.parts)

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
