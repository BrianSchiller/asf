"""Per-arm reward models (what the bandit believes about one arm).

A posterior is a Bayesian model of an arm's reward given the model input ``x``.
Every posterior supports two ways of being built, so the same class serves
both computation modes of the bandit:

* incrementally -- ``update`` one weighted observation at a time and
  ``decay`` all accumulated evidence (exponential forgetting), or
* from scratch -- ``fit`` a weighted batch (used when the weights of past
  observations depend on the current instance, e.g. nearest neighbours).

The exploration policies only need ``mean``, ``width`` and ``sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Posterior(ABC):
    """
    Abstract posterior over one arm's reward.

    Parameters
    ----------
    dim : int
        Length of the model input ``x``.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    # ---------------------------------------------------------- learning
    @abstractmethod
    def reset(self) -> None:
        """Forget all observations (back to the prior)."""

    @abstractmethod
    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Add one observation with the given weight."""

    @abstractmethod
    def decay(self, gamma: float) -> None:
        """Multiply the weight of every observation seen so far by ``gamma``."""

    def fit(self, X: np.ndarray, rewards: np.ndarray, weights: np.ndarray) -> None:
        """Reset, then add a weighted batch (``X`` has shape ``(n, dim)``)."""
        self.reset()
        for x, r, w in zip(X, rewards, weights):
            if w > 0:
                self.update(x, float(r), float(w))

    # ---------------------------------------------------------- querying
    @abstractmethod
    def mean(self, x: np.ndarray) -> float:
        """Posterior mean of the expected reward at ``x``."""

    @abstractmethod
    def width(self, x: np.ndarray) -> float:
        """Posterior standard deviation of the expected reward at ``x``."""

    def sample(
        self, x: np.ndarray, rng: np.random.Generator, scale: float = 1.0
    ) -> float:
        """
        Draw the expected reward at ``x`` from the (scaled) posterior.

        For a Gaussian posterior, sampling a parameter vector and taking its
        prediction at a single ``x`` is equivalent in distribution to drawing
        ``N(mean(x), (scale * width(x))**2)``, which is what is done here.
        Non-Gaussian posteriors (e.g. Normal-Inverse-Gamma) override this.
        """
        sd = self.width(x)
        if not np.isfinite(sd):
            return float("inf")  # no information yet: always worth a try
        return self.mean(x) + scale * sd * float(rng.standard_normal())


class BayesianLinearRegression(Posterior):
    """
    Weighted Bayesian linear regression with known noise variance.

    Prior ``theta ~ N(0, noise_var / prior_precision * I)``, likelihood
    ``r ~ N(x @ theta, noise_var / w)`` for an observation of weight ``w``.
    The posterior is ``N(theta_hat, noise_var * A^-1)`` with::

        A = prior_precision * I + sum_i w_i x_i x_i^T
        b = sum_i w_i r_i x_i,      theta_hat = A^-1 b

    which is exactly the ridge estimate used by LinUCB and LinTS.

    Parameters
    ----------
    dim : int
        Length of ``x``.
    prior_precision : float, default=1.0
        Ridge regularisation ``lambda`` (must be > 0).
    noise_var : float, default=1.0
        Reward noise variance ``sigma^2``; scales ``width``.
    """

    def __init__(
        self, dim: int, prior_precision: float = 1.0, noise_var: float = 1.0
    ) -> None:
        super().__init__(dim)
        if prior_precision <= 0:
            raise ValueError("prior_precision must be > 0 for a linear posterior.")
        if noise_var <= 0:
            raise ValueError("noise_var must be > 0.")
        self.prior_precision = float(prior_precision)
        self.noise_var = float(noise_var)
        self.reset()

    def reset(self) -> None:
        self.A = self.prior_precision * np.eye(self.dim)
        self.b = np.zeros(self.dim)
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        x = np.asarray(x, dtype=float)
        self.A += weight * np.outer(x, x)
        self.b += weight * reward * x
        self._cache = None

    def decay(self, gamma: float) -> None:
        # Only the data part decays; the prior stays (keeps A well-conditioned).
        prior = self.prior_precision * np.eye(self.dim)
        self.A = gamma * (self.A - prior) + prior
        self.b = gamma * self.b
        self._cache = None

    def fit(self, X: np.ndarray, rewards: np.ndarray, weights: np.ndarray) -> None:
        X = np.asarray(X, dtype=float).reshape(-1, self.dim)
        w = np.asarray(weights, dtype=float)
        self.A = self.prior_precision * np.eye(self.dim) + (X * w[:, None]).T @ X
        self.b = X.T @ (w * np.asarray(rewards, dtype=float))
        self._cache = None

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """``(theta_hat, A^-1)``, cached until the next change."""
        if self._cache is None:
            A_inv = np.linalg.inv(self.A)
            self._cache = (A_inv @ self.b, A_inv)
        return self._cache

    @property
    def theta(self) -> np.ndarray:
        """Posterior mean of the coefficients."""
        return self._posterior()[0]

    def mean(self, x: np.ndarray) -> float:
        return float(np.asarray(x, dtype=float) @ self.theta)

    def width(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        A_inv = self._posterior()[1]
        return float(np.sqrt(self.noise_var * max(x @ A_inv @ x, 0.0)))


class GaussianPosterior(Posterior):
    """
    Context-free Gaussian posterior of an arm's mean reward.

    The one-dimensional special case of :class:`BayesianLinearRegression`
    with ``x = [1]``: precision ``lambda + sum w``, mean
    ``sum(w r) / precision``. Unlike the linear version it allows
    ``prior_precision=0``; an unplayed arm then has infinite width, so UCB
    tries every arm once (classic UCB1 behaviour).

    Parameters
    ----------
    dim : int
        Ignored except for validation (the input is always ``[1]``).
    prior_precision : float, default=1.0
        Pseudo-count of the zero-mean prior (``>= 0``).
    noise_var : float, default=1.0
        Reward noise variance.
    """

    def __init__(
        self, dim: int = 1, prior_precision: float = 1.0, noise_var: float = 1.0
    ) -> None:
        super().__init__(dim)
        if prior_precision < 0:
            raise ValueError("prior_precision must be >= 0.")
        if noise_var <= 0:
            raise ValueError("noise_var must be > 0.")
        self.prior_precision = float(prior_precision)
        self.noise_var = float(noise_var)
        self.reset()

    def reset(self) -> None:
        self.precision = self.prior_precision
        self.weighted_sum = 0.0

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        self.precision += weight
        self.weighted_sum += weight * reward

    def decay(self, gamma: float) -> None:
        self.precision = gamma * (self.precision - self.prior_precision)
        self.precision += self.prior_precision
        self.weighted_sum *= gamma

    def fit(self, X: np.ndarray, rewards: np.ndarray, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=float)
        self.precision = self.prior_precision + float(w.sum())
        self.weighted_sum = float(w @ np.asarray(rewards, dtype=float))

    @property
    def n_effective(self) -> float:
        """Total weight of the observations (the effective number of pulls)."""
        return self.precision - self.prior_precision

    def mean(self, x: np.ndarray | None = None) -> float:
        return self.weighted_sum / self.precision if self.precision > 0 else 0.0

    def width(self, x: np.ndarray | None = None) -> float:
        if self.precision <= 0:
            return float("inf")
        return float(np.sqrt(self.noise_var / self.precision))
