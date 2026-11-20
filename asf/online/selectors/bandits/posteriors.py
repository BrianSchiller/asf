"""Per-arm Bayesian reward models given the model input ``x``.

Each can be built incrementally (``update`` one weighted observation,
``decay`` all evidence) or from scratch (``fit`` a weighted batch, when past
weights depend on the current instance). Policies use ``mean``, ``width``
and ``sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Posterior(ABC):
    """
    Abstract posterior over one arm's reward (input length ``dim``).
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
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        scale: float = 1.0,
        size: int | None = None,
    ) -> float | np.ndarray:
        """
        Draw the expected reward at ``x`` (``size`` draws at once if given).

        For a Gaussian posterior this equals sampling parameters and
        predicting at ``x``: ``N(mean(x), (scale * width(x))**2)``.
        """
        sd = self.width(x)
        if not np.isfinite(sd):  # no information yet: always worth a try
            return float("inf") if size is None else np.full(size, np.inf)
        if size is None:
            return self.mean(x) + scale * sd * float(rng.standard_normal())
        return self.mean(x) + scale * sd * rng.standard_normal(size)


class BayesianLinearRegression(Posterior):
    """
    Weighted Bayesian linear regression with known noise variance.

    Prior ``theta ~ N(0, noise_var / prior_precision * I)``, likelihood
    ``r ~ N(x @ theta, noise_var / w)``; posterior ``N(A^-1 b, noise_var
    A^-1)`` with ``A = prior_precision I + sum w x x^T``, ``b = sum w r x``:
    the ridge estimate of LinUCB and LinTS. ``prior_precision > 0``.
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
    Context-free Gaussian posterior of an arm's mean reward: the 1-d case of
    :class:`BayesianLinearRegression` (``x = [1]``, ``dim`` ignored), but
    ``prior_precision=0`` is allowed: an unplayed arm then has infinite
    width, so UCB tries every arm once (classic UCB1).
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


class BetaPosterior(Posterior):
    """
    Beta-Bernoulli posterior for rewards in ``[0, 1]`` (context-free).

    Fractional counts ``alpha += w r``, ``beta += w (1 - r)``; the prior
    pseudo-counts (default uniform, as in f-dsw TS) are never decayed.
    """

    def __init__(
        self, dim: int = 1, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> None:
        super().__init__(dim)
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.reset()

    def reset(self) -> None:
        self.alpha = 0.0
        self.beta = 0.0

    @staticmethod
    def _check(rewards: np.ndarray | float) -> None:
        r = np.asarray(rewards, dtype=float)
        if np.any((r < 0) | (r > 1)):
            raise ValueError(
                "BetaPosterior needs rewards in [0, 1] (use RewardTransform('unit'))."
            )

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        self._check(reward)
        self.alpha += weight * reward
        self.beta += weight * (1.0 - reward)

    def decay(self, gamma: float) -> None:
        self.alpha *= gamma
        self.beta *= gamma

    def fit(self, X: np.ndarray, rewards: np.ndarray, weights: np.ndarray) -> None:
        r, w = np.asarray(rewards, dtype=float), np.asarray(weights, dtype=float)
        self._check(r)
        self.alpha = float(w @ r)
        self.beta = float(w @ (1.0 - r))

    def _params(self) -> tuple[float, float]:
        return self.alpha + self.prior_alpha, self.beta + self.prior_beta

    def mean(self, x: np.ndarray | None = None) -> float:
        a, b = self._params()
        return a / (a + b)

    def width(self, x: np.ndarray | None = None) -> float:
        a, b = self._params()
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))

    def sample(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        scale: float = 1.0,
        size: int | None = None,
    ) -> float | np.ndarray:
        """Draw from the Beta posterior (``scale`` is not used)."""
        draws = rng.beta(*self._params(), size)
        return float(draws) if size is None else draws
