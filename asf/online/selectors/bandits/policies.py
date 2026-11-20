"""Exploration policies: how a posterior becomes an arm's score (UCB or TS)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from asf.online.selectors.bandits.posteriors import Posterior


class ExplorationPolicy(ABC):
    """Abstract exploration policy."""

    @abstractmethod
    def score(
        self, posterior: Posterior, x: np.ndarray, t: float, rng: np.random.Generator
    ) -> float:
        """
        Score one arm at model input ``x`` and time ``t``.
        """


class UCBPolicy(ExplorationPolicy):
    """
    Optimism in the face of uncertainty: ``mean + alpha * width``.

    ``ucb1=True`` multiplies the bonus by ``sqrt(2 ln t)``: with a
    ``GaussianPosterior`` (``prior_precision=0``, ``noise_var=1``) and
    ``alpha=1`` the classic UCB1 index. Without it the bonus is constant in
    ``t``, as in LinUCB (Li et al., 2010).
    """

    def __init__(self, alpha: float = 1.0, ucb1: bool = False) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = float(alpha)
        self.ucb1 = bool(ucb1)

    def score(
        self, posterior: Posterior, x: np.ndarray, t: float, rng: np.random.Generator
    ) -> float:
        width = posterior.width(x)
        if not np.isfinite(width):
            return float("inf")
        bonus = self.alpha * width
        if self.ucb1:
            bonus *= np.sqrt(2.0 * np.log(max(t, 1)))
        return posterior.mean(x) + bonus


class ThompsonPolicy(ExplorationPolicy):
    """
    Thompson Sampling: score = one posterior draw; ``scale`` inflates the
    posterior sd (``v`` in Agrawal & Goyal's LinTS).
    """

    def __init__(self, scale: float = 1.0) -> None:
        if scale < 0:
            raise ValueError("scale must be >= 0.")
        self.scale = float(scale)

    def score(
        self,
        posterior: Posterior,
        x: np.ndarray,
        t: float,
        rng: np.random.Generator,
        size: int | None = None,
    ) -> float | np.ndarray:
        """One draw, or ``size`` draws at once (for the decision uncertainty)."""
        return posterior.sample(x, rng, self.scale, size)
