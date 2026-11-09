"""Exploration policies: how a posterior becomes a score for one arm.

The bandit plays the arm with the highest score. UCB and Thompson Sampling
differ only here; everything else (posteriors, weighting, ...) is shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from asf.online.selectors.bandits.posteriors import Posterior


class ExplorationPolicy(ABC):
    """Abstract exploration policy."""

    @abstractmethod
    def score(
        self, posterior: Posterior, x: np.ndarray, t: int, rng: np.random.Generator
    ) -> float:
        """
        Score one arm.

        Parameters
        ----------
        posterior : Posterior
            The arm's posterior.
        x : np.ndarray
            Model input of the current instance.
        t : int
            Number of observations so far (the bandit's clock).
        rng : np.random.Generator
            Source of randomness.
        """


class UCBPolicy(ExplorationPolicy):
    """
    Optimism in the face of uncertainty: ``mean + bonus * width``.

    Parameters
    ----------
    alpha : float, default=1.0
        Exploration strength.
    ucb1 : bool, default=False
        Multiply the bonus by ``sqrt(2 ln t)``. With a ``GaussianPosterior``
        (``prior_precision=0``, ``noise_var=1``) and ``alpha=1`` this is the
        classic UCB1 index ``mean + sqrt(2 ln t / n)``. Without it, the bonus
        is constant in ``t``, as in LinUCB (Li et al., 2010).
    """

    def __init__(self, alpha: float = 1.0, ucb1: bool = False) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = float(alpha)
        self.ucb1 = bool(ucb1)

    def score(
        self, posterior: Posterior, x: np.ndarray, t: int, rng: np.random.Generator
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
    Thompson Sampling: score = one draw from the posterior.

    Parameters
    ----------
    scale : float, default=1.0
        Inflates the posterior's standard deviation (``v`` in Agrawal & Goyal's
        LinTS); larger means more exploration.
    """

    def __init__(self, scale: float = 1.0) -> None:
        if scale < 0:
            raise ValueError("scale must be >= 0.")
        self.scale = float(scale)

    def score(
        self, posterior: Posterior, x: np.ndarray, t: int, rng: np.random.Generator
    ) -> float:
        return posterior.sample(x, rng, self.scale)
