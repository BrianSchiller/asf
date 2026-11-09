"""UCB baselines: UCB1, LinUCB and the non-stationary DS-UCB and SW-UCB."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Hashable

import numpy as np

from asf.online.selectors.bandits.base import AbstractContextualBandit
from asf.online.selectors.bandits.policies import UCBPolicy
from asf.online.selectors.bandits.posteriors import (
    BayesianLinearRegression,
    GaussianPosterior,
)
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.weighting import ExponentialDiscount, SlidingWindow


class UCBSelector(AbstractContextualBandit):
    """
    Context-free UCB (UCB1 by default) over algorithms.

    Score ``mean + alpha * sqrt(noise_var / n) * sqrt(2 ln t)``. With the
    defaults this is the classic UCB1 index; unplayed arms are tried first.

    Parameters
    ----------
    alpha : float, default=1.0
        Exploration strength.
    ucb1 : bool, default=True
        Use the ``sqrt(2 ln t)`` factor of UCB1.
    prior_precision : float, default=0.0
        Pseudo-count of the zero-mean prior (0 = none).
    noise_var : float, default=1.0
        Reward noise variance.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit` (e.g. ``weighting``,
        ``partitioner``, ``budget``, ``random_state``).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        ucb1: bool = True,
        prior_precision: float = 0.0,
        noise_var: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            posterior=partial(
                GaussianPosterior, prior_precision=prior_precision, noise_var=noise_var
            ),
            policy=UCBPolicy(alpha=alpha, ucb1=ucb1),
            contextual=False,
            **kwargs,
        )


class LinUCBSelector(AbstractContextualBandit):
    """
    LinUCB with disjoint linear models (Li et al., 2010).

    Score ``x @ theta_a + alpha * sqrt(noise_var * x^T A_a^-1 x)`` with
    ``x = [standardised features, 1]``.

    Parameters
    ----------
    alpha : float, default=1.0
        Exploration strength.
    prior_precision : float, default=1.0
        Ridge regularisation ``lambda``.
    noise_var : float, default=1.0
        Reward noise variance.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prior_precision: float = 1.0,
        noise_var: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            posterior=partial(
                BayesianLinearRegression,
                prior_precision=prior_precision,
                noise_var=noise_var,
            ),
            policy=UCBPolicy(alpha=alpha, ucb1=False),
            contextual=True,
            **kwargs,
        )


class DiscountedUCBSelector(UCBSelector):
    """
    Discounted UCB (D-UCB, Garivier & Moulines, 2011).

    Index ``mu_hat + 2 B sqrt(xi ln n_t / N_t)`` with the discounted play count
    ``N_t = sum_s gamma^(t-s) [arm played at s]``, the discounted mean
    ``mu_hat`` and ``n_t = sum_arms N_t``. Unplayed arms are tried first.

    Rewards default to the ``"unit"`` reward in ``[0, 1]`` (so ``B = 1``); pass
    ``budget=<cutoff>``. The framework weights the newest observation by 1,
    the paper by ``gamma``; this is corrected internally.

    Parameters
    ----------
    gamma : float, default=0.99
        Discount factor in ``(0, 1]``.
    B : float, default=1.0
        Upper bound on the rewards.
    xi : float, default=0.6
        Exploration constant.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self, gamma: float = 0.99, B: float = 1.0, xi: float = 0.6, **kwargs: Any
    ) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        kwargs.setdefault("reward", RewardTransform("unit"))
        # alpha * width * sqrt(2 ln n) with width = sqrt(gamma / N_framework)
        # equals 2 B sqrt(xi ln n / N_paper).
        super().__init__(
            alpha=2.0 * B * math.sqrt(xi / 2.0),
            ucb1=True,
            prior_precision=0.0,
            noise_var=1.0 / gamma,
            weighting=ExponentialDiscount(gamma),
            **kwargs,
        )
        self.gamma = float(gamma)

    def _policy_time(self, z: np.ndarray, cell: Hashable) -> float:
        """``n_t(gamma)``: the discounted total play count (paper convention)."""
        return self.gamma * sum(
            self._posterior_for(arm, z, cell).n_effective for arm in self.algorithms
        )


class SlidingWindowUCBSelector(UCBSelector):
    """
    Sliding-Window UCB (SW-UCB, Garivier & Moulines, 2011).

    Index ``mu_hat + B sqrt(xi ln min(t, window) / N_t)`` with ``N_t`` and
    ``mu_hat`` computed over the last ``window`` steps; ``t`` is the
    (1-based) current round. Unplayed arms are tried first.

    Rewards default to the ``"unit"`` reward in ``[0, 1]`` (so ``B = 1``); pass
    ``budget=<cutoff>``.

    Parameters
    ----------
    window : int, default=100
        Window length ``tau`` in steps, shared by all arms.
    B : float, default=1.0
        Upper bound on the rewards.
    xi : float, default=0.6
        Exploration constant.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self, window: int = 100, B: float = 1.0, xi: float = 0.6, **kwargs: Any
    ) -> None:
        kwargs.setdefault("reward", RewardTransform("unit"))
        super().__init__(
            alpha=B * math.sqrt(xi / 2.0),
            ucb1=True,
            prior_precision=0.0,
            noise_var=1.0,
            weighting=SlidingWindow(window),
            **kwargs,
        )
        self.window = int(window)

    def _policy_time(self, z: np.ndarray, cell: Hashable) -> float:
        """``min(t, window)`` with the 1-based current round ``t``."""
        return float(min(self.clock_ + 1, self.window))
