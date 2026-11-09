"""Thompson Sampling baselines.

Stationary: Gaussian TS and LinTS. Non-stationary (context-free): DS-TS,
SW-TS and f-dsw TS.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from asf.online.selectors.bandits.base import Aggregate, AbstractContextualBandit
from asf.online.selectors.bandits.policies import ThompsonPolicy
from asf.online.selectors.bandits.posteriors import (
    BayesianLinearRegression,
    BetaPosterior,
    GaussianPosterior,
)
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.weighting import (
    ExponentialDiscount,
    LastPulls,
    SlidingWindow,
)


class ThompsonSamplingSelector(AbstractContextualBandit):
    """
    Context-free Gaussian Thompson Sampling (Agrawal & Goyal, 2013).

    Parameters
    ----------
    scale : float, default=1.0
        Posterior inflation (exploration strength).
    prior_precision : float, default=1.0
        Pseudo-count of the zero-mean prior.
    noise_var : float, default=1.0
        Reward noise variance.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit` (e.g. ``weighting``,
        ``partitioner``, ``confidence``, ``budget``, ``random_state``).
    """

    def __init__(
        self,
        scale: float = 1.0,
        prior_precision: float = 1.0,
        noise_var: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            posterior=partial(
                GaussianPosterior, prior_precision=prior_precision, noise_var=noise_var
            ),
            policy=ThompsonPolicy(scale=scale),
            contextual=False,
            **kwargs,
        )


class LinTSSelector(AbstractContextualBandit):
    """
    Linear Thompson Sampling with disjoint models (Agrawal & Goyal, 2013).

    Each step draws ``theta_a ~ N(theta_hat_a, scale^2 * noise_var * A_a^-1)``
    per arm and plays ``argmax_a x @ theta_a``.

    Parameters
    ----------
    scale : float, default=1.0
        Posterior inflation ``v``.
    prior_precision : float, default=1.0
        Ridge regularisation ``lambda``.
    noise_var : float, default=1.0
        Reward noise variance.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self,
        scale: float = 1.0,
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
            policy=ThompsonPolicy(scale=scale),
            contextual=True,
            **kwargs,
        )


class DiscountedTSSelector(ThompsonSamplingSelector):
    """
    Discounted Thompson Sampling (DS-TS).

    Per arm, ``theta ~ N(mu_hat, c * sigma^2 / N)`` with the discounted play
    count ``N = sum_j gamma^(t-j) [arm played at j]`` and the discounted mean
    ``mu_hat``. All arms decay every round, so the variance of rarely played
    arms grows and they are re-explored.

    The paper weights the newest observation by ``gamma``, the framework by 1;
    ``noise_var`` is divided by ``gamma`` internally so the posterior is
    exactly the paper's.

    Parameters
    ----------
    gamma : float, default=0.99
        Discount factor in ``(0, 1]``.
    noise_var : float, default=1.0
        Reward variance ``sigma^2``.
    c : float, default=1.0
        Variance inflation (4 in the theoretical analysis, 1 in practice).
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        noise_var: float = 1.0,
        c: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1].")
        super().__init__(
            scale=c**0.5,
            prior_precision=0.0,
            noise_var=noise_var / gamma,
            weighting=ExponentialDiscount(gamma),
            **kwargs,
        )


class SlidingWindowTSSelector(ThompsonSamplingSelector):
    """
    Sliding-Window Thompson Sampling (SW-TS).

    Per arm, ``theta ~ N(mu_hat, c * sigma^2 / N)`` where ``N`` and ``mu_hat``
    are computed over the last ``window`` global steps only.

    Parameters
    ----------
    window : int, default=100
        Window length ``tau`` in steps (instances), shared by all arms.
    noise_var : float, default=1.0
        Reward variance ``sigma^2``.
    c : float, default=1.0
        Variance inflation.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self,
        window: int = 100,
        noise_var: float = 1.0,
        c: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scale=c**0.5,
            prior_precision=0.0,
            noise_var=noise_var,
            weighting=SlidingWindow(window),
            **kwargs,
        )


class FDSWTSSelector(AbstractContextualBandit):
    """
    f-Discounted-Sliding-Window Thompson Sampling (f-dsw TS).

    Every arm keeps two Beta posteriors over a reward in ``[0, 1]``:

    * historic trace -- all observations, discounted by ``gamma`` per step
      (all arms decay every round);
    * hot trace -- only the last ``window`` pulls of that arm.

    One sample is drawn from each and combined with ``aggregate``
    (``"min"``, ``"max"`` or ``"mean"``); the arm with the largest combined
    sample is played. Rewards are the ``"unit"`` reward (log-scaled PAR10 in
    ``[0, 1]``) with fractional Beta updates.

    Parameters
    ----------
    gamma : float, default=0.99
        Discount factor of the historic trace.
    window : int, default=30
        Per-arm window size ``n`` of the hot trace.
    aggregate : {"mean", "min", "max"} or callable, default="mean"
        The aggregation function ``f``.
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit`.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        window: int = 30,
        aggregate: str | Aggregate = "mean",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("reward", RewardTransform("unit"))
        super().__init__(
            posterior=BetaPosterior,
            policy=ThompsonPolicy(),
            contextual=False,
            traces={
                "historic": ExponentialDiscount(gamma),
                "hot": LastPulls(window),
            },
            aggregate=aggregate,
            **kwargs,
        )
