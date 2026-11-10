"""Thompson Sampling baselines.

Stationary: Gaussian TS and LinTS. Non-stationary (context-free): DS-TS,
SW-TS and f-dsw TS. Clustered: DTS-OC.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from asf.online.base import Observation
from asf.online.selectors.bandits.base import (
    Aggregate,
    AbstractContextualBandit,
    PosteriorFactory,
)
from asf.online.selectors.bandits.partitioning import OnlineKMeans
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
    PullDiscount,
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


class DTSOCSelector(AbstractContextualBandit):
    """
    DTS-OC: discounted Beta Thompson Sampling per online cluster.

    Per instance:

    1. Assign it to the nearest centroid of an :class:`OnlineKMeans`
        partitioner (distances on the standardised features).
    2. Draw one Beta sample per algorithm from that cluster's posteriors and
        run the algorithm with the largest sample.
    3. Update the played arm of that cluster with the reward scaled by the
        fuzzy penalty ``phi`` (computed before the centroid moves), after
        discounting its counts by ``gamma``; then move the centroid.

    Differences to the paper: the centroid moves in ``observe`` instead of
    ``propose`` (so ``propose`` has no side effects); rewards are the
    fractional ``"unit"`` reward instead of Bernoulli (use
    ``reward=RewardTransform("solved")`` for a binary one); the Beta prior is
    added at sampling time and never discounted.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters ``K``.
    gamma : float, default=0.95
        Discount of the bandit counts.
    cluster_gamma : float or None, default=None
        Discount of the centroid statistics (``None`` = ``gamma``).
    fuzziness : float, default=1.3
        Fuzziness exponent ``l > 1`` of the penalty; values close to 1 make
        ``phi`` close to 1 for all but truly ambiguous instances.
    per_step_discount : bool, default=False
        ``False`` (paper): only the updated arm of the assigned cluster is
        discounted. ``True``: every arm in every cluster decays every step
        (as in DS-TS), so arms that are not played are re-explored.
    posterior : Callable[[int], Posterior] or None, default=None
        Per-arm model; ``None`` = :class:`BetaPosterior`. E.g.
        ``partial(BayesianLinearRegression, ...)`` with ``contextual=True``
        and a ``"log"`` reward gives LinTS per cluster.
    contextual : bool, default=False
        Feed the features to the per-arm model (needs a linear posterior).
    **kwargs : Any
        Forwarded to :class:`AbstractContextualBandit` (e.g. ``budget``,
        ``random_state``, ``reward``, ``confidence``).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        gamma: float = 0.95,
        cluster_gamma: float | None = None,
        fuzziness: float = 1.3,
        per_step_discount: bool = False,
        posterior: PosteriorFactory | None = None,
        contextual: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("reward", RewardTransform("unit"))
        partitioner = OnlineKMeans(
            n_clusters,
            gamma=cluster_gamma if cluster_gamma is not None else gamma,
            fuzziness=fuzziness,
            random_state=kwargs.get("random_state"),
        )
        weighting = (
            ExponentialDiscount(gamma) if per_step_discount else PullDiscount(gamma)
        )
        super().__init__(
            posterior=posterior if posterior is not None else BetaPosterior,
            policy=ThompsonPolicy(),
            contextual=contextual,
            weighting=weighting,
            partitioner=partitioner,
            **kwargs,
        )

    def _observation_weight(
        self, z: np.ndarray, observation: Observation | None
    ) -> float:
        """Fuzzy penalty ``phi`` (times a confidence score, if one is set)."""
        assert isinstance(self.partitioner, OnlineKMeans)
        phi = self.partitioner.membership(z)
        return phi * super()._observation_weight(z, observation)

