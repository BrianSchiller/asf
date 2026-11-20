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
    Context-free Gaussian Thompson Sampling (Agrawal & Goyal, 2013);
    ``scale`` inflates the posterior (exploration strength).
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
    Linear Thompson Sampling with disjoint models (Agrawal & Goyal, 2013):
    per arm ``theta_a ~ N(theta_hat_a, scale^2 * noise_var * A_a^-1)``, play
    ``argmax_a x @ theta_a``.
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

    Per arm ``theta ~ N(mu_hat, c * sigma^2 / N)`` with the play count ``N``
    and mean ``mu_hat`` discounted by ``gamma`` every round, so rarely played
    arms are re-explored (``c``: 4 in the analysis, 1 in practice). The paper
    weights the newest observation by ``gamma``, so ``noise_var`` is divided
    by ``gamma`` internally.
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
    Sliding-Window Thompson Sampling (SW-TS): ``theta ~ N(mu_hat, c * sigma^2
    / N)`` over the last ``window`` steps (shared by all arms).
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

    Two Beta posteriors per arm: a historic trace discounted by ``gamma``
    every round and a hot trace of the arm's last ``window`` pulls. One
    sample from each is combined by ``aggregate``. Rewards are the ``"unit"``
    reward (log PAR10 in ``[0, 1]``) with fractional Beta updates.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        window: int = 30,
        aggregate: Aggregate = "mean",
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

    Each instance goes to the nearest :class:`OnlineKMeans` centroid; that
    cluster's Beta posteriors pick the arm; the played arm is discounted by
    ``gamma`` and updated with the reward times the fuzzy penalty ``phi``;
    then the centroid moves (discount ``cluster_gamma``, default ``gamma``).

    Differences to the paper: the centroid moves in ``observe``, not
    ``propose``; the reward is the fractional ``"unit"`` reward (use
    ``RewardTransform("solved")`` for a binary one); the Beta prior is never
    discounted. ``per_step_discount=True`` decays every arm of every cluster
    each step, as in DS-TS (paper: only the updated arm). ``posterior`` and
    ``contextual`` swap the per-arm model, e.g. LinTS per cluster.
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
