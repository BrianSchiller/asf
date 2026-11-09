"""Thompson Sampling baselines: Gaussian TS and LinTS."""

from __future__ import annotations

from functools import partial
from typing import Any

from asf.online.selectors.bandits.base import AbstractContextualBandit
from asf.online.selectors.bandits.policies import ThompsonPolicy
from asf.online.selectors.bandits.posteriors import (
    BayesianLinearRegression,
    GaussianPosterior,
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
