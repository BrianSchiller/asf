"""UCB baselines: context-free UCB and LinUCB."""

from __future__ import annotations

from functools import partial
from typing import Any

from asf.online.selectors.bandits.base import AbstractContextualBandit
from asf.online.selectors.bandits.policies import UCBPolicy
from asf.online.selectors.bandits.posteriors import (
    BayesianLinearRegression,
    GaussianPosterior,
)


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
