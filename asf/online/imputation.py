"""
Imputation of right-censored runtimes (Tornede et al., 2022).

A run stopped at its budget only tells us that the true value ``y`` is at
least ``bound``. These rules replace it by a number (or drop it), given the
selector's belief ``N(mean, sd**2)`` on its model's scale:

* :class:`CutoffImputer` -- ``bound`` (the "impute by the cutoff" rule).
* :class:`DropCensored` -- ``NaN``, i.e. ignore the run (BlindUCB).
* :class:`ExpectedImputer` -- ``E[y | y >= bound]`` (Buckley and James 1979;
  one step of Schmee and Hahn 1979).
* :class:`SampledImputer` -- a draw from the belief above ``bound``
  (Algorithm 1, lines 17-20 of Tornede et al., and their code).

Reference
---------
Tornede, A., Bengs, V., Huellermeier, E. (2022). "Machine Learning for Online
Algorithm Selection under Censored Feedback". AAAI 2022.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


class CensoredImputer(ABC):
    """
    Rule that fills in right-censored values (true value ``>= bound``).

    ``USES_MODEL``: the rule needs ``mean`` and ``sd``. ``PREDICTIVE``: ``sd``
    should be that of a new observation (model uncertainty plus noise), not
    of the mean prediction.
    """

    USES_MODEL: bool = True
    PREDICTIVE: bool = False

    @abstractmethod
    def __call__(
        self,
        bound: ArrayLike,
        mean: ArrayLike,
        sd: ArrayLike,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Imputed values (at least ``bound``; ``NaN`` = drop). ``sd >= 0``,
        may be ``inf`` (no belief yet)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CutoffImputer(CensoredImputer):
    """Impute the bound itself (eq. 11 of Tornede et al.; biased downwards)."""

    USES_MODEL = False

    def __call__(
        self,
        bound: ArrayLike,
        mean: ArrayLike,
        sd: ArrayLike,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.array(bound, dtype=float, ndmin=1)


class DropCensored(CensoredImputer):
    """Ignore censored values (eq. 9 of Tornede et al., BlindUCB)."""

    USES_MODEL = False

    def __call__(
        self,
        bound: ArrayLike,
        mean: ArrayLike,
        sd: ArrayLike,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.full(np.shape(np.array(bound, ndmin=1)), np.nan)


class ExpectedImputer(CensoredImputer):
    """
    Impute ``E[y | y >= bound]`` of ``N(mean, sd**2)``, the truncated mean::

        mean + sd * phi(a) / (1 - Phi(a)),    a = (bound - mean) / sd

    (computed in log space). Buckley and James (1979), as described by
    Tornede et al.; one step of Schmee and Hahn (1979). ``sd == 0`` gives
    ``max(mean, bound)``, ``sd == inf`` gives ``bound``.
    """

    PREDICTIVE = True

    def __call__(
        self,
        bound: ArrayLike,
        mean: ArrayLike,
        sd: ArrayLike,
        rng: np.random.Generator,
    ) -> np.ndarray:
        bound, mean, sd = _broadcast(bound, mean, sd)
        out = np.maximum(mean, bound)
        out[~np.isfinite(sd)] = bound[~np.isfinite(sd)]
        spread = np.isfinite(sd) & (sd > 0)
        if spread.any():
            m, s = mean[spread], sd[spread]
            a = (bound[spread] - m) / s
            ratio = np.exp(norm.logpdf(a) - norm.logsf(a))  # inverse Mills ratio
            out[spread] = np.maximum(m + s * ratio, bound[spread])
        return out


class SampledImputer(CensoredImputer):
    """
    Impute a draw from ``N(mean, sd**2)`` above ``bound``.

    Rejection sampling as in Tornede et al. (Algorithm 1, lines 17-20, and
    their code): the first of ``max_tries`` draws above ``bound``, else
    ``bound``. Their ``sd`` is that of the mean prediction, hence
    ``PREDICTIVE = False``. ``sd == inf`` gives ``bound``.
    """

    def __init__(self, max_tries: int = 20) -> None:
        if max_tries < 1:
            raise ValueError("max_tries must be >= 1.")
        self.max_tries = int(max_tries)

    def __call__(
        self,
        bound: ArrayLike,
        mean: ArrayLike,
        sd: ArrayLike,
        rng: np.random.Generator,
    ) -> np.ndarray:
        bound, mean, sd = _broadcast(bound, mean, sd)
        finite_sd = np.where(np.isfinite(sd), sd, 0.0)
        draws = mean[:, None] + finite_sd[:, None] * rng.standard_normal(
            (len(bound), self.max_tries)
        )
        accepted = draws > bound[:, None]
        first = accepted.argmax(axis=1)
        out = np.where(accepted.any(axis=1), draws[np.arange(len(bound)), first], bound)
        out[~np.isfinite(sd)] = bound[~np.isfinite(sd)]
        return out

    def __repr__(self) -> str:
        return f"SampledImputer(max_tries={self.max_tries})"


def _broadcast(
    bound: ArrayLike, mean: ArrayLike, sd: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float arrays of a common 1-d shape; ``sd`` must be non-negative."""
    arrays = np.broadcast_arrays(
        np.array(bound, dtype=float, ndmin=1),
        np.array(mean, dtype=float, ndmin=1),
        np.array(sd, dtype=float, ndmin=1),
    )
    b, m, s = (np.array(a, dtype=float).reshape(-1) for a in arrays)
    if np.any(s < 0):
        raise ValueError("sd must be >= 0.")
    return b, m, s
