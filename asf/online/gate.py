"""
Extra-run gates: decide, per instance, which additional algorithms to run.

Running more than the selected algorithm costs compute time but yields
performance information the selector can learn from. A gate makes that
decision from the selector's prediction (including its uncertainty), the
selector's nominees and, optionally, a detector's per-instance score.

Every :class:`~asf.online.base.AbstractOnlineSelector` owns one gate
(``gate=...``); without one it uses the selector's default, which reproduces
the selector's original behaviour (Rook selector: all ``n_additional``
nominees; bandits: none).

:class:`ScoreGate` is the base of all criterion gates: a subclass rates each
instance with one number ("how much reason is there for an extra run?") and
the base class turns it into a decision, either with a fixed ``threshold``
or with a ``rate`` ("run extras on the q% highest-rated recent instances").
:class:`UncertaintyGate` rates by the selector's uncertainty; further
criteria are in :mod:`asf.online.gate_criteria`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from asf.online.drift.abstract_detector import DriftSignal
from asf.online.prediction import PredictionRecord

if TYPE_CHECKING:
    from asf.online.base import Observation


@dataclass(frozen=True)
class GateContext:
    """
    Everything a gate may base its decision on.

    Attributes
    ----------
    prediction : PredictionRecord
        The selector's decision for this instance (``extras`` still empty).
    candidates : tuple[str, ...]
        Algorithms the gate may choose from, most useful first (the
        selector's nominees, ``prediction.candidates``).
    algorithms : tuple[str, ...]
        All algorithms of the portfolio.
    features : pd.DataFrame
        The single-row feature frame of the instance.
    step : int
        0-based index of the instance in the evaluated stream.
    instance_score : float or None
        Per-instance unfamiliarity score of the selector's feature detector
        (higher = less familiar). Only computed if the gate sets
        ``needs_instance_score`` and the detector is fitted.
    last_signal : DriftSignal or None
        The detector's output on the previous instance.
    steps_since_drift : int or None
        Instances since the last drift signal (``None`` = no signal yet).
    """

    prediction: PredictionRecord
    candidates: tuple[str, ...]
    algorithms: tuple[str, ...]
    features: pd.DataFrame
    step: int
    instance_score: float | None = None
    last_signal: DriftSignal | None = None
    steps_since_drift: int | None = None

    def ranked_alternatives(self) -> tuple[str, ...]:
        """
        All algorithms except the chosen one, most promising first.

        The selector's nominees come first (in their order), then the
        remaining algorithms by descending ``prediction.scores`` (portfolio
        order for ties or without scores). Useful for gates that pick extras
        themselves, e.g. for selectors that nominate nothing (bandits).
        """
        chosen = self.prediction.chosen
        head = [a for a in self.candidates if a != chosen]
        rest = [a for a in self.algorithms if a != chosen and a not in head]
        scores = self.prediction.scores
        if scores is not None:
            order = {a: i for i, a in enumerate(self.algorithms)}
            rest.sort(key=lambda a: (-float(scores.get(a, float("-inf"))), order[a]))
        return tuple(head + rest)


class AbstractExtraRunGate(ABC):
    """
    Decides which extra algorithms to run for one instance.

    Subclasses implement :meth:`extras`. They must not change the selector;
    any state they keep (e.g. a running threshold) is their own.

    Attributes
    ----------
    needs_instance_score : bool
        If ``True``, the selector computes its feature detector's
        per-instance score and passes it as ``context.instance_score``.
    """

    needs_instance_score: bool = False

    @abstractmethod
    def extras(self, context: GateContext) -> Sequence[str]:
        """
        Return the extra algorithms to run, in order.

        The selector drops duplicates and the primary choice and rejects
        unknown algorithms. Returning algorithms outside
        ``context.candidates`` is allowed.
        """

    def observe(self, prediction: PredictionRecord, observation: "Observation") -> None:
        """
        See what happened on the instance the gate last decided on.

        Called by the selector once per instance, after the scheduled
        algorithms ran. Gates that track what their extras cost or yielded
        (e.g. a compute budget) override it; the default does nothing.

        Parameters
        ----------
        prediction : PredictionRecord
            The final record of the instance; ``prediction.extras`` are the
            extra algorithms that were scheduled.
        observation : Observation
            Runtimes and censoring of all executed algorithms.
        """

    def reset(self) -> None:
        """
        Forget all state. Called when the selector is fitted; the default
        does nothing.
        """


class NeverProbe(AbstractExtraRunGate):
    """Never run extra algorithms (the bandits' default)."""

    def extras(self, context: GateContext) -> Sequence[str]:
        return ()


class AlwaysProbe(AbstractExtraRunGate):
    """
    Always run the first ``n`` nominees (all of them for ``n=None``).

    With ``n=None`` this is the default of selectors that nominate a fixed
    number of extras, such as the Rook selector (``n_additional``).

    Parameters
    ----------
    n : int or None, default=None
        Number of nominees to run; ``None`` = all.
    """

    def __init__(self, n: int | None = None) -> None:
        if n is not None and n < 0:
            raise ValueError("n must be >= 0 or None.")
        self.n = None if n is None else int(n)

    def extras(self, context: GateContext) -> Sequence[str]:
        if self.n is None:
            return context.candidates
        return context.candidates[: self.n]


class ScoreGate(AbstractExtraRunGate):
    """
    Base class of gates that rate every instance with one number.

    Subclasses implement :meth:`_score`: how much reason there is to run an
    extra algorithm on this instance (higher = more; ``None`` = cannot say).
    The decision rule is shared; exactly one of ``threshold`` and ``rate``
    must be given:

    * ``threshold``: fire if ``score >= threshold``. Instances without a
      score never fire. The scale of the threshold is the criterion's, so
      thresholds are not comparable across selectors or criteria.
    * ``rate`` (``q`` in ``[0, 1]``): fire on the share ``q`` of instances
      with the highest scores among the last ``window_size`` scored instances
      (the current one included). With ``g`` earlier scores in the window
      greater than the current score, ``e`` equal to it and ``k`` in the
      window in total, the gate fires if ``(g + u * (e + 1)) / (k + 1) < q``
      with ``u`` uniform in ``[0, 1)``. That is "the score is above the
      ``1 - q`` quantile of the recent scores", with ties broken at random
      so that discrete scores (vote gaps) also give a rate of ``q``.
      Instances without a score fire with probability ``q``. ``q = 1``
      always fires and ``q = 0`` never does.

    Under a rate, the share of instances with extras is ``q`` as long as the
    scores are stable. When they rise (e.g. after a drift) the gate fires
    more often until the window has turned over.

    When the gate fires it runs the first ``n`` nominees of the selector; if
    the selector nominates none (bandits), the first ``n`` of
    :meth:`GateContext.ranked_alternatives`.

    Parameters
    ----------
    threshold : float or None, default=None
        Minimum score for extra runs.
    n : int or None, default=1
        Number of extra algorithms to run when the gate fires; ``None`` =
        all nominees (all other algorithms if there are no nominees).
    rate : float or None, default=None
        Share ``q`` of instances that get extra runs, in ``[0, 1]``.
    window_size : int, default=200
        Number of recent scores a rate is taken from.
    random_state : int or None, default=None
        Seed of the tie-breaking under a rate.

    Attributes
    ----------
    last_score_ : float or None
        Score of the last instance decided on.
    """

    def __init__(
        self,
        threshold: float | None = None,
        n: int | None = 1,
        *,
        rate: float | None = None,
        window_size: int = 200,
        random_state: int | None = None,
    ) -> None:
        if (threshold is None) == (rate is None):
            raise ValueError("Give exactly one of threshold and rate.")
        if rate is not None and not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1].")
        if n is not None and n < 0:
            raise ValueError("n must be >= 0 or None.")
        if window_size < 1:
            raise ValueError("window_size must be >= 1.")
        self.threshold = None if threshold is None else float(threshold)
        self.n = None if n is None else int(n)
        self.rate = None if rate is None else float(rate)
        self.window_size = int(window_size)
        self.random_state = random_state
        self.reset()

    @abstractmethod
    def _score(self, context: GateContext) -> float | None:
        """Reason for an extra run on this instance (higher = more)."""

    def reset(self) -> None:
        """Forget the recent scores and restart the random stream."""
        self._window: deque[float] = deque(maxlen=self.window_size)
        self._rng = np.random.default_rng(self.random_state)
        self.last_score_: float | None = None

    def extras(self, context: GateContext) -> Sequence[str]:
        score = self._score(context)
        if score is not None:
            score = float(score)
            if np.isnan(score):
                score = None
        self.last_score_ = score
        if not self._fires(score):
            return ()
        pool = context.candidates or context.ranked_alternatives()
        return pool if self.n is None else pool[: self.n]

    def _fires(self, score: float | None) -> bool:
        if self.rate is None:
            assert self.threshold is not None
            return score is not None and score >= self.threshold
        u = float(self._rng.random())  # one draw per instance, always
        if score is None:
            return u < self.rate
        past = np.fromiter(self._window, dtype=float, count=len(self._window))
        greater = int((past > score).sum())
        equal = int((past == score).sum()) + 1
        self._window.append(score)
        return (greater + u * equal) / (len(past) + 1) < self.rate


class UncertaintyGate(ScoreGate):
    """
    Run extras when the selector is uncertain.

    The score is ``prediction.uncertainty``. With a ``threshold`` this is
    stream-based active learning: an extra run is "queried" only if
    ``prediction.uncertainty >= threshold`` (Kuş et al., "Frugal Algorithm
    Selection"), and predictions without an uncertainty never trigger extra
    runs. With a ``rate`` the extras go to the most uncertain share of the
    recent instances, which is comparable across selectors.

    Parameters
    ----------
    threshold : float or None, default=None
        Minimum uncertainty for extra runs. Its scale is the selector's (e.g.
        least confidence in ``[0, 0.5]`` for the frugal pairwise selector).
    n, rate, window_size, random_state
        See :class:`ScoreGate`.
    """

    def _score(self, context: GateContext) -> float | None:
        return context.prediction.uncertainty
