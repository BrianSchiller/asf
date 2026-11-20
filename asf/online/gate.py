"""
Extra-run gates: decide, per instance, which additional algorithms to run.

Extra runs cost compute but give the selector more to learn from. Every
online selector owns one gate; its default reproduces the selector's original
behaviour (Rook: all ``n_additional`` nominees, bandits: none).

:class:`ScoreGate` rates each instance with one number and fires either above
a ``threshold`` or on the ``rate`` share of highest-rated recent instances.
Criteria: :class:`UncertaintyGate`, :class:`CloseCallGate`,
:class:`UnfamiliarityGate`, :class:`TimeoutRiskGate`.
"""

from __future__ import annotations

import math
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
        The selector's decision (``extras`` still empty).
    step : int
        0-based index of the instance in the evaluated stream.
    instance_score : float or None
        Unfamiliarity score of the selector's feature detector (higher = less
        familiar); only computed if the gate sets ``needs_instance_score``.
    last_signal : DriftSignal or None
        The detector's output on the previous instance.
    steps_since_drift : int or None
        Instances since the last drift signal (``None`` = none yet).
    """

    prediction: PredictionRecord
    algorithms: tuple[str, ...]
    features: pd.DataFrame
    step: int
    instance_score: float | None = None
    last_signal: DriftSignal | None = None
    steps_since_drift: int | None = None

    @property
    def candidates(self) -> tuple[str, ...]:
        """The selector's nominees for extra runs, most useful first."""
        return self.prediction.candidates

    def ranked_alternatives(self) -> tuple[str, ...]:
        """
        All algorithms except the chosen one: the nominees first, then the
        rest by descending ``prediction.scores`` (portfolio order on ties).
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

    A gate must not change the selector. With ``needs_instance_score = True``
    the selector passes its feature detector's score as
    ``context.instance_score``.
    """

    needs_instance_score: bool = False

    @abstractmethod
    def extras(self, context: GateContext) -> Sequence[str]:
        """
        The extra algorithms to run, in order. Duplicates and the primary
        choice are dropped by the selector; non-nominees are allowed.
        """

    def observe(self, prediction: PredictionRecord, observation: "Observation") -> None:
        """Called after the scheduled runs, e.g. to track a compute budget."""

    def reset(self) -> None:
        """Forget all state (called when the selector is fitted)."""


class NoExtrasGate(AbstractExtraRunGate):
    """Never run extra algorithms (the bandits' default)."""

    def extras(self, context: GateContext) -> Sequence[str]:
        return ()


class AfterDriftGate(AbstractExtraRunGate):
    """
    Run every algorithm on the first instance after a drift signal; ``gate``
    (default: no extras) decides on all other instances.

    One full round right after a change gives the selector a comparison of
    all algorithms on the new data (for a pairwise selector: every pair).
    """

    def __init__(self, gate: AbstractExtraRunGate | None = None) -> None:
        self.gate = gate if gate is not None else NoExtrasGate()

    @property
    def needs_instance_score(self) -> bool:  # type: ignore[override]
        return self.gate.needs_instance_score

    def extras(self, context: GateContext) -> Sequence[str]:
        if context.steps_since_drift == 1:  # the signal came on the previous instance
            return context.ranked_alternatives()
        return self.gate.extras(context)

    def observe(self, prediction: PredictionRecord, observation: "Observation") -> None:
        self.gate.observe(prediction, observation)

    def reset(self) -> None:
        self.gate.reset()


class AllNomineesGate(AbstractExtraRunGate):
    """
    Always run the first ``n`` nominees (``None`` = all; the default of
    selectors with a fixed number of extras, such as Rook's ``n_additional``).
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
    Base of gates that rate every instance with one number.

    Subclasses implement :meth:`_score` (higher = more reason for an extra
    run, ``None`` = no rating). Exactly one decision rule is given:

    * ``threshold``: fire if ``score >= threshold`` (unrated instances never
      fire). The scale is the criterion's.
    * ``rate`` ``q``: fire on the share ``q`` of highest scores among the last
      ``window_size`` rated instances. With ``g`` earlier scores greater,
      ``e`` equal and ``k`` in total, fire if ``(g + u (e + 1)) / (k + 1) < q``
      with ``u ~ U[0, 1)``, so ties (e.g. vote gaps) still give rate ``q``.
      Unrated instances fire with probability ``q``. After a drift, scores
      rise and the gate fires more often until the window has turned over.

    When it fires it runs the first ``n`` nominees (``None`` = all); without
    nominees (bandits) the first ``n`` of :meth:`GateContext.ranked_alternatives`.
    ``random_state`` seeds the tie-breaking; ``last_score_`` is the last rating.
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
    Run extras when the selector is uncertain (``prediction.uncertainty``).

    With a ``threshold`` this is stream-based active learning as in Kuş et
    al., "Frugal Algorithm Selection"; the scale is the selector's (e.g. least
    confidence in ``[0, 0.5]`` for the frugal pairwise selector).
    """

    def _score(self, context: GateContext) -> float | None:
        return context.prediction.uncertainty


class CloseCallGate(ScoreGate):
    """
    Run extras when the two best entries of ``prediction.scores`` are nearly
    tied (votes, predicted performance, bandit index; non-finite ignored).

    ``max_gap`` replaces ``threshold``: fire if the gap is at most ``max_gap``.
    ``relative=True`` divides the gap by the larger absolute score, e.g. to
    compare predicted runtimes of easy and hard instances.
    """

    def __init__(
        self,
        max_gap: float | None = None,
        n: int | None = 1,
        *,
        relative: bool = False,
        rate: float | None = None,
        window_size: int = 200,
        random_state: int | None = None,
    ) -> None:
        if max_gap is not None and max_gap < 0:
            raise ValueError("max_gap must be >= 0.")
        super().__init__(
            threshold=None if max_gap is None else -float(max_gap),
            n=n,
            rate=rate,
            window_size=window_size,
            random_state=random_state,
        )
        self.max_gap = None if max_gap is None else float(max_gap)
        self.relative = bool(relative)

    def _score(self, context: GateContext) -> float | None:
        scores = context.prediction.scores
        if scores is None:
            return None
        values = sorted(
            (float(v) for v in scores.values() if math.isfinite(v)), reverse=True
        )
        if len(values) < 2:
            return None
        gap = values[0] - values[1]
        if self.relative and gap > 0:
            gap /= max(abs(values[0]), abs(values[1]))
        return -gap


class UnfamiliarityGate(ScoreGate):
    """
    Run extras on instances the selector's feature detector finds unfamiliar
    (its per-instance ``score``). Unrated until the detector is fitted.
    """

    needs_instance_score = True

    def _score(self, context: GateContext) -> float | None:
        return context.instance_score


class TimeoutRiskGate(ScoreGate):
    """
    Run extras when the chosen algorithm is predicted to be slow
    (``prediction.predicted_runtime``, raw scale, e.g. ``threshold=0.8 *
    cutoff``): a likely timeout makes a second run pay off.
    """

    def _score(self, context: GateContext) -> float | None:
        runtime = context.prediction.predicted_runtime
        if runtime is None or not math.isfinite(runtime):
            return None
        return float(runtime)
