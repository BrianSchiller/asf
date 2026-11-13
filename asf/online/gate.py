"""
Extra-run gates: decide, per instance, which additional algorithms to run.

Running more than the selected algorithm costs compute time but yields
performance information the selector can learn from. A gate makes that
decision from the selector's prediction (including its uncertainty), the
selector's nominees and, optionally, a detector's per-instance score.

Every :class:`~asf.online.base.AbstractOnlineSelector` owns one gate
(``gate=...``); without one it uses the selector's default, which reproduces
the selector's original behaviour (Rook selector: all ``n_additional``
nominees; bandits: none). :class:`UncertaintyGate` runs extras only when the
selector is uncertain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from asf.online.drift.abstract_detector import DriftSignal
from asf.online.prediction import PredictionRecord


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


class UncertaintyGate(AbstractExtraRunGate):
    """
    Run the first ``n`` nominees when the selector is uncertain enough.

    This is stream-based active learning: an extra run is "queried" only if
    ``prediction.uncertainty >= threshold`` (Kuş et al., "Frugal Algorithm
    Selection"). Predictions without an uncertainty never trigger extra runs.

    Parameters
    ----------
    threshold : float
        Minimum uncertainty for extra runs. Its scale is the selector's (e.g.
        least confidence in ``[0, 0.5]`` for the frugal pairwise selector).
    n : int or None, default=1
        Number of nominees to run when the gate fires; ``None`` = all.
    """

    def __init__(self, threshold: float, n: int | None = 1) -> None:
        if n is not None and n < 0:
            raise ValueError("n must be >= 0 or None.")
        self.threshold = float(threshold)
        self.n = None if n is None else int(n)

    def extras(self, context: GateContext) -> Sequence[str]:
        uncertainty = context.prediction.uncertainty
        if uncertainty is None or uncertainty < self.threshold:
            return ()
        if self.n is None:
            return context.candidates
        return context.candidates[: self.n]
