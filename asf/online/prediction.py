"""What a selector predicted for one instance (the prediction record)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionRecord:
    """
    A selector's decision for one instance, built in ``_select``; the base
    class fills in ``extras`` (the gate's choice) and ``instance_id``.

    Attributes
    ----------
    budget : float, default=0.0
        Budget of the primary run and the extras (``<= 0`` = the cutoff).
    predicted_runtime : float or None
        Predicted performance of ``chosen`` on the stream's raw scale
        (``None`` for selectors without a runtime model).
    score, scores : float / Mapping[str, float] or None
        Selector-internal preference of ``chosen`` / of every algorithm
        (higher = preferred; votes, UCB index, ...).
    uncertainty : float or None
        Selector-defined uncertainty (higher = less sure).
    candidates : tuple[str, ...]
        Nominees for extra runs, most useful first; the gate picks from them.

    ``score`` and ``uncertainty`` are not comparable across selectors.
    """

    chosen: str
    budget: float = 0.0
    predicted_runtime: float | None = None
    score: float | None = None
    uncertainty: float | None = None
    scores: Mapping[str, float] | None = None
    candidates: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    instance_id: Any = None

    @property
    def schedule(self) -> list[tuple[str, float]]:
        """ASlib-style schedule: the primary run first, then the extras."""
        return [(self.chosen, self.budget)] + [(a, self.budget) for a in self.extras]
