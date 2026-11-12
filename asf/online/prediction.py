"""What a selector predicted for one instance (the prediction record)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionRecord:
    """
    A selector's decision for one instance, before and after the extra-run gate.

    Selectors build it in ``_select``; the base class fills in ``extras`` (the
    gate's decision) and ``instance_id``, stores it as
    ``selector.last_prediction_`` and the evaluator logs it per step.

    Attributes
    ----------
    chosen : str
        The primary algorithm (its result is what the user gets in
        ``"primary"`` result mode).
    budget : float, default=0.0
        Run budget of the primary run. ``0`` or less means "the scenario
        cutoff". Extra runs get the same budget.
    predicted_runtime : float or None, default=None
        Predicted performance of ``chosen`` on the raw scale of the stream
        (seconds for runtime scenarios). ``None`` if the selector has no such
        model (e.g. vote-based or reward-based selectors).
    score : float or None, default=None
        Selector-internal score of ``chosen``; higher means preferred (votes,
        UCB index, sampled reward, ...). Not comparable across selectors.
    uncertainty : float or None, default=None
        Selector-defined uncertainty about the decision; higher means less
        sure. Not comparable across selectors.
    scores : Mapping[str, float] or None, default=None
        Selector-internal score per algorithm, for gates that compare
        alternatives.
    candidates : tuple[str, ...], default=()
        The selector's nominees for extra runs, most useful first. The gate
        chooses the actual extras from these. The default gate of a selector
        (see ``AbstractOnlineSelector._default_gate``) reproduces its built-in
        behaviour, e.g. the Rook selector nominates exactly ``n_additional``
        algorithms and runs all of them.
    extras : tuple[str, ...], default=()
        Extra algorithms actually scheduled (set by the base class).
    instance_id : Any, default=None
        Identifier of the instance (set by the base class).
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
