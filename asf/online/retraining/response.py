"""
Drift responses: HOW a selector reacts when its detector signals drift.

Each response calls one selector hook (``REQUIRES``); fitting a selector
that lacks it raises. After a retrain or reset the selector re-references
its detector. ``ReweightResponse`` and ``RaiseExplorationResponse`` are not
implemented by any selector yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from asf.online.drift.abstract_detector import DriftSignal

if TYPE_CHECKING:
    from asf.online.base import AbstractOnlineSelector


class AbstractDriftResponse(ABC):
    """One way of reacting to drift; ``NAME`` is logged in the results."""

    NAME: ClassVar[str]
    REQUIRES: ClassVar[str]

    @abstractmethod
    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        """React to ``signal`` by calling the selector's hook."""


class RereferenceResponse(AbstractDriftResponse):
    """
    Only re-reference the detector on the buffer; keep the model.

    Stops a detector from signalling the same change again and again when the
    reaction lies elsewhere (e.g. :class:`~asf.online.gate.AfterDriftGate`).
    """

    NAME = "rereference"
    # Needs no capability hook; every online selector implements ``_observe``.
    REQUIRES = "_observe"

    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        selector.rereference_detector()


class RetrainResponse(AbstractDriftResponse):
    """Retrain the selector's model on its buffer (Rook et al.)."""

    NAME = "retrain"
    REQUIRES = "_retrain"

    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        selector.retrain()


class ResetResponse(AbstractDriftResponse):
    """Forget everything learnt so far (bandits: ``reset_arms``)."""

    NAME = "reset"
    REQUIRES = "_reset_model"

    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        selector.reset_model()


class ReweightResponse(AbstractDriftResponse):
    """Multiply the weight of everything seen before the drift by ``factor``
    in ``[0, 1]`` (0 = forget, 1 = no change)."""

    NAME = "reweight"
    REQUIRES = "_discount_history"

    def __init__(self, factor: float) -> None:
        if not 0.0 <= factor <= 1.0:
            raise ValueError("factor must be in [0, 1].")
        self.factor = float(factor)

    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        selector._discount_history(self.factor)


class RaiseExplorationResponse(AbstractDriftResponse):
    """
    Explore more for ``steps`` instances (``None`` = until the next drift),
    e.g. wider confidence bounds; ``factor > 0`` is the selector-defined
    strength (1 = no change).
    """

    NAME = "raise_exploration"
    REQUIRES = "_boost_exploration"

    def __init__(self, factor: float, steps: int | None) -> None:
        if not factor > 0:
            raise ValueError("factor must be > 0.")
        if steps is not None and steps < 1:
            raise ValueError("steps must be >= 1 or None.")
        self.factor = float(factor)
        self.steps = None if steps is None else int(steps)

    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        selector._boost_exploration(self.factor, self.steps)
