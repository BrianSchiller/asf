"""
Drift responses: HOW a selector reacts when its detector signals drift.

A response is called from :meth:`AbstractOnlineSelector.on_drift` and uses
exactly one capability hook of the selector (``REQUIRES``). Selectors that do
not implement the hook cannot be combined with the response; this is checked
when the selector is fitted.

==============================  ======================  =====================
Response                        Selector hook           Implemented by (F1)
==============================  ======================  =====================
:class:`RetrainResponse`        ``_retrain``            OnlinePairwiseSelector
:class:`ResetResponse`          ``_reset_model``        bandits (``reset_arms``)
:class:`ReweightResponse`       ``_discount_history``   -- (planned)
:class:`RaiseExplorationResponse` ``_boost_exploration``  -- (planned)
==============================  ======================  =====================

Retrain and reset rebuild the selector's model, so the selector re-references
its detector afterwards (refit on the current buffer, or reset for scalar
detectors). Reweighting and raising exploration keep the model and do not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from asf.online.drift.abstract_detector import DriftSignal

if TYPE_CHECKING:
    from asf.online.base import AbstractOnlineSelector


class AbstractDriftResponse(ABC):
    """
    One way of reacting to a drift signal.

    Attributes
    ----------
    NAME : str
        Short label logged in the ``response`` column of the results.
    REQUIRES : str
        Name of the :class:`~asf.online.base.AbstractOnlineSelector` hook the
        response calls.
    """

    NAME: ClassVar[str]
    REQUIRES: ClassVar[str]

    @abstractmethod
    def respond(self, selector: "AbstractOnlineSelector", signal: DriftSignal) -> None:
        """React to ``signal`` by calling the selector's hook."""


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
    """
    Down-weight everything observed before the drift.

    Parameters
    ----------
    factor : float
        Multiplier in ``[0, 1]`` applied to the weight of all past
        observations (0 = forget, 1 = no change).
    """

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
    Temporarily explore more (e.g. widen confidence bounds, lower a gate's
    threshold).

    Parameters
    ----------
    factor : float
        Selector-defined strength of the boost (``> 0``; 1 = no change).
    steps : int or None
        Number of instances the boost lasts; ``None`` = until the next drift
        response.
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
