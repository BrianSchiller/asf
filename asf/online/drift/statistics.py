"""Adapters turning an observation plus a prediction into a scalar for detectors."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from asf.online.prediction import PredictionRecord

if TYPE_CHECKING:
    from asf.online.base import Observation


@dataclass(frozen=True)
class ScalarSample:
    """
    One value of a scalar drift stream.

    Attributes
    ----------
    value : float
        The statistic.
    censored : bool, default=False
        ``True`` if ``value`` is a bound rather than an exact value.
    """

    value: float
    censored: bool = False


class AbstractDriftStatistic(ABC):
    """Maps ``(observation, prediction)`` to a :class:`ScalarSample`."""

    @abstractmethod
    def __call__(
        self, observation: "Observation", prediction: PredictionRecord | None
    ) -> ScalarSample | None:
        """
        Return the sample for this instance, or ``None`` if there is nothing
        to feed to the detector (e.g. the selector made no runtime prediction).
        """


class LogRuntimeResidual(AbstractDriftStatistic):
    """
    ``log(observed) - log(predicted)`` runtime of the primary algorithm.

    Positive values mean the algorithm was slower than predicted. If the
    primary run was censored, the observed runtime is the budget, so the
    value is a *lower bound* of the true residual.

    Requires a selector that sets ``PredictionRecord.predicted_runtime``;
    otherwise no value is produced. Intended for runtime minimisation.

    Parameters
    ----------
    censored : {"emit", "skip"}, default="emit"
        ``"emit"``: pass censored residuals on with ``censored=True``.
        ``"skip"``: produce no value for censored runs.
    floor : float, default=1e-6
        Runtimes (observed and predicted) are clipped to at least ``floor``
        before taking logs, so zero runtimes stay finite.
    """

    def __init__(
        self, censored: Literal["emit", "skip"] = "emit", floor: float = 1e-6
    ) -> None:
        if censored not in ("emit", "skip"):
            raise ValueError(f"Unknown censored handling: {censored!r}")
        if not floor > 0:
            raise ValueError("floor must be > 0.")
        self.censored = censored
        self.floor = float(floor)

    def __call__(
        self, observation: "Observation", prediction: PredictionRecord | None
    ) -> ScalarSample | None:
        if prediction is None or prediction.predicted_runtime is None:
            return None
        predicted = float(prediction.predicted_runtime)
        observed = observation.runtimes.get(prediction.chosen)
        if observed is None or not math.isfinite(predicted):
            return None
        observed = float(observed)
        if not math.isfinite(observed):
            return None
        is_censored = prediction.chosen in observation.censored
        if is_censored and self.censored == "skip":
            return None
        value = math.log(max(observed, self.floor)) - math.log(
            max(predicted, self.floor)
        )
        return ScalarSample(value=value, censored=is_censored)
