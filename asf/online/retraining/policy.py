"""Policies deciding WHEN a selector is retrained."""

from __future__ import annotations

from abc import ABC, abstractmethod

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


class AbstractRetrainPolicy(ABC):
    """Decides, after each observed instance, whether to retrain."""

    @abstractmethod
    def should_retrain(self, step: int, signal: DriftSignal | None = None) -> bool:
        """
        Parameters
        ----------
        step : int
            Number of instances observed since the stream started (1-based).
        signal : DriftSignal or None
            Latest detector output, if a detector is attached.
        """

    def on_retrained(self, step: int) -> None:  # noqa: B027
        """Hook called after the selector was retrained."""


class PeriodicRetrain(AbstractRetrainPolicy):
    """Retrain every ``every`` instances (e.g. 10), regardless of drift."""

    def __init__(self, every: int = 10) -> None:
        if every < 1:
            raise ValueError("every must be >= 1.")
        self.every = int(every)

    def should_retrain(self, step: int, signal: DriftSignal | None = None) -> bool:
        return step % self.every == 0


class DriftTriggeredRetrain(AbstractRetrainPolicy):
    """Retrain only when the attached detector signals drift."""

    def __init__(self, detector: AbstractDriftDetector | None = None) -> None:
        self.detector = detector

    def should_retrain(self, step: int, signal: DriftSignal | None = None) -> bool:
        return bool(signal is not None and signal.drift)
