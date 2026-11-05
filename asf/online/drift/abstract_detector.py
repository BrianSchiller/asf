"""Interface for drift detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftSignal:
    """
    Output of a drift detector for one update.

    Attributes
    ----------
    drift : bool
        ``True`` if the detector signals a drift at this step.
    warning : bool, default=False
        ``True`` if the detector is in a warning zone (if it has one).
    score : float or None, default=None
        Detector-specific drift statistic, useful for plotting.
    """

    drift: bool
    warning: bool = False
    score: float | None = None


class AbstractDriftDetector(ABC):
    """
    Base class for drift detectors.

    A detector is *independent of any selector*: it only sees features and,
    for performance-aware detectors, the observed performance of the instance.
    Retraining policies and selectors consume its signals.

    Two usage levels are supported:

    * stream level: :meth:`update` is called once per instance and returns a
      :class:`DriftSignal` (when to reconfigure).
    * instance level: :meth:`score` rates how unfamiliar instances are (optional;
      used to weight instances or to trigger extra solver runs).
    """

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame | None = None,
    ) -> "AbstractDriftDetector":
        """Fit the reference distribution on (warm-start / post-drift) data."""

    @abstractmethod
    def update(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame | None = None,
    ) -> DriftSignal:
        """
        Consume one new instance (single-row frames) and report drift.

        ``performance`` may be partially observed (NaN for non-executed
        algorithms) and is ignored by feature-based detectors.
        """

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Per-instance unfamiliarity score (higher = more drifted)."""
        raise NotImplementedError(f"{type(self).__name__} has no instance-level score.")

    def reset(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> None:
        """Re-reference the detector after a handled drift (default: refit)."""
        self.fit(features, performance)
