"""
Interfaces for drift detectors.

Two kinds of detectors exist:

* :class:`AbstractDriftDetector` -- *feature-based* (or performance-row-based):
  sees each instance's features and partially observed performance, needs a
  reference fit, may score single instances. Example: PCA reconstruction.
* :class:`AbstractScalarDriftDetector` -- *scalar-stream*: sees one number
  per instance (e.g. a prediction residual) and needs no reference data.
  Examples: ADWIN, EDDM. An
  :class:`~asf.online.drift.statistics.AbstractDriftStatistic` turns an
  observation plus the selector's prediction into that number.

Both report a :class:`DriftSignal` per update.
"""

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
    Base class for feature-based drift detectors.

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


class AbstractScalarDriftDetector(ABC):
    """
    Base class for detectors on a stream of scalars (ADWIN, EDDM, ...).

    The detector sees one value per instance, typically produced by an
    :class:`~asf.online.drift.statistics.AbstractDriftStatistic` from the
    observation and the selector's prediction. It needs no reference fit.
    All window sizes, confidence levels and thresholds must be constructor
    parameters.
    """

    @abstractmethod
    def update(self, value: float, censored: bool = False) -> DriftSignal:
        """
        Consume one value and report drift.

        Parameters
        ----------
        value : float
            The statistic for the current instance.
        censored : bool, default=False
            ``True`` if ``value`` is only a bound (e.g. the run was stopped at
            its budget, so the true runtime is at least the observed one).
            Detectors that cannot use bounds may ignore such values.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget the stream seen so far (called after a handled drift)."""
