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
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

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

    Attributes
    ----------
    feature_names_ : list[str]
        Feature names seen in ``fit`` (empty before).
    """

    feature_names_: list[str] = []
    stride: int
    _since_check: int = 0

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

    # ------------------------------------------------- helpers for subclasses
    def _reference_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Check the fit data, remember its columns, return it as an array."""
        if len(features) < 2:
            raise ValueError("At least two reference instances are required.")
        if features.shape[1] == 0:
            raise ValueError("At least one feature is required.")
        self.feature_names_ = [str(c) for c in features.columns]
        return features.to_numpy(float)

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        """``features`` as an array, columns in the order seen in ``fit``."""
        if not self.feature_names_:
            raise RuntimeError("Call fit() before update()/score().")
        if list(features.columns) == self.feature_names_:
            return features.to_numpy(float)
        missing = [
            c for c in self.feature_names_ if c not in map(str, features.columns)
        ]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]}")
        return features.rename(columns=str)[self.feature_names_].to_numpy(float)

    def _strided_checks(
        self, items: Iterable[Any], window: deque, check: Callable[[], bool]
    ) -> bool:
        """
        Append ``items`` to ``window`` one by one and run ``check`` every
        ``self.stride`` items once the window is full; drift if any check
        found drift.
        """
        drift = False
        for item in items:
            window.append(item)
            self._since_check += 1
            if len(window) == window.maxlen and self._since_check >= self.stride:
                self._since_check = 0
                drift |= check()
        return drift


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
