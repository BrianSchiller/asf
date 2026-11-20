"""
Interfaces for drift detectors, both reporting a :class:`DriftSignal` per update.

* :class:`AbstractDriftDetector`: sees features (and the partially observed
  performance row), needs a reference fit, may score single instances.
* :class:`AbstractScalarDriftDetector`: sees one number per instance (e.g. a
  prediction residual from a
  :data:`~asf.online.drift.statistics.DriftStatistic`), no reference needed.
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
    """Output of one detector update; ``score`` is detector-specific (for plots)."""

    drift: bool
    warning: bool = False
    score: float | None = None


class AbstractDriftDetector(ABC):
    """
    Base class for feature-based drift detectors (independent of any selector).

    :meth:`update` reports drift once per instance; the optional :meth:`score`
    rates how unfamiliar single instances are (e.g. for extra-run gates).
    ``feature_names_`` holds the columns seen in ``fit``.
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
        """Consume one instance (``performance`` may contain NaN for runs not made)."""

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
        """Append ``items`` to ``window``; once it is full, run ``check``
        every ``self.stride`` items. Drift if any check found drift."""
        drift = False
        for item in items:
            window.append(item)
            self._since_check += 1
            if len(window) == window.maxlen and self._since_check >= self.stride:
                self._since_check = 0
                drift |= check()
        return drift


class AbstractScalarDriftDetector(ABC):
    """Base class for detectors on a stream of scalars (ADWIN, EDDM, ...)."""

    @abstractmethod
    def update(self, value: float, censored: bool = False) -> DriftSignal:
        """Consume one value; ``censored`` means it is only a lower bound."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the stream seen so far (called after a handled drift)."""
