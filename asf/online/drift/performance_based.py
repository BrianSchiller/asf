"""Performance-aware drift detection on solver performance / regret."""

from __future__ import annotations

import pandas as pd

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


class PerformanceDriftDetector(AbstractDriftDetector):
    """
    Performance-aware drift detection on solver performance / regret.

    TODO: implement. See the thesis proposal for the reference.
    """

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "PerformanceDriftDetector":
        raise NotImplementedError

    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        raise NotImplementedError
