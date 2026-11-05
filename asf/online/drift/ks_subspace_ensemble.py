"""Ensemble of univariate Kolmogorov-Smirnov tests over feature subspaces."""

from __future__ import annotations

import pandas as pd

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


class KSSubspaceEnsembleDetector(AbstractDriftDetector):
    """
    Ensemble of univariate Kolmogorov-Smirnov tests over feature subspaces.

    TODO: implement. See the thesis proposal for the reference.
    """

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "KSSubspaceEnsembleDetector":
        raise NotImplementedError

    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        raise NotImplementedError
