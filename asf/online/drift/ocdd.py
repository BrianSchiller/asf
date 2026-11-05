"""One-class drift detector (OCDD) based on a one-class SVM."""

from __future__ import annotations

import pandas as pd

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


class OneClassDriftDetector(AbstractDriftDetector):
    """
    One-class drift detector (OCDD) based on a one-class SVM.

    TODO: implement. See the thesis proposal for the reference.
    """

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "OneClassDriftDetector":
        raise NotImplementedError

    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        raise NotImplementedError
