"""Baseline: an offline ASF selector deployed on a stream without adapting."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.prediction import PredictionRecord
from asf.selectors.abstract_selector import AbstractSelector


class StaticOnlineSelector(AbstractOnlineSelector):
    """
    An offline ASF selector (``prediction_mode="aslib"``) fitted once on the
    warm start and never updated: the "no adaptation" baseline.
    """

    SUPPORTS_COLD_START = False

    def __init__(self, selector: AbstractSelector, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if selector.prediction_mode != "aslib":
            raise ValueError("The wrapped selector must use prediction_mode='aslib'.")
        self.selector = selector

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        self.selector.fit(features, performance, **kwargs)

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        prediction = self.selector.predict(features)
        if not isinstance(prediction, dict):
            raise TypeError("The wrapped selector must return ASlib-style predictions.")
        schedule = cast(
            "list[tuple[str, float | None]]", next(iter(prediction.values()))
        )
        algo, budget = schedule[0]
        return PredictionRecord(str(algo), budget=float(budget or 0.0))

    def _observe(self, observation: Observation) -> None:
        return None
