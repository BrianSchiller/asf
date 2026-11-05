"""Baseline: an offline ASF selector deployed on a stream without adapting."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.selectors.abstract_selector import AbstractSelector


class StaticOnlineSelector(AbstractOnlineSelector):
    """
    Wrap any offline ASF selector so it can run in the online loop.

    It is fitted once on the warm-start data and never updated; it is the
    "no adaptation" lower bound for drifting streams and doubles as an adapter
    check for the online interface.

    Parameters
    ----------
    selector : AbstractSelector
        An unfitted offline selector with ``prediction_mode="aslib"``.
    **kwargs : Any
        Forwarded to :class:`AbstractOnlineSelector`.
    """

    def __init__(self, selector: AbstractSelector, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if selector.prediction_mode != "aslib":
            raise ValueError("The wrapped selector must use prediction_mode='aslib'.")
        self.selector = selector

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        self.selector.fit(features, performance, **kwargs)

    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        prediction = self.selector.predict(features)
        if not isinstance(prediction, dict):
            raise TypeError("The wrapped selector must return ASlib-style predictions.")
        schedule = cast(
            "list[tuple[str, float | None]]", next(iter(prediction.values()))
        )
        return [(str(a), float(b or 0.0)) for a, b in schedule[:1]]

    def _observe(self, observation: Observation) -> None:
        return None
