"""Frugal algorithm selection variant of the online pairwise selector."""

from __future__ import annotations

from typing import Any

import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation


class FrugalPairwiseSelector(AbstractOnlineSelector):
    """
    Like :class:`OnlinePairwiseSelector`, but a second solver is only run when
    the pair classifier is uncertain (e.g. margin of ``predict_proba``, reached
    via ``wrapper.model_class.predict_proba`` as SATzilla/CSHC do), see
    Kus et al., "Frugal Algorithm Selection" (2024).

    TODO: implement.
    """

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        raise NotImplementedError

    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        raise NotImplementedError

    def _observe(self, observation: Observation) -> None:
        raise NotImplementedError
