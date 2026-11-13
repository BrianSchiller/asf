"""Frugal algorithm selection variant of the online pairwise selector."""

from __future__ import annotations

from typing import Any

import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.prediction import PredictionRecord


class FrugalPairwiseSelector(AbstractOnlineSelector):
    """
    Like :class:`OnlinePairwiseSelector`, but a second solver is only run when
    the pair classifier is uncertain (e.g. margin of ``predict_proba``, reached
    via ``wrapper.model_class.predict_proba`` as SATzilla/CSHC do), see
    Kus et al., "Frugal Algorithm Selection" (2024).

    TODO: implement. Intended shape (F1 interfaces): ``_select`` returns a
    ``PredictionRecord`` with ``uncertainty`` set and the nominee(s) in
    ``candidates``; the uncertainty criterion lives in an
    ``AbstractExtraRunGate`` passed as ``gate``.
    """

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        raise NotImplementedError

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        raise NotImplementedError

    def _observe(self, observation: Observation) -> None:
        raise NotImplementedError
