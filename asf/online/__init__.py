"""
Online algorithm selection under distribution drift.

Layout:
    base         -- AbstractOnlineSelector, Observation (propose/observe contract)
    prediction   -- PredictionRecord (what a selector decided)
    gate         -- extra-run gates (which additional algorithms to run)
    buffer       -- sliding window over partially observed instances
    imputation   -- rules for learning from censored (timed-out) runs
    drift        -- drift detectors (WHEN to react)
    retraining   -- drift responses (HOW to react)
    selectors    -- online selectors
    streams      -- instance streams with covariate drift from benchmark data
    evaluation   -- prequential evaluation loop and metrics
"""

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.buffer import ObservationBuffer

__all__ = ["AbstractOnlineSelector", "Observation", "ObservationBuffer"]
