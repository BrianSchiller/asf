"""
Online algorithm selection under distribution drift.

Layout:
    base         -- AbstractOnlineSelector, Observation (propose/observe contract)
    buffer       -- sliding window over partially observed instances
    drift        -- drift detectors (WHEN to reconfigure)
    retraining   -- retrain policies tying detectors to selectors
    selectors    -- online selectors (HOW to reconfigure)
    streams      -- instance streams with controlled drift
    evaluation   -- prequential evaluation loop and metrics
"""

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.buffer import ObservationBuffer

__all__ = ["AbstractOnlineSelector", "Observation", "ObservationBuffer"]
