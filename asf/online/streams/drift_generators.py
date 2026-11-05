"""
Turn a static (ASlib-like) dataset into a stream with controlled drift.

ASlib scenarios have no temporal order, so drift has to be constructed.
TODO: check how Rook et al. built their streams and reuse that construction for
comparability.
"""

from __future__ import annotations

import pandas as pd

from asf.online.streams.stream import InstanceStream


def abrupt_drift(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    cutoff: float,
    group_labels: pd.Series,
    group_order: list[object],
    n_per_group: int | None = None,
    seed: int | None = None,
) -> InstanceStream:
    """Concatenate instance groups (e.g. clusters) one after another."""
    raise NotImplementedError


def gradual_drift(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    cutoff: float,
    group_labels: pd.Series,
    group_order: list[object],
    transition_length: int,
    seed: int | None = None,
) -> InstanceStream:
    """Blend consecutive groups with a linearly changing sampling probability."""
    raise NotImplementedError


def recurring_drift(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    cutoff: float,
    group_labels: pd.Series,
    schedule: list[object],
    seed: int | None = None,
) -> InstanceStream:
    """Revisit groups according to ``schedule`` (e.g. A, B, A, C, B)."""
    raise NotImplementedError
