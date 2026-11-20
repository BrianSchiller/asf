"""Instance streams with hidden ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class StreamItem:
    """One arriving instance (what the selector is allowed to see)."""

    step: int
    instance_id: object
    features: pd.DataFrame  # single row


@dataclass
class InstanceStream:
    """
    Time-ordered instances plus the *complete* performance matrix, which only
    the evaluator reads (selectors see the runs they make).

    ``performance`` holds raw runtimes (values at or above ``cutoff`` count as
    unsolved). ``drift_points`` are the injected drift positions and
    ``transition_intervals`` the half-open ``[start, end)`` gradual shifts.
    """

    features: pd.DataFrame
    performance: pd.DataFrame
    cutoff: float
    drift_points: tuple[int, ...] = field(default_factory=tuple)
    name: str = "stream"
    transition_intervals: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.features.index.equals(self.performance.index):
            raise ValueError("features and performance must share the same index.")

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[StreamItem]:
        for step, idx in enumerate(self.features.index):
            yield StreamItem(step, idx, self.features.loc[[idx]])

    def true_runtimes(self, instance_id: object) -> pd.Series:
        """Ground-truth runtimes of all algorithms (evaluator only)."""
        return self.performance.loc[instance_id]

    def split(self, n_warmstart: int) -> tuple["InstanceStream", "InstanceStream"]:
        """Split into a warm-start prefix and the remaining stream (0 = cold start)."""
        if not 0 <= n_warmstart < len(self):
            raise ValueError("n_warmstart must be in [0, len(stream)).")
        head_intervals = tuple(
            (start, min(end, n_warmstart))
            for start, end in self.transition_intervals
            if start < n_warmstart and end > 0
        )
        tail_intervals = tuple(
            (max(0, start - n_warmstart), end - n_warmstart)
            for start, end in self.transition_intervals
            if end > n_warmstart and start < len(self)
        )
        head = InstanceStream(
            self.features.iloc[:n_warmstart],
            self.performance.iloc[:n_warmstart],
            self.cutoff,
            drift_points=tuple(p for p in self.drift_points if p < n_warmstart),
            name=f"{self.name}[warmstart]",
            transition_intervals=head_intervals,
        )
        tail = InstanceStream(
            self.features.iloc[n_warmstart:],
            self.performance.iloc[n_warmstart:],
            self.cutoff,
            tuple(p - n_warmstart for p in self.drift_points if p >= n_warmstart),
            name=self.name,
            transition_intervals=tail_intervals,
        )
        return head, tail
