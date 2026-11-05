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
    Time-ordered instances plus the *complete* performance matrix.

    The full matrix is the ground truth and is only accessed by the evaluator;
    selectors only learn the entries of the algorithms they actually run.

    Attributes
    ----------
    features : pd.DataFrame
        Instance features in arrival order (index = instance ids).
    performance : pd.DataFrame
        Raw runtimes, same index and order as ``features`` (timeouts NOT yet
        penalised; values above ``cutoff`` count as unsolved).
    cutoff : float
        Scenario cutoff time.
    drift_points : tuple[int, ...], default=()
        Stream positions where drift was injected (ground truth for detection
        metrics).
    name : str, default="stream"
        Label used in result tables.
    """

    features: pd.DataFrame
    performance: pd.DataFrame
    cutoff: float
    drift_points: tuple[int, ...] = field(default_factory=tuple)
    name: str = "stream"

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
        head = InstanceStream(
            self.features.iloc[:n_warmstart],
            self.performance.iloc[:n_warmstart],
            self.cutoff,
            name=f"{self.name}[warmstart]",
        )
        tail = InstanceStream(
            self.features.iloc[n_warmstart:],
            self.performance.iloc[n_warmstart:],
            self.cutoff,
            tuple(p - n_warmstart for p in self.drift_points if p >= n_warmstart),
            name=self.name,
        )
        return head, tail
