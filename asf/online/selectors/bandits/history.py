"""Per-arm record of everything a bandit has observed."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class ArmData:
    """Stored observations of one arm (optionally restricted to one cell).

    Attributes
    ----------
    steps : np.ndarray
        Arrival clock of each observation, shape ``(n,)``.
    Z : np.ndarray
        Feature vectors, shape ``(n, d)``.
    rewards : np.ndarray
        Rewards, shape ``(n,)``.
    weights : np.ndarray
        Weights fixed at arrival (e.g. confidence scores), shape ``(n,)``.
    """

    steps: np.ndarray
    Z: np.ndarray
    rewards: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.steps)


class BanditHistory:
    """
    Stores ``(step, z, cell, reward, weight)`` per arm.

    Parameters
    ----------
    arms : Sequence[str]
        Arm names.
    dim : int
        Length of the feature vectors ``z``.
    max_size : int or None, default=None
        Keep at most this many observations per arm (oldest dropped first).
        ``None`` keeps everything.
    """

    def __init__(
        self, arms: Sequence[str], dim: int, max_size: int | None = None
    ) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1.")
        self.dim = int(dim)
        self.max_size = max_size
        self._rows: dict[str, deque[tuple[int, np.ndarray, Hashable, float, float]]]
        self._rows = {str(a): deque(maxlen=max_size) for a in arms}

    def add(
        self,
        arm: str,
        step: int,
        z: np.ndarray,
        cell: Hashable,
        reward: float,
        weight: float = 1.0,
    ) -> None:
        """Record one observation of ``arm``."""
        self._rows[arm].append((int(step), np.asarray(z, float), cell, reward, weight))

    def clear(self) -> None:
        """Forget everything."""
        for rows in self._rows.values():
            rows.clear()

    def n_observations(self, arm: str) -> int:
        """Number of stored observations of ``arm``."""
        return len(self._rows[arm])

    def get(self, arm: str, cell: Hashable | None = None) -> ArmData:
        """Observations of ``arm``; only those of ``cell`` if it is given."""
        rows = [r for r in self._rows[arm] if cell is None or r[2] == cell]
        if not rows:
            return ArmData(
                np.zeros(0, dtype=int),
                np.zeros((0, self.dim)),
                np.zeros(0),
                np.zeros(0),
            )
        return ArmData(
            steps=np.array([r[0] for r in rows], dtype=int),
            Z=np.vstack([r[1] for r in rows]).reshape(len(rows), self.dim),
            rewards=np.array([r[3] for r in rows], dtype=float),
            weights=np.array([r[4] for r in rows], dtype=float),
        )
