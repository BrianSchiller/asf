"""Partitioners: route an instance to its own group of arm models.

A partitioner splits feature space into cells (clusters, tree leaves, ...).
Every cell has its own posterior per arm, and only the cell's observations
are used for it. The default is a single cell (one global model per arm).

Cell keys are assigned when an observation arrives and stored with it, so
both computation modes see the same data. Online clustering (DTS-OC) or a
dyadic partitioning tree implement :class:`Partitioner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable

import numpy as np


class Partitioner(ABC):
    """Abstract partitioner over feature vectors ``z``."""

    def fit(self, Z: np.ndarray) -> None:
        """Initialise from warm-start feature vectors (default: nothing)."""

    def update(self, z: np.ndarray) -> None:
        """React to a newly observed instance (default: nothing)."""

    @abstractmethod
    def assign(self, z: np.ndarray) -> Hashable:
        """Key of the cell that ``z`` belongs to."""


class SinglePartition(Partitioner):
    """Everything in one cell: a single global model per arm."""

    def assign(self, z: np.ndarray) -> Hashable:
        return 0
