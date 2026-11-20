"""Shared helpers: toy streams, fake detectors and a fake selector."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.drift import AbstractDriftDetector, AbstractScalarDriftDetector
from asf.online.drift import DriftSignal
from asf.online.gate import AbstractExtraRunGate, GateContext
from asf.online.prediction import PredictionRecord
from asf.online.streams.stream import InstanceStream

CUTOFF = 100.0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running end-to-end test")


# ---------------------------------------------------------------- streams
def stream_from(
    x: Any,
    perf: Any,
    algos: Sequence[str] = ("a", "b", "c"),
    cutoff: float = CUTOFF,
    drift_points: Sequence[int] = (),
    name: str = "toy",
) -> InstanceStream:
    """Stream with rows ``i0, i1, ...``, features ``f0, f1, ...``."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    perf = np.asarray(perf, dtype=float)
    idx = pd.Index([f"i{k}" for k in range(len(x))])
    return InstanceStream(
        pd.DataFrame(x, index=idx, columns=[f"f{j}" for j in range(x.shape[1])]),
        pd.DataFrame(perf, index=idx, columns=list(algos)[: perf.shape[1]]),
        cutoff=cutoff,
        drift_points=tuple(drift_points),
        name=name,
    )


def make_stream(n: int = 120, drift_at: int = 80, seed: int = 0) -> InstanceStream:
    """``a`` is best when f0 > 0 before the drift; after it the roles flip."""
    f = np.random.default_rng(seed).normal(size=(n, 2))
    a_best = f[:, 0] > 0
    a_best[drift_at:] = ~a_best[drift_at:]
    perf = np.where(a_best[:, None], [[5.0, 50.0]], [[50.0, 5.0]])
    return stream_from(f, perf, drift_points=(drift_at,))


def two_region_stream(
    n_before: int = 150, n_after: int = 150, seed: int = 0, cutoff: float = 1000.0
) -> InstanceStream:
    """f0 ~ N(0, 1) with ``a`` best, then f0 ~ N(8, 1) with ``b`` best."""
    rng = np.random.default_rng(seed)
    f0 = np.concatenate([rng.normal(0, 1, n_before), rng.normal(8, 1, n_after)])
    x = np.column_stack([f0, rng.normal(size=len(f0))])
    before = (np.arange(len(f0)) < n_before)[:, None]
    perf = np.where(before, [[5, 50, 20]], [[50, 5, 20]])
    return stream_from(x, perf, cutoff=cutoff, drift_points=(n_before,))


def three_algo_stream(n: int = 80) -> InstanceStream:
    """Constant runtimes a=5, b=50, c=20."""
    x = np.random.default_rng(1).normal(size=(n, 2))
    return stream_from(x, np.tile([5.0, 50.0, 20.0], (n, 1)))


def observe_runs(stream: InstanceStream, i: int, algos: Sequence[str]) -> Observation:
    """Observation of running ``algos`` on instance ``i`` (capped at the cutoff)."""
    item_id = stream.features.index[i]
    truth = stream.true_runtimes(item_id)
    cutoff = stream.cutoff
    runs = {a: float(min(truth[a], cutoff)) for a in algos}
    censored = frozenset(a for a in algos if truth[a] > cutoff)
    return Observation(item_id, stream.features.iloc[[i]], runs, censored, cutoff)


@pytest.fixture
def stream() -> InstanceStream:
    return make_stream()


# ---------------------------------------------------------------- fakes
class ScriptedDetector(AbstractDriftDetector):
    """Signals drift at given update counts; records fits and resets."""

    def __init__(self, drift_at: set[int] = frozenset()) -> None:  # type: ignore[assignment]
        self.drift_at = set(drift_at)
        self.n_updates = 0
        self.n_resets = 0
        self.fit_sizes: list[int] = []

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "ScriptedDetector":
        self.fit_sizes.append(len(features))
        return self

    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        self.n_updates += 1
        return DriftSignal(drift=self.n_updates in self.drift_at, score=1.0)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 0.25)

    def reset(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> None:
        self.n_resets += 1
        self.fit_sizes.append(len(features))


class ScalarRecorder(AbstractScalarDriftDetector):
    """Records its inputs; drifts when a value exceeds 1."""

    def __init__(self) -> None:
        self.values: list[tuple[float, bool]] = []
        self.n_resets = 0

    def update(self, value: float, censored: bool = False) -> DriftSignal:
        self.values.append((value, censored))
        return DriftSignal(drift=value > 1.0, score=value)

    def reset(self) -> None:
        self.n_resets += 1


class PredictingSelector(AbstractOnlineSelector):
    """Always picks ``a`` and predicts a runtime of 5 for it; counts resets."""

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kw: Any
    ) -> None:
        self.n_resets = 0

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        return PredictionRecord(
            chosen="a",
            predicted_runtime=5.0,
            uncertainty=0.5,
            scores={"a": 1.0, "b": 0.0},
        )

    def _observe(self, observation: Observation) -> None:
        pass

    def _reset_model(self) -> None:
        self.n_resets += 1


class FixedGate(AbstractExtraRunGate):
    """Always answers ``answer``; records the contexts it saw."""

    def __init__(self, answer: Sequence[str], needs_score: bool = False) -> None:
        self.answer = list(answer)
        self.needs_instance_score = needs_score
        self.contexts: list[GateContext] = []

    def extras(self, context: GateContext) -> Sequence[str]:
        self.contexts.append(context)
        return self.answer
