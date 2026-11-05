"""Tests for the Rook-et-al.-style online pairwise selector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asf.online.base import Observation
from asf.online.drift import AbstractDriftDetector, DriftSignal
from asf.online.evaluation import run_prequential
from asf.online.retraining import PeriodicRetrain
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams.stream import InstanceStream

ALGOS = ["a", "b", "c"]


def covariate_shift_stream(
    n_before: int = 150, n_after: int = 150, seed: int = 0
) -> InstanceStream:
    """Region 1 (f0 ~ N(0,1)): a is best. Region 2 (f0 ~ N(8,1)): b is best."""
    rng = np.random.default_rng(seed)
    f0 = np.concatenate([rng.normal(0, 1, n_before), rng.normal(8, 1, n_after)])
    x = np.column_stack([f0, rng.normal(size=len(f0))])
    perf = np.where(
        (np.arange(len(f0)) < n_before)[:, None], [[5, 50, 20]], [[50, 5, 20]]
    ).astype(float)
    idx = pd.Index([f"i{k}" for k in range(len(f0))])
    return InstanceStream(
        pd.DataFrame(x, index=idx, columns=["f0", "f1"]),
        pd.DataFrame(perf, index=idx, columns=ALGOS),
        cutoff=1000.0,
        drift_points=(n_before,),
    )


class _ScriptedDetector(AbstractDriftDetector):
    """Signals drift at fixed update counts; records what it was fitted on."""

    def __init__(self, drift_at: set[int]) -> None:
        self.drift_at = drift_at
        self.n_updates = 0
        self.fit_sizes: list[int] = []

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "_ScriptedDetector":
        self.fit_sizes.append(len(features))
        return self

    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        self.n_updates += 1
        return DriftSignal(drift=self.n_updates in self.drift_at)


def _observe_all(sel: OnlinePairwiseSelector, row: pd.DataFrame, rts: dict) -> None:
    sel.observe(Observation(row.index[0], row, rts, frozenset(), 1000.0))


def test_reference_period_samples_random_pairs_then_trains() -> None:
    stream = covariate_shift_stream()
    sel = OnlinePairwiseSelector(reference_period=20, random_state=0)
    sel.fit(stream.features.iloc[:0], stream.performance.iloc[:0])
    seen_pairs = set()
    for item in list(stream)[:20]:
        sched = sel.propose(item.features)
        algos = [a for a, _ in sched]
        assert len(set(algos)) == 2  # selected + 1 additional, no replacement
        seen_pairs.add(frozenset(algos))
        _observe_all(
            sel,
            item.features,
            {a: stream.true_runtimes(item.instance_id)[a] for a in algos},
        )
    assert len(seen_pairs) == 3  # all three pairs get explored
    assert sel._reference_remaining == 0
    assert sel.retrain_steps == []  # initial fit is not a retrain
    assert sel.buffer is not None and sel.buffer.window_size is None


def test_pair_without_comparison_uses_random_predictor() -> None:
    stream = covariate_shift_stream()
    sel = OnlinePairwiseSelector(reference_period=1, random_state=0)
    sel.fit(stream.features.iloc[:0], stream.performance.iloc[:0])
    item = next(iter(stream))
    sel.propose(item.features)
    _observe_all(sel, item.features, {"a": 1.0, "b": 2.0})  # only pair (a,b) seen
    assert type(sel._models[(0, 2)]).__name__ == "_RandomBinaryPredictor"
    assert type(sel._models[(1, 2)]).__name__ == "_RandomBinaryPredictor"
    assert type(sel._models[(0, 1)]).__name__ != "_RandomBinaryPredictor"


def test_labels_follow_at_least_as_good_with_ties_to_first() -> None:
    sel = OnlinePairwiseSelector()
    labels = sel._labels(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 1.0]))
    assert list(labels) == [1, 1, 0]
    sel_max = OnlinePairwiseSelector(maximize=True)
    assert list(
        sel_max._labels(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 1.0]))
    ) == [0, 1, 1]


def test_error_guided_probabilities_are_inverse_to_accuracy() -> None:
    stream = covariate_shift_stream()
    sel = OnlinePairwiseSelector(reference_period=0, random_state=0)
    sel.fit(stream.features.iloc[:0], stream.performance.iloc[:0])
    sel.pair_accuracy_[0, 1] = sel.pair_accuracy_[1, 0] = 0.5
    sel.pair_accuracy_[0, 2] = sel.pair_accuracy_[2, 0] = 1.0
    p = sel._extra_probabilities(0, np.array([1, 2]))
    assert p is not None and np.allclose(p, [2 / 3, 1 / 3])
    sel.additional_strategy = "uniform"
    assert sel._extra_probabilities(0, np.array([1, 2])) is None


def test_warm_start_trains_immediately() -> None:
    stream = covariate_shift_stream()
    sel = OnlinePairwiseSelector(random_state=0)
    sel.fit(stream.features.iloc[:60], stream.performance.iloc[:60])
    assert sel._reference_remaining == 0
    pred = sel.predict(stream.features.iloc[:5])
    assert all(v[0][0] == "a" for v in pred.values())  # region 1: a is best


def test_drift_signal_retrains_on_full_history_and_resets_detector() -> None:
    stream = covariate_shift_stream()
    detector = _ScriptedDetector(drift_at={30})
    sel = OnlinePairwiseSelector(reference_period=40, detector=detector, random_state=0)
    res = run_prequential(sel, stream, n_warmstart=0)
    assert res.drift_steps == res.retrain_steps and len(res.drift_steps) == 1
    # detector fitted after the reference period (40 rows), then reset on ALL
    # history seen up to and including the instance that triggered the retrain
    assert detector.fit_sizes[0] == 40
    assert detector.fit_sizes[1] == 40 + 30
    assert res.steps["retrained"].sum() == 1


def test_retraining_recovers_from_covariate_shift() -> None:
    stream = covariate_shift_stream()
    kw = dict(reference_period=40, random_state=0)
    static = run_prequential(OnlinePairwiseSelector(**kw), stream)  # never retrains
    # Two signals (~30 and ~80 instances into region 2). A single retraining is
    # not enough here: right after the shift only pairs involving the selected
    # algorithm get compared, so the pair (b, c) still carries only region-1
    # evidence (c beats b) until a later retraining sees the new comparisons.
    detector = _ScriptedDetector(drift_at={140, 190})
    adaptive = run_prequential(OnlinePairwiseSelector(detector=detector, **kw), stream)
    last = slice(len(static.steps) - 50, None)
    assert adaptive.steps["user_cost"].iloc[last].mean() == pytest.approx(5.0, abs=0.5)
    assert (
        adaptive.steps["user_cost"].iloc[last].mean()
        < static.steps["user_cost"].iloc[last].mean()
    )
    assert static.summary()["n_retrains"] == 0 and len(adaptive.retrain_steps) == 2


def test_periodic_policy_without_detector() -> None:
    stream = covariate_shift_stream(60, 60)
    sel = OnlinePairwiseSelector(
        reference_period=20, retrain_policy=PeriodicRetrain(every=25), random_state=0
    )
    res = run_prequential(sel, stream)
    assert len(res.retrain_steps) >= 3 and res.drift_steps == []


def test_best_of_scheduled_is_default_and_two_runs_per_instance() -> None:
    stream = covariate_shift_stream(60, 60)
    sel = OnlinePairwiseSelector(reference_period=20, random_state=0)
    res = run_prequential(sel, stream)
    assert (res.steps["n_runs"] == 2).all()
    assert (res.steps["user_cost"] <= res.steps["compute_cost"]).all()


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        OnlinePairwiseSelector(n_additional=0)
    with pytest.raises(ValueError):
        OnlinePairwiseSelector(additional_strategy="nope")  # type: ignore[arg-type]
    stream = covariate_shift_stream()
    two = stream.performance[["a", "b"]]
    with pytest.raises(ValueError):
        OnlinePairwiseSelector(n_additional=2).fit(
            stream.features.iloc[:0], two.iloc[:0]
        )
