"""Tests for the gate criteria and rate-based thresholds (A3)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.drift import AbstractDriftDetector, DriftSignal
from asf.online.evaluation import describe, run_prequential
from asf.online.gate import (
    AbstractExtraRunGate,
    AlwaysProbe,
    GateContext,
    NeverProbe,
    ScoreGate,
    UncertaintyGate,
)
from asf.online.gate_criteria import (
    CloseCallGate,
    TimeoutRiskGate,
    UnfamiliarityGate,
)
from asf.online.prediction import PredictionRecord
from asf.online.selectors import FrugalPairwiseSelector

from .conftest import make_stream

ALGOS = ("a", "b", "c")
_FEATURES = pd.DataFrame({"f0": [0.0]}, index=pd.Index(["x"]))


# ---------------------------------------------------------------- helpers
def _ctx(
    uncertainty: float | None = None,
    scores: Mapping[str, float] | None = None,
    predicted_runtime: float | None = None,
    candidates: tuple[str, ...] = ("b", "c"),
    instance_score: float | None = None,
) -> GateContext:
    record = PredictionRecord(
        chosen="a",
        uncertainty=uncertainty,
        scores=scores,
        predicted_runtime=predicted_runtime,
        candidates=candidates,
    )
    return GateContext(
        prediction=record,
        candidates=candidates,
        algorithms=ALGOS,
        features=_FEATURES,
        step=0,
        instance_score=instance_score,
    )


class _FixedRng:
    """Stands in for the gate's generator: every draw is ``u``."""

    def __init__(self, u: float) -> None:
        self.u = u

    def random(self) -> float:
        return self.u


def _rate_gate(
    q: float, window: list[float], u: float, window_size: int = 200
) -> UncertaintyGate:
    gate = UncertaintyGate(rate=q, window_size=window_size)
    gate._window.extend(window)
    gate._rng = _FixedRng(u)  # type: ignore[assignment]
    return gate


def _fires(gate: ScoreGate, score: float | None) -> bool:
    return len(gate.extras(_ctx(uncertainty=score))) > 0


def _decisions(gate: ScoreGate, scores: list[float | None]) -> np.ndarray:
    return np.array([_fires(gate, s) for s in scores])


# ------------------------------------------------------------ rate rule
@pytest.mark.parametrize(
    ("score", "u", "expected"),
    [
        # Window [1, 2, 3, 4], q = 0.5; rule: (greater + u * (equal + 1)) / 5 < q.
        (5.0, 0.99, True),  # (0 + 0.99) / 5 = 0.198
        (3.5, 0.99, True),  # (1 + 0.99) / 5 = 0.398
        (2.5, 0.49, True),  # (2 + 0.49) / 5 = 0.498
        (2.5, 0.50, False),  # (2 + 0.50) / 5 = 0.5, not below q
        (1.5, 0.0, False),  # (3 + 0) / 5 = 0.6
        (0.0, 0.0, False),  # (4 + 0) / 5 = 0.8
    ],
)
def test_rate_rule_hand_cases(score: float, u: float, expected: bool) -> None:
    gate = _rate_gate(0.5, [1.0, 2.0, 3.0, 4.0], u)
    assert _fires(gate, score) is expected
    assert gate.last_score_ == score


@pytest.mark.parametrize(("u", "expected"), [(0.49, True), (0.5, False)])
def test_rate_rule_breaks_ties_at_random(u: float, expected: bool) -> None:
    # All five scores equal: (0 + u * 5) / 5 = u, so the gate fires iff u < q.
    assert _fires(_rate_gate(0.5, [3.0] * 4, u), 3.0) is expected


@pytest.mark.parametrize(("u", "expected"), [(0.29, True), (0.3, False)])
def test_rate_rule_without_a_score_fires_with_probability_q(
    u: float, expected: bool
) -> None:
    gate = _rate_gate(0.3, [1.0, 2.0], u)
    assert _fires(gate, None) is expected
    assert _fires(gate, float("nan")) is expected  # NaN = no score
    assert list(gate._window) == [1.0, 2.0]  # unscored instances are not stored


def test_first_instance_fires_with_probability_q() -> None:
    assert _fires(_rate_gate(0.3, [], 0.29), 7.0)
    assert not _fires(_rate_gate(0.3, [], 0.3), 7.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_rate_endpoints_always_and_never(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores: list[float | None] = [None, 1.0, 1.0, None, *rng.normal(size=200), None]
    always = UncertaintyGate(rate=1.0, random_state=seed)
    never = UncertaintyGate(rate=0.0, random_state=seed)
    assert _decisions(always, scores).all()
    assert not _decisions(never, scores).any()


@pytest.mark.parametrize("q", [0.1, 0.3, 0.7])
@pytest.mark.parametrize("kind", ["continuous", "constant", "discrete", "missing"])
def test_realised_rate_is_q(q: float, kind: str) -> None:
    rng = np.random.default_rng(42)
    n = 4000
    scores: list[float | None] = {
        "continuous": list(rng.normal(size=n)),
        "constant": [0.5] * n,
        "discrete": [float(v) for v in rng.integers(0, 3, size=n)],  # vote gaps
        "missing": [None] * n,
    }[kind]
    fired = _decisions(UncertaintyGate(rate=q, random_state=0), scores)
    # Standard error of a rate over 4000 instances is below 0.008.
    assert fired.mean() == pytest.approx(q, abs=0.03)


def test_rate_fires_on_the_highest_scores() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=3000)
    fired = _decisions(UncertaintyGate(rate=0.2, random_state=0), list(scores))
    assert scores[fired].min() > np.quantile(scores, 0.5)
    assert scores[fired].mean() > scores[~fired].mean() + 1.0


def test_rate_uses_only_the_last_window_size_scores() -> None:
    gate = UncertaintyGate(rate=0.3, window_size=3, random_state=0)
    _decisions(gate, [10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    assert list(gate._window) == [1.0, 1.0, 1.0]
    gate._rng = _FixedRng(0.99)  # type: ignore[assignment]
    # 2 beats the whole window: (0 + 0.99) / 4 < 0.3. With the old 10s it would not.
    assert _fires(gate, 2.0)


def test_rate_rises_after_a_jump_and_returns_to_q() -> None:
    rng = np.random.default_rng(5)
    before = list(rng.normal(0.0, 1.0, size=300))
    after = list(rng.normal(20.0, 1.0, size=1500))
    gate = UncertaintyGate(rate=0.2, window_size=50, random_state=0)
    _decisions(gate, before)
    fired = _decisions(gate, after)
    # The j-th instance after the jump has at most j - 1 greater scores in the
    # window, so (j - 1 + u) / 51 < 0.2 holds for the first 10 for any u.
    assert fired[:10].all()
    assert fired[500:].mean() == pytest.approx(0.2, abs=0.05)


def test_rate_is_deterministic_and_reset_restarts() -> None:
    scores = [float(v) for v in np.random.default_rng(1).integers(0, 4, size=300)]
    first = _decisions(UncertaintyGate(rate=0.4, random_state=7), scores)
    gate = UncertaintyGate(rate=0.4, random_state=7)
    np.testing.assert_array_equal(_decisions(gate, scores), first)
    gate.reset()
    assert len(gate._window) == 0 and gate.last_score_ is None
    np.testing.assert_array_equal(_decisions(gate, scores), first)
    other = _decisions(UncertaintyGate(rate=0.4, random_state=8), scores)
    assert (other != first).any()


def test_threshold_rule_keeps_no_state_and_ignores_missing_scores() -> None:
    gate = UncertaintyGate(0.3)
    assert gate.threshold == 0.3 and gate.rate is None and gate.n == 1
    assert _decisions(gate, [0.31, 0.3, 0.29, None, float("nan")]).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert len(gate._window) == 0


# ------------------------------------------------------- which extras run
def test_extras_are_the_first_n_nominees() -> None:
    assert list(UncertaintyGate(0.0).extras(_ctx(0.1))) == ["b"]
    assert list(UncertaintyGate(0.0, n=2).extras(_ctx(0.1))) == ["b", "c"]
    assert list(UncertaintyGate(0.0, n=None).extras(_ctx(0.1))) == ["b", "c"]
    assert list(UncertaintyGate(0.0, n=0).extras(_ctx(0.1))) == []
    assert list(
        UncertaintyGate(rate=1.0, n=1).extras(_ctx(0.1, candidates=("c",)))
    ) == ["c"]


def test_extras_fall_back_to_ranked_alternatives_without_nominees() -> None:
    # A bandit nominates nothing: the best-scored other algorithms are used.
    ctx = _ctx(scores={"a": 3.0, "b": 1.0, "c": 2.0}, candidates=())
    assert list(CloseCallGate(rate=1.0).extras(ctx)) == ["c"]
    assert list(CloseCallGate(rate=1.0, n=None).extras(ctx)) == ["c", "b"]
    assert list(CloseCallGate(rate=0.0, n=None).extras(ctx)) == []


# ------------------------------------------------------------- criteria
def test_close_call_gap_of_the_two_best_scores() -> None:
    gate = CloseCallGate(max_gap=1.0)
    # Votes a=3, b=1, c=2: the two best differ by 1.
    assert gate._score(_ctx(scores={"a": 3.0, "b": 1.0, "c": 2.0})) == -1.0
    assert list(gate.extras(_ctx(scores={"a": 3.0, "b": 1.0, "c": 2.0}))) == ["b"]
    assert list(gate.extras(_ctx(scores={"a": 3.0, "b": 1.0, "c": 1.9}))) == []
    assert list(CloseCallGate(max_gap=0.0).extras(_ctx(scores={"a": 2, "c": 2}))) == [
        "b"
    ]
    # The chosen algorithm need not be among the two best.
    assert gate._score(_ctx(scores={"a": 0.0, "b": 5.0, "c": 4.5})) == -0.5


def test_close_call_relative_gap() -> None:
    gate = CloseCallGate(max_gap=0.2, relative=True)
    assert gate._score(_ctx(scores={"a": 100.0, "b": 80.0})) == pytest.approx(-0.2)
    # Negated predicted runtimes 10 s and 12.5 s: gap 2.5 of 12.5.
    assert gate._score(_ctx(scores={"a": -10.0, "b": -12.5})) == pytest.approx(-0.2)
    assert gate._score(_ctx(scores={"a": 0.0, "b": 0.0})) == 0.0
    absolute = CloseCallGate(max_gap=0.2)
    assert absolute._score(_ctx(scores={"a": 100.0, "b": 80.0})) == -20.0


def test_close_call_ignores_non_finite_and_needs_two_scores() -> None:
    gate = CloseCallGate(max_gap=10.0)
    inf = float("inf")
    assert gate._score(_ctx(scores={"a": 2.0, "b": -inf, "c": 2.0})) == 0.0
    assert gate._score(_ctx(scores={"a": 2.0, "b": float("nan"), "c": 1.0})) == -1.0
    assert gate._score(_ctx(scores={"a": 2.0, "b": -inf})) is None
    assert gate._score(_ctx(scores=None)) is None
    assert list(gate.extras(_ctx(scores={"a": 2.0}))) == []  # unrated: no extras


def test_unfamiliarity_gate_uses_the_instance_score() -> None:
    gate = UnfamiliarityGate(0.5)
    assert gate.needs_instance_score is True
    assert not UncertaintyGate(0.5).needs_instance_score
    assert list(gate.extras(_ctx(instance_score=0.6))) == ["b"]
    assert list(gate.extras(_ctx(instance_score=0.5))) == ["b"]
    assert list(gate.extras(_ctx(instance_score=0.4))) == []
    assert list(gate.extras(_ctx(instance_score=None))) == []
    assert list(gate.extras(_ctx(uncertainty=9.0, instance_score=0.1))) == []


def test_timeout_risk_gate_uses_the_predicted_runtime() -> None:
    gate = TimeoutRiskGate(80.0)  # 0.8 * cutoff of 100
    assert list(gate.extras(_ctx(predicted_runtime=90.0))) == ["b"]
    assert list(gate.extras(_ctx(predicted_runtime=80.0))) == ["b"]
    assert list(gate.extras(_ctx(predicted_runtime=79.0))) == []
    for missing in (None, float("nan"), float("inf")):
        assert gate._score(_ctx(predicted_runtime=missing)) is None
        assert list(gate.extras(_ctx(predicted_runtime=missing))) == []
    # Under a rate the slowest predictions get the extras.
    ranked = TimeoutRiskGate(rate=0.5, random_state=0)
    ranked._window.extend([10.0, 20.0, 30.0, 40.0])
    assert list(ranked.extras(_ctx(predicted_runtime=50.0))) == ["b"]
    assert list(ranked.extras(_ctx(predicted_runtime=5.0))) == []


# ------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither threshold nor rate
        {"threshold": 0.2, "rate": 0.5},
        {"rate": -0.1},
        {"rate": 1.1},
        {"threshold": 0.2, "n": -1},
        {"rate": 0.5, "window_size": 0},
    ],
)
def test_score_gate_rejects_bad_arguments(kwargs: dict[str, Any]) -> None:
    for cls in (UncertaintyGate, UnfamiliarityGate, TimeoutRiskGate):
        with pytest.raises(ValueError):
            cls(**kwargs)


def test_close_call_gate_rejects_bad_arguments() -> None:
    for kwargs in ({}, {"max_gap": 1.0, "rate": 0.5}, {"max_gap": -1.0}):
        with pytest.raises(ValueError):
            CloseCallGate(**kwargs)  # type: ignore[arg-type]


def test_gates_are_described_for_result_files() -> None:
    desc = describe(CloseCallGate(rate=0.25, relative=True, random_state=3))
    assert desc["class"] == "asf.online.gate_criteria.CloseCallGate"
    assert desc["params"] == {
        "max_gap": None,
        "n": 1,
        "relative": True,
        "rate": 0.25,
        "window_size": 200,
        "random_state": 3,
    }
    assert describe(UncertaintyGate(0.25))["params"]["threshold"] == 0.25


# ------------------------------------------------------- inside a selector
class _Toy(AbstractOnlineSelector):
    """Picks ``a``, nominates ``b``; uncertainty and runtime follow ``f0``."""

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kw: Any
    ) -> None:
        pass

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        f0 = float(features["f0"].iloc[0])
        return PredictionRecord(
            chosen="a",
            predicted_runtime=50.0 + 10.0 * f0,
            uncertainty=abs(f0),
            scores={"a": 1.0, "b": 1.0 - abs(f0)},
            candidates=("b",),
        )

    def _observe(self, observation: Observation) -> None:
        pass


class _RecordingGate(AbstractExtraRunGate):
    """Runs ``b`` on every second instance and records the feedback."""

    def __init__(self) -> None:
        self.seen: list[tuple[PredictionRecord, Observation]] = []
        self.n_resets = 0
        self.n_decisions = 0

    def extras(self, context: GateContext) -> tuple[str, ...]:
        self.n_decisions += 1
        return ("b",) if context.step % 2 == 0 else ()

    def observe(self, prediction: PredictionRecord, observation: Observation) -> None:
        assert len(self.seen) == self.n_decisions - 1  # after the decision
        self.seen.append((prediction, observation))

    def reset(self) -> None:
        self.n_resets += 1


class _ConstantScore(AbstractDriftDetector):
    """Feature detector whose per-instance score is ``f0``."""

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "_ConstantScore":
        return self

    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        return DriftSignal(drift=False)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return features["f0"].to_numpy(dtype=float)


def test_gate_sees_the_outcome_of_every_instance() -> None:
    gate = _RecordingGate()
    stream = make_stream()
    result = run_prequential(_Toy(gate=gate), stream, n_warmstart=10)
    assert gate.n_resets == 1  # at fit
    assert len(gate.seen) == len(result.steps) == 110
    for step, (prediction, observation) in enumerate(gate.seen):
        extras = ("b",) if step % 2 == 0 else ()
        assert prediction.extras == extras
        assert prediction.instance_id == observation.instance_id
        assert set(observation.runtimes) == {"a", *extras}
    assert [o.instance_id for _, o in gate.seen] == list(stream.features.index[10:])


def test_gate_gets_no_feedback_without_a_decision() -> None:
    gate = _RecordingGate()
    stream = make_stream()
    selector = _Toy(gate=gate)
    selector.fit(stream.features.iloc[:5], stream.performance.iloc[:5])
    selector.observe(
        Observation("i5", stream.features.iloc[[5]], {"a": 1.0}, cutoff=100.0)
    )
    assert gate.seen == []


def test_refitting_a_selector_resets_its_gate() -> None:
    stream = make_stream()
    selector = _Toy(gate=UncertaintyGate(rate=0.3, random_state=0))
    first = run_prequential(selector, stream, n_warmstart=10).steps["extras"]
    second = run_prequential(selector, stream, n_warmstart=10).steps["extras"]
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize(
    "gate_class", [UncertaintyGate, CloseCallGate, UnfamiliarityGate, TimeoutRiskGate]
)
def test_rate_sweep_in_a_selector(gate_class: type[ScoreGate]) -> None:
    stream = make_stream(n=600, drift_at=10_000)
    rates = {}
    for q in (0.0, 0.25, 1.0):
        selector = _Toy(
            gate=gate_class(rate=q, random_state=0), detector=_ConstantScore()
        )
        steps = run_prequential(selector, stream, n_warmstart=10).steps
        rates[q] = float((steps["n_runs"] > 1).mean())
        if gate_class is UnfamiliarityGate:
            assert steps["instance_score"].notna().all()
        else:
            assert steps["instance_score"].isna().all()  # only computed on request
    assert rates[0.0] == 0.0 and rates[1.0] == 1.0
    assert rates[0.25] == pytest.approx(0.25, abs=0.06)


def test_rated_instances_get_the_extras_in_a_selector() -> None:
    stream = make_stream(n=600, drift_at=10_000)
    f0 = stream.features["f0"].to_numpy()[10:]
    for gate, rating in (
        (UncertaintyGate(rate=0.2, random_state=0), np.abs(f0)),
        (CloseCallGate(rate=0.2, random_state=0), np.abs(f0)),
        (UnfamiliarityGate(rate=0.2, random_state=0), f0),
        (TimeoutRiskGate(rate=0.2, random_state=0), f0),
    ):
        selector = _Toy(gate=gate, detector=_ConstantScore())
        steps = run_prequential(selector, stream, n_warmstart=10).steps
        extra = (steps["n_runs"] > 1).to_numpy()
        if isinstance(gate, CloseCallGate):  # small gap = small |f0|
            assert rating[extra].mean() < rating[~extra].mean()
        else:
            assert rating[extra].mean() > rating[~extra].mean()


# -------------------------------------------------- frugal pairwise selector
def _frugal_steps(gate: AbstractExtraRunGate) -> pd.DataFrame:
    selector = FrugalPairwiseSelector(
        model_class=DecisionTreeClassifier, gate=gate, random_state=0
    )
    result = run_prequential(selector, make_stream(), n_warmstart=20)
    return result.steps.drop(columns=["selector_seconds", "detector_seconds"])


def test_rate_one_is_always_two_and_rate_zero_is_never() -> None:
    always = _frugal_steps(UncertaintyGate(rate=1.0, random_state=0))
    pd.testing.assert_frame_equal(always, _frugal_steps(AlwaysProbe(1)))
    assert (always["n_runs"] == 2).all()
    never = _frugal_steps(UncertaintyGate(rate=0.0, random_state=0))
    pd.testing.assert_frame_equal(never, _frugal_steps(NeverProbe()))
    assert (never["n_runs"] == 1).all()


def test_frugal_selector_with_a_rate() -> None:
    steps = _frugal_steps(UncertaintyGate(rate=0.3, random_state=0))
    assert (steps["n_runs"] > 1).mean() == pytest.approx(0.3, abs=0.15)  # 100 steps
    pd.testing.assert_frame_equal(
        steps, _frugal_steps(UncertaintyGate(rate=0.3, random_state=0))
    )
