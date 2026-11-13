"""F1 interfaces: prediction record, extra-run gate, detectors, drift responses."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.drift import (
    AbstractDriftDetector,
    AbstractScalarDriftDetector,
    DriftSignal,
    LogRuntimeResidual,
    ScalarSample,
)
from asf.online.evaluation import run_prequential
from asf.online.gate import (
    AbstractExtraRunGate,
    AlwaysProbe,
    GateContext,
    NeverProbe,
)
from asf.online.prediction import PredictionRecord
from asf.online.retraining import (
    PeriodicRetrain,
    RaiseExplorationResponse,
    ResetResponse,
    RetrainResponse,
    ReweightResponse,
)
from asf.online.selectors import (
    DiscountedTSSelector,
    OnlinePairwiseSelector,
    StaticOnlineSelector,
    ThompsonSamplingSelector,
)
from asf.online.streams.stream import InstanceStream
from asf.selectors import PairwiseClassifier

from .conftest import CUTOFF, make_stream


# ---------------------------------------------------------------- helpers
class _ScriptedDetector(AbstractDriftDetector):
    """Feature detector that signals drift at given update counts."""

    def __init__(self, drift_at: set[int]) -> None:
        self.drift_at = drift_at
        self.n_updates = 0
        self.fit_sizes: list[int] = []
        self.n_resets = 0

    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "_ScriptedDetector":
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


class _ScalarRecorder(AbstractScalarDriftDetector):
    """Scalar detector recording its inputs; drifts when a value exceeds 1."""

    def __init__(self) -> None:
        self.values: list[tuple[float, bool]] = []
        self.n_resets = 0

    def update(self, value: float, censored: bool = False) -> DriftSignal:
        self.values.append((value, censored))
        return DriftSignal(drift=value > 1.0, score=value)

    def reset(self) -> None:
        self.n_resets += 1


class _Predicting(AbstractOnlineSelector):
    """Always picks ``a`` and predicts a runtime of 5 for it."""

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


class _FixedGate(AbstractExtraRunGate):
    def __init__(self, answer: Sequence[str], needs_score: bool = False) -> None:
        self.answer = list(answer)
        self.needs_instance_score = needs_score
        self.contexts: list[GateContext] = []

    def extras(self, context: GateContext) -> Sequence[str]:
        self.contexts.append(context)
        return self.answer


def three_algo_stream(n: int = 80) -> InstanceStream:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 2))
    perf = np.tile([5.0, 50.0, 20.0], (n, 1))
    idx = pd.Index([f"i{k}" for k in range(n)])
    return InstanceStream(
        pd.DataFrame(x, index=idx, columns=["f0", "f1"]),
        pd.DataFrame(perf, index=idx, columns=["a", "b", "c"]),
        cutoff=CUTOFF,
    )


# ------------------------------------------------------- prediction record
def test_prediction_record_of_rook_and_bandit() -> None:
    stream = three_algo_stream()
    rook = OnlinePairwiseSelector(reference_period=10, random_state=0)
    res = run_prequential(rook, stream)
    record = rook.last_prediction_
    assert record is not None and record.instance_id == stream.features.index[-1]
    assert len(record.extras) == 1 and record.extras == record.candidates
    assert record.scores is not None and record.score == record.scores[record.chosen]
    assert res.steps["pred_score"].iloc[10:].notna().all()  # after the reference
    assert res.steps["pred_score"].iloc[:10].isna().all()  # random sampling
    assert res.steps["pred_runtime"].isna().all()

    bandit = ThompsonSamplingSelector(budget=CUTOFF, random_state=0)
    res = run_prequential(bandit, stream, n_warmstart=5)
    assert bandit.last_prediction_ is not None
    assert bandit.last_prediction_.extras == ()
    assert (res.steps["n_runs"] == 1).all()
    assert res.steps["pred_score"].notna().all()


def test_custom_record_is_logged() -> None:
    res = run_prequential(_Predicting(), make_stream(), n_warmstart=10)
    assert (res.steps["pred_runtime"] == 5.0).all()
    assert (res.steps["pred_uncertainty"] == 0.5).all()


# ---------------------------------------------------------------- gates
def test_default_gates_reproduce_built_in_extras() -> None:
    assert isinstance(OnlinePairwiseSelector().gate_, AlwaysProbe)
    stream = three_algo_stream()
    sel = OnlinePairwiseSelector(n_additional=2, reference_period=10, random_state=0)
    assert (run_prequential(sel, stream).steps["n_runs"] == 3).all()


def test_never_probe_turns_rook_into_single_runs() -> None:
    sel = OnlinePairwiseSelector(reference_period=10, gate=NeverProbe(), random_state=0)
    steps = run_prequential(sel, three_algo_stream()).steps
    assert (steps["n_runs"] == 1).all()
    assert (steps["extras"] == "[]").all()


def test_gate_can_add_extras_to_a_bandit_and_answers_are_cleaned() -> None:
    gate = _FixedGate(["c", "c", "a", "b"])  # duplicates and possibly primary
    sel = ThompsonSamplingSelector(budget=CUTOFF, gate=gate, random_state=0)
    steps = run_prequential(sel, three_algo_stream(), n_warmstart=5).steps
    assert (steps["n_runs"] == 3).all()
    ctx = gate.contexts[3]
    assert ctx.step == 3 and ctx.candidates == ()
    alternatives = ctx.ranked_alternatives()
    assert sorted(alternatives + (ctx.prediction.chosen,)) == ["a", "b", "c"]
    scores = ctx.prediction.scores
    assert scores is not None
    assert [scores[a] for a in alternatives] == sorted(
        (scores[a] for a in alternatives), reverse=True
    )


def test_gate_with_unknown_algorithm_fails() -> None:
    sel = ThompsonSamplingSelector(budget=CUTOFF, gate=_FixedGate(["nope"]))
    with pytest.raises(ValueError, match="unknown algorithms"):
        run_prequential(sel, three_algo_stream(), n_warmstart=5)


def test_extras_use_the_primary_budget() -> None:
    class _Budgeted(_Predicting):
        def _select(self, features: pd.DataFrame) -> PredictionRecord:
            return PredictionRecord(chosen="a", budget=7.0, candidates=("b",))

    sel = _Budgeted()
    sel.fit(make_stream().features.iloc[:5], make_stream().performance.iloc[:5])
    assert sel.propose(make_stream().features.iloc[[6]]) == [("a", 7.0), ("b", 7.0)]


def test_gate_receives_instance_score_and_drift_context() -> None:
    gate = _FixedGate([], needs_score=True)
    detector = _ScriptedDetector(drift_at={3})
    sel = ThompsonSamplingSelector(
        budget=CUTOFF, gate=gate, detector=detector, random_state=0
    )
    steps = run_prequential(sel, make_stream(), n_warmstart=10).steps
    assert gate.contexts[0].instance_score == 0.25
    assert (steps["instance_score"] == 0.25).all()
    assert gate.contexts[2].steps_since_drift is None
    assert gate.contexts[3].steps_since_drift == 1  # drift on the 3rd update
    assert gate.contexts[3].last_signal is not None


# ------------------------------------------------------------- detectors
def test_feature_detector_needs_reference_size_on_cold_start() -> None:
    sel = ThompsonSamplingSelector(
        budget=CUTOFF, detector=_ScriptedDetector(set()), random_state=0
    )
    with pytest.raises(ValueError, match="detector_reference_size"):
        run_prequential(sel, make_stream())

    detector = _ScriptedDetector(set())
    sel = ThompsonSamplingSelector(
        budget=CUTOFF,
        detector=detector,
        detector_reference_size=25,
        window_size=None,
        random_state=0,
    )
    steps = run_prequential(sel, make_stream()).steps
    assert detector.fit_sizes == [25]
    assert detector.n_updates == len(steps) - 25
    assert steps["detector_score"].iloc[:25].isna().all()
    assert (steps["detector_score"].iloc[25:] == 1.0).all()


def test_warm_start_fits_feature_detector_on_the_prefix() -> None:
    detector = _ScriptedDetector(set())
    sel = ThompsonSamplingSelector(budget=CUTOFF, detector=detector, random_state=0)
    run_prequential(sel, make_stream(), n_warmstart=30)
    assert detector.fit_sizes == [30]


def test_scalar_detector_requires_statistic() -> None:
    with pytest.raises(ValueError, match="drift_statistic"):
        _Predicting(detector=_ScalarRecorder())
    with pytest.raises(ValueError, match="scalar"):
        _Predicting(drift_statistic=LogRuntimeResidual())
    with pytest.raises(TypeError):
        _Predicting(detector=object())  # type: ignore[arg-type]


def test_scalar_detector_gets_log_residual() -> None:
    detector = _ScalarRecorder()
    sel = _Predicting(detector=detector, drift_statistic=LogRuntimeResidual())
    stream = make_stream()
    steps = run_prequential(sel, stream, n_warmstart=10).steps
    observed = stream.performance["a"].iloc[10:].to_numpy()
    expected = np.log(observed) - math.log(5.0)
    assert np.allclose([v for v, _ in detector.values], expected)
    assert np.allclose(steps["drift_statistic"], expected)
    # a is 50 instead of 5 on about half of the instances -> log(10) > 1
    assert steps["drift_signal"].sum() == (observed == 50.0).sum()


def test_log_residual_censoring_and_missing_prediction() -> None:
    obs = Observation("i", pd.DataFrame({"f": [0.0]}), {"a": 10.0}, frozenset({"a"}))
    record = PredictionRecord(chosen="a", predicted_runtime=1.0)
    emit = LogRuntimeResidual()(obs, record)
    assert emit == ScalarSample(value=math.log(10.0), censored=True)
    assert LogRuntimeResidual(censored="skip")(obs, record) is None
    assert LogRuntimeResidual()(obs, PredictionRecord(chosen="a")) is None
    assert LogRuntimeResidual()(obs, None) is None
    zero = Observation("i", pd.DataFrame({"f": [0.0]}), {"a": 0.0})
    floor = LogRuntimeResidual(floor=1e-3)(zero, record)
    assert floor is not None and floor.value == pytest.approx(math.log(1e-3))


# -------------------------------------------------------- drift responses
def test_reset_response_resets_bandit_and_rereferences_detector() -> None:
    detector = _ScriptedDetector(drift_at={5})
    sel = DiscountedTSSelector(
        budget=CUTOFF,
        detector=detector,
        drift_response=ResetResponse(),
        random_state=0,
    )
    stream = make_stream()
    res = run_prequential(sel, stream, n_warmstart=20)
    assert res.drift_steps == [4]
    assert res.steps["response"].tolist().count("reset") == 1
    assert res.steps["response"].iloc[4] == "reset"
    assert detector.n_resets == 1
    # history only holds what was observed after the reset
    assert sum(sel.history_.n_observations(a) for a in ("a", "b")) == len(stream) - 25


def test_custom_selector_reset_hook_and_scalar_rereference() -> None:
    detector = _ScalarRecorder()
    sel = _Predicting(
        detector=detector,
        drift_statistic=LogRuntimeResidual(),
        drift_response=ResetResponse(),
    )
    res = run_prequential(sel, make_stream(), n_warmstart=10)
    n = int(res.steps["drift_signal"].sum())
    assert n > 0 and sel.n_resets == n
    assert detector.n_resets == 1 + n  # once when fitted, once per reset


def test_unsupported_response_or_policy_fails_at_fit() -> None:
    stream = make_stream()
    for kwargs in (
        {"drift_response": RetrainResponse()},
        {"drift_response": ReweightResponse(0.5)},
        {"drift_response": RaiseExplorationResponse(2.0, steps=10)},
        {"retrain_policy": PeriodicRetrain(10)},
    ):
        sel = ThompsonSamplingSelector(budget=CUTOFF, **kwargs)
        with pytest.raises(ValueError, match="does not implement"):
            run_prequential(sel, stream, n_warmstart=10)
    with pytest.raises(TypeError):
        ThompsonSamplingSelector(drift_response=["retrain"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        ReweightResponse(1.5)


def test_rook_defaults_to_retrain_response_and_retrains_once_per_step() -> None:
    assert isinstance(
        OnlinePairwiseSelector(detector=_ScriptedDetector(set())).drift_responses_[0],
        RetrainResponse,
    )
    assert (
        OnlinePairwiseSelector(
            detector=_ScriptedDetector(set()), retrain_policy=PeriodicRetrain(5)
        ).drift_responses_
        == ()
    )

    # drift response and policy fire on the same step: a single retraining
    detector = _ScriptedDetector(drift_at={10})
    sel = OnlinePairwiseSelector(
        reference_period=20,
        detector=detector,
        drift_response=RetrainResponse(),
        retrain_policy=PeriodicRetrain(30),
        random_state=0,
    )
    res = run_prequential(sel, three_algo_stream())
    assert res.drift_steps == [29]
    assert res.retrain_steps == [29, 59]
    assert detector.n_resets == 2
    assert res.steps["retrained"].sum() == 2


# ------------------------------------------------------- protocol & params
def test_protocol_is_checked() -> None:
    stream = make_stream()
    with pytest.raises(ValueError, match="protocol"):
        run_prequential(_Predicting(), stream, n_warmstart=0, protocol="warm")
    with pytest.raises(ValueError, match="protocol"):
        run_prequential(_Predicting(), stream, n_warmstart=5, protocol="cold")
    static = StaticOnlineSelector(PairwiseClassifier(model_class=_Dummy))
    with pytest.raises(ValueError, match="warm start"):
        run_prequential(static, stream)
    res = run_prequential(_Predicting(), stream, n_warmstart=5, protocol="warm")
    assert res.meta["protocol"] == "warm"


class _Dummy:  # never fitted: the cold-start check fails before
    pass


def test_constructor_arguments_are_captured() -> None:
    sel = DiscountedTSSelector(gamma=0.9, budget=50.0, random_state=4)
    assert sel.init_params_["gamma"] == 0.9
    assert sel.init_params_["c"] == 1.0  # default of the outermost class
    assert sel.init_params_["budget"] == 50.0  # flattened **kwargs
    params = sel.describe_params()
    assert params["random_state"] == 4 and params["window_size"] == 200
