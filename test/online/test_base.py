"""The online selector base: prediction record, detectors, drift responses."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from asf.online.base import Observation
from asf.online.drift import (
    LogRuntimeResidual,
)
from asf.online.evaluation import run_prequential
from asf.online.gate import (
    AfterDriftGate,
)
from asf.online.prediction import PredictionRecord
from asf.online.retraining import (
    RaiseExplorationResponse,
    RereferenceResponse,
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
from asf.selectors import PairwiseClassifier

from .conftest import (
    CUTOFF,
    FixedGate,
    PredictingSelector,
    ScalarRecorder,
    ScriptedDetector,
    make_stream,
    three_algo_stream,
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
    res = run_prequential(PredictingSelector(), make_stream(), n_warmstart=10)
    assert (res.steps["pred_runtime"] == 5.0).all()
    assert (res.steps["pred_uncertainty"] == 0.5).all()


# ------------------------------------------------------------- detectors
def test_feature_detector_reference_size_and_warm_start_fit() -> None:
    sel = ThompsonSamplingSelector(
        budget=CUTOFF, detector=ScriptedDetector(set()), random_state=0
    )
    with pytest.raises(ValueError, match="detector_reference_size"):
        run_prequential(sel, make_stream())

    detector = ScriptedDetector(set())
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

    detector = ScriptedDetector(set())  # warm start: fitted on the prefix
    sel = ThompsonSamplingSelector(budget=CUTOFF, detector=detector, random_state=0)
    run_prequential(sel, make_stream(), n_warmstart=30)
    assert detector.fit_sizes == [30]


def test_scalar_detector_requires_statistic() -> None:
    with pytest.raises(ValueError, match="drift_statistic"):
        PredictingSelector(detector=ScalarRecorder())
    with pytest.raises(ValueError, match="scalar"):
        PredictingSelector(drift_statistic=LogRuntimeResidual())
    with pytest.raises(TypeError):
        PredictingSelector(detector=object())  # type: ignore[arg-type]


def test_scalar_detector_gets_log_residual() -> None:
    detector = ScalarRecorder()
    sel = PredictingSelector(detector=detector, drift_statistic=LogRuntimeResidual())
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
    assert emit == (math.log(10.0), True)
    assert LogRuntimeResidual(censored="skip")(obs, record) is None
    assert LogRuntimeResidual()(obs, PredictionRecord(chosen="a")) is None
    assert LogRuntimeResidual()(obs, None) is None
    zero = Observation("i", pd.DataFrame({"f": [0.0]}), {"a": 0.0})
    floor = LogRuntimeResidual(floor=1e-3)(zero, record)
    assert floor is not None and floor[0] == pytest.approx(math.log(1e-3))


# -------------------------------------------------------- drift responses
def test_after_drift_gate_with_rereference_response() -> None:
    inner = FixedGate([], needs_score=True)
    detector = ScriptedDetector(drift_at={3})
    sel = ThompsonSamplingSelector(
        budget=CUTOFF,
        gate=AfterDriftGate(inner),
        detector=detector,
        drift_response=RereferenceResponse(),
        random_state=0,
    )
    res = run_prequential(sel, three_algo_stream(), n_warmstart=10)
    steps = res.steps
    assert res.drift_steps == [2]  # drift on the 3rd update
    assert steps["response"].iloc[2] == "rereference"
    assert detector.n_resets == 1  # only re-referenced, nothing else
    # all algorithms right after the signal; the inner gate (no extras) elsewhere
    assert steps["n_runs"].iloc[3] == 3
    assert set(steps["n_runs"].drop(index=3)) == {1}
    assert len(inner.contexts) == len(steps) - 1
    assert (steps["instance_score"] == 0.25).all()  # inner gate asked for it


def test_reset_response_resets_bandit_and_rereferences_detector() -> None:
    detector = ScriptedDetector(drift_at={5})
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
    detector = ScalarRecorder()
    sel = PredictingSelector(
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
        {"retrain_every": 10},
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
        OnlinePairwiseSelector(detector=ScriptedDetector(set())).drift_responses_[0],
        RetrainResponse,
    )
    assert (
        OnlinePairwiseSelector(
            detector=ScriptedDetector(set()), retrain_every=5
        ).drift_responses_
        == ()
    )

    # drift response and policy fire on the same step: a single retraining
    detector = ScriptedDetector(drift_at={10})
    sel = OnlinePairwiseSelector(
        reference_period=20,
        detector=detector,
        drift_response=RetrainResponse(),
        retrain_every=30,
        random_state=0,
    )
    res = run_prequential(sel, three_algo_stream())
    assert res.drift_steps == [29]
    assert res.retrain_steps == [29, 59]
    assert detector.n_resets == 2
    assert res.steps["retrained"].sum() == 2


# ------------------------------------------------------- protocol & params
def test_protocol_follows_n_warmstart() -> None:
    stream = make_stream()
    static = StaticOnlineSelector(PairwiseClassifier(model_class=_Dummy))
    with pytest.raises(ValueError, match="warm start"):
        run_prequential(static, stream)
    warm = run_prequential(PredictingSelector(), stream, n_warmstart=5)
    assert warm.meta["protocol"] == "warm"
    assert run_prequential(PredictingSelector(), stream).meta["protocol"] == "cold"


class _Dummy:  # never fitted: the cold-start check fails before
    pass


def test_constructor_arguments_are_captured() -> None:
    sel = DiscountedTSSelector(gamma=0.9, budget=50.0, random_state=4)
    assert sel.init_params_["gamma"] == 0.9
    assert sel.init_params_["c"] == 1.0  # default of the outermost class
    assert sel.init_params_["budget"] == 50.0  # flattened **kwargs
    params = sel.describe_params()
    assert params["random_state"] == 4 and params["window_size"] == 200
