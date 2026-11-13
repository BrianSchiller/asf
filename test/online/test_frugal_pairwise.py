"""Tests for the frugal pairwise selector and the uncertainty gate (A2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from asf.online.base import Observation
from asf.online.evaluation.prequential import run_prequential
from asf.online.gate import AlwaysProbe, GateContext, UncertaintyGate
from asf.online.prediction import PredictionRecord
from asf.online.retraining.policy import PeriodicRetrain
from asf.online.retraining.response import ResetResponse, RetrainResponse
from asf.online.selectors import FrugalPairwiseSelector
from asf.online.selectors.frugal_pairwise import least_confidence

from .conftest import make_stream

ALGOS = ["a", "b", "c"]


def _warm(perf: list[list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.Index([f"w{k}" for k in range(len(perf))])
    x = pd.DataFrame({"f0": np.arange(len(perf), dtype=float)}, index=idx)
    return x, pd.DataFrame(perf, index=idx, columns=ALGOS)


def _selector(**kwargs: object) -> FrugalPairwiseSelector:
    kwargs.setdefault("model_class", DecisionTreeClassifier)
    kwargs.setdefault("random_state", 0)
    return FrugalPairwiseSelector(**kwargs)  # type: ignore[arg-type]


def _fitted(**kwargs: object) -> FrugalPairwiseSelector:
    sel = _selector(**kwargs)
    sel.fit(*_warm([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]))
    return sel


def _one(value: float = 0.0, name: str = "x") -> pd.DataFrame:
    return pd.DataFrame({"f0": [value]}, index=pd.Index([name]))


def _context(uncertainty: float | None, candidates: tuple[str, ...]) -> GateContext:
    record = PredictionRecord(chosen="a", uncertainty=uncertainty)
    return GateContext(
        prediction=record,
        candidates=candidates,
        algorithms=("a", "b", "c"),
        features=_one(),
        step=0,
    )


# Hand case: P(a>b)=0.9, P(a>c)=0.4, P(b>c)=0.3 -> votes a=1, b=0, c=2.
# Chosen c; P(c>a)=0.6 (LC 0.4), P(c>b)=0.7 (LC 0.3) -> nominees (a, b).
HAND_MODELS = {(0, 1): 0.9, (0, 2): 0.4, (1, 2): 0.3}


# ---------------------------------------------------------------- helpers
def test_least_confidence_hand_values() -> None:
    p = np.array([0.0, 0.5, 0.9, 0.25, 1.0])
    np.testing.assert_allclose(least_confidence(p), [0.0, 0.5, 0.1, 0.25, 0.0])


def test_least_confidence_ranks_like_margin_and_entropy() -> None:
    p = np.array([0.05, 0.3, 0.45, 0.8, 0.99])
    margin = np.abs(2 * p - 1)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    order = np.argsort(-least_confidence(p))
    assert list(order) == list(np.argsort(margin))
    assert list(order) == list(np.argsort(-entropy))


def test_labels_hand_cases() -> None:
    sel = _fitted()
    perf_i = np.array([1.0, 5.0, 3.0, 10.0, 4.0])
    perf_j = np.array([2.0, 4.0, 3.0, 10.0, 10.0])
    t_i = np.array([False, False, False, True, False])
    t_j = np.array([False, False, False, False, True])
    # win, loss, tie -> i, i timed out, j timed out
    assert list(sel._labels(perf_i, perf_j, t_i, t_j)) == [1, 0, 1, 0, 1]


def test_labels_maximize() -> None:
    sel = _selector(maximize=True)
    sel.fit(*_warm([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]))
    no = np.zeros(2, dtype=bool)
    assert list(sel._labels(np.array([1.0, 3.0]), np.array([2.0, 2.0]), no, no)) == [
        0,
        1,
    ]


# ---------------------------------------------------------------- gate
def test_uncertainty_gate_threshold() -> None:
    gate = UncertaintyGate(threshold=0.3, n=1)
    assert list(gate.extras(_context(0.31, ("b", "c")))) == ["b"]
    assert list(gate.extras(_context(0.3, ("b", "c")))) == ["b"]
    assert list(gate.extras(_context(0.29, ("b", "c")))) == []
    assert list(gate.extras(_context(None, ("b", "c")))) == []
    assert list(UncertaintyGate(0.0, n=None).extras(_context(0.1, ("b", "c")))) == [
        "b",
        "c",
    ]
    assert list(UncertaintyGate(0.0, n=0).extras(_context(0.5, ("b",)))) == []


def test_uncertainty_gate_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        UncertaintyGate(0.2, n=-1)


# ---------------------------------------------------------------- selection
def test_select_hand_case() -> None:
    sel = _fitted()
    sel._models = dict(HAND_MODELS)
    record = sel._select(_one())
    assert record.chosen == "c"
    assert record.scores == {"a": 1.0, "b": 0.0, "c": 2.0}
    assert record.score == 2.0
    assert record.candidates == ("a", "b")
    assert record.uncertainty == pytest.approx(0.4)
    assert record.predicted_runtime is None


def test_default_gate_runs_most_uncertain_nominee() -> None:
    sel = _fitted()
    sel._models = dict(HAND_MODELS)
    assert sel.propose(_one()) == [("c", 0.0), ("a", 0.0)]
    sel.gate_ = UncertaintyGate(threshold=0.45)
    assert sel.propose(_one()) == [("c", 0.0)]


def test_random_candidate_order_is_seeded() -> None:
    orders = set()
    for seed in range(20):
        sel = _fitted(candidate_order="random", random_state=seed)
        sel._models = dict(HAND_MODELS)
        record = sel._select(_one())
        again = _fitted(candidate_order="random", random_state=seed)
        again._models = dict(HAND_MODELS)
        assert again._select(_one()) == record
        assert record.uncertainty == pytest.approx(0.4)  # order does not change it
        orders.add(record.candidates)
    assert orders == {("a", "b"), ("b", "a")}


def test_cold_start_is_maximally_uncertain_and_ties_are_seeded() -> None:
    chosen = set()
    for seed in range(20):
        sel = _selector(random_state=seed)
        sel.fit(*_warm([])[0:1], pd.DataFrame(columns=ALGOS))
        record = sel._select(_one())
        assert record.uncertainty == 0.5
        assert set(record.candidates) == set(ALGOS) - {record.chosen}
        assert set(record.scores.values()) == {0.0}  # type: ignore[union-attr]
        twin = _selector(random_state=seed)
        twin.fit(*_warm([])[0:1], pd.DataFrame(columns=ALGOS))
        assert twin._select(_one()).chosen == record.chosen
        chosen.add(record.chosen)
    assert chosen == set(ALGOS)


def test_single_label_pair_uses_laplace_estimate() -> None:
    sel = _fitted()  # a beats c and b beats c twice; a vs b split
    assert sel._models[(0, 2)] == pytest.approx(3 / 4)
    assert sel._models[(1, 2)] == pytest.approx(3 / 4)
    assert not isinstance(sel._models[(0, 1)], float)  # mixed labels -> trained


def test_model_without_predict_proba_is_rejected() -> None:
    with pytest.raises(TypeError, match="predict_proba"):
        _fitted(model_class=LinearSVC)


@pytest.mark.parametrize(
    "kwargs",
    [{"candidate_order": "best"}, {"update_every": 0}],
)
def test_invalid_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _selector(**kwargs)


def test_needs_two_algorithms() -> None:
    x, perf = _warm([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="two algorithms"):
        _selector().fit(x, perf[["a"]])


# ---------------------------------------------------------------- training
def test_double_timeouts_are_not_used_for_training() -> None:
    sel = _selector(cutoff=10.0)
    # rows: a wins, double timeout, b wins, a times out
    sel.fit(*_warm([[1, 2, 0], [10, 12, 0], [3, 1, 0], [12, 5, 0]]))
    assert sel.pair_n_train_[0, 1] == 3
    assert sel.pair_n_train_[0, 2] == 4


def test_observe_refits_only_pairs_that_ran() -> None:
    sel = _fitted()
    before = dict(sel._models)
    n_before = sel.pair_n_train_.copy()
    sel.propose(_one(0.5, "s0"))
    sel.observe(Observation("s0", _one(0.5, "s0"), {"a": 1.0, "b": 4.0}, cutoff=10))
    expected = n_before.copy()
    expected[0, 1] = expected[1, 0] = n_before[0, 1] + 1
    np.testing.assert_array_equal(sel.pair_n_train_, expected)
    assert sel._models[(0, 2)] is before[(0, 2)]
    assert sel._models[(1, 2)] is before[(1, 2)]

    sel.propose(_one(0.7, "s1"))
    sel.observe(Observation("s1", _one(0.7, "s1"), {"c": 1.0}, cutoff=10))
    np.testing.assert_array_equal(sel.pair_n_train_, expected)  # primary only


def test_update_every_batches_refits() -> None:
    sel = _fitted(update_every=3)
    n0 = sel.pair_n_train_[0, 1]
    for k in range(3):
        name = f"s{k}"
        sel.propose(_one(k, name))
        sel.observe(Observation(name, _one(k, name), {"a": 1.0, "b": 2.0}))
        assert sel.pair_n_train_[0, 1] == (n0 if k < 2 else n0 + 3)


# ---------------------------------------------------------------- timeouts
def test_timeout_filter_drops_double_timeout_nominees() -> None:
    sel = _fitted(timeout_filter=True, cutoff=10.0)
    sel._models = dict(HAND_MODELS)
    sel._timeout_models = {0: 0.9, 1: 0.1, 2: 0.9}  # a and c predicted to time out
    record = sel._select(_one())
    assert record.chosen == "c"
    assert record.candidates == ("b",)
    assert record.uncertainty == pytest.approx(0.3)

    sel._timeout_models = {0: 0.9, 1: 0.9, 2: 0.9}
    record = sel._select(_one())
    assert record.candidates == ()
    assert record.uncertainty == 0.0

    sel._timeout_models = {0: 0.9, 1: 0.9, 2: 0.1}  # chosen expected to finish
    assert sel._select(_one()).candidates == ("a", "b")


def test_timeout_voting_excludes_predicted_timeouts() -> None:
    sel = _fitted(timeout_voting=True, cutoff=10.0)
    sel._models = dict(HAND_MODELS)
    sel._timeout_models = {2: 0.9}
    record = sel._select(_one())
    assert record.chosen == "a"  # c excluded; a beats b
    assert record.scores is not None and record.scores["c"] == -np.inf

    sel._timeout_models = {0: 0.9, 1: 0.9, 2: 0.9}  # all: no exclusion
    assert sel._select(_one()).chosen == "c"


def test_timeout_predictors_are_trained_from_warm_start() -> None:
    sel = _selector(timeout_filter=True, cutoff=10.0)
    sel.fit(*_warm([[1, 12, 3], [2, 15, 4]]))
    assert sel._timeout_models[0] == pytest.approx(1 / 4)  # never timed out
    assert sel._timeout_models[1] == pytest.approx(3 / 4)  # always timed out


def test_timeout_predictors_need_cutoff_on_warm_start() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        _fitted(timeout_filter=True)


def test_timeout_predictors_need_minimisation() -> None:
    with pytest.raises(ValueError, match="minimised"):
        _selector(timeout_voting=True, maximize=True)


# ---------------------------------------------------------------- responses
def test_reset_forgets_seen_data() -> None:
    sel = _fitted(drift_response=ResetResponse())
    sel.reset_model()
    assert sel._models == {}
    assert sel._select(_one()).uncertainty == 0.5
    sel.propose(_one(0.5, "s0"))
    sel.observe(Observation("s0", _one(0.5, "s0"), {"a": 1.0, "b": 4.0}))
    assert sel.pair_n_train_[0, 1] == 1


def test_retrain_refits_everything() -> None:
    sel = _fitted(drift_response=RetrainResponse())
    sel._models = {}
    sel.retrain()
    assert set(sel._models) == {(0, 1), (0, 2), (1, 2)}
    assert sel.retrain_steps == [0]


# ---------------------------------------------------------------- end to end
def _run(seed: int, n_warmstart: int, **kwargs: object) -> pd.DataFrame:
    sel = _selector(random_state=seed, **kwargs)
    protocol = "warm" if n_warmstart else "cold"
    result = run_prequential(
        sel, make_stream(), n_warmstart=n_warmstart, protocol=protocol
    )
    return result.steps.drop(columns=["selector_seconds", "detector_seconds"])


def test_prequential_is_deterministic_and_frugal() -> None:
    steps = _run(0, 0)
    pd.testing.assert_frame_equal(steps, _run(0, 0))
    assert 0.0 < (steps["n_runs"] > 1).mean() < 1.0
    assert steps["pred_uncertainty"].notna().all()
    assert steps["pred_uncertainty"].between(0.0, 0.5).all()
    assert (steps["user_cost"] == steps["cost_primary"]).all()  # RESULT_MODE


def test_always_probe_gives_two_runs_per_instance() -> None:
    steps = _run(0, 20, gate=AlwaysProbe(1))
    assert (steps["n_runs"] == 2).all()


def test_periodic_retrain_policy_is_supported() -> None:
    sel = _selector(retrain_policy=PeriodicRetrain(25))
    result = run_prequential(sel, make_stream(), n_warmstart=20)
    assert len(result.retrain_steps) == 4
