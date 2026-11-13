"""Tests for the regression-based online selector (Degroote et al., 2018)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from asf.online.base import Observation
from asf.online.drift.abstract_detector import AbstractScalarDriftDetector, DriftSignal
from asf.online.drift.statistics import LogRuntimeResidual
from asf.online.evaluation.prequential import OnlineResult, run_prequential
from asf.online.gate import AlwaysProbe
from asf.online.retraining.policy import PeriodicRetrain
from asf.online.retraining.response import ResetResponse, ReweightResponse
from asf.online.selectors import OnlineEPMSelector
from asf.online.selectors.online_epm import jackknife_after_bootstrap_sd
from asf.online.streams.stream import InstanceStream

from .conftest import CUTOFF


def small_forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=20, min_samples_split=2)


def make_selector(**kwargs: Any) -> OnlineEPMSelector:
    kwargs.setdefault("estimator", small_forest())
    kwargs.setdefault("budget", CUTOFF)
    kwargs.setdefault("random_state", 0)
    return OnlineEPMSelector(**kwargs)


def frames(
    perf: list[list[float]], algos: tuple[str, ...] = ("a", "b")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.Index([f"w{k}" for k in range(len(perf))])
    rng = np.random.default_rng(1)
    feats = pd.DataFrame(
        rng.normal(size=(len(perf), 2)), index=idx, columns=["f0", "f1"]
    )
    return feats, pd.DataFrame(perf, index=idx, columns=list(algos), dtype=float)


def obs(
    i: int, runtimes: dict[str, float], censored: tuple[str, ...] = ()
) -> Observation:
    feats = pd.DataFrame([[0.1 * i, -0.1 * i]], index=[f"o{i}"], columns=["f0", "f1"])
    return Observation(f"o{i}", feats, runtimes, frozenset(censored), CUTOFF)


def one_row(i: int = 0) -> pd.DataFrame:
    return pd.DataFrame([[0.0, 0.0]], index=[f"q{i}"], columns=["f0", "f1"])


def inject(selector: OnlineEPMSelector, table: dict[str, tuple[float, float]]) -> None:
    """Replace the models by fixed (mean, sd) predictions."""
    selector.models_ = {a: a for a in table}  # type: ignore[misc]

    def fake(model: Any, x: np.ndarray) -> tuple[float, float]:
        return table[model]

    selector._predict_model = fake  # type: ignore[method-assign]


# ------------------------------------------------------------- uncertainty
def test_jackknife_hand_computed() -> None:
    # 3 training rows, 4 trees with predictions 1, 2, 3, 6 (mean 3).
    trees = np.array([[1.0, 2.0, 3.0, 6.0]])
    oob = np.array(
        [
            [True, True, False, False],  # out-of-bag mean (1+2)/2 = 1.5
            [False, False, False, True],  # out-of-bag mean 6
            [False, False, False, False],  # never out-of-bag: ignored
        ]
    )
    # (n-1)/n * ((1.5-3)^2 + (6-3)^2) = 2/3 * 11.25 = 7.5
    plain = jackknife_after_bootstrap_sd(trees, oob, bias_correction=False)
    assert plain[0] == pytest.approx(math.sqrt(7.5))
    # bias: (e-1) * 3 / 16 * (4 + 1 + 0 + 9) = 2.625 (e-1)
    corrected = jackknife_after_bootstrap_sd(trees, oob, bias_correction=True)
    assert corrected[0] == pytest.approx(math.sqrt(7.5 - 2.625 * (math.e - 1)))


def test_jackknife_edge_cases() -> None:
    trees = np.array([[1.0, 2.0, 3.0, 6.0]])
    # One training row: undefined -> 0.
    assert jackknife_after_bootstrap_sd(trees, np.ones((1, 4), bool))[0] == 0.0
    # Every row out-of-bag everywhere: jackknife term 0, bias > 0 -> floored at 0.
    assert jackknife_after_bootstrap_sd(trees, np.ones((3, 4), bool))[0] == 0.0
    # Several queries at once.
    two = jackknife_after_bootstrap_sd(
        np.vstack([trees, trees + 5]), np.eye(4, dtype=bool)
    )
    assert two[0] == pytest.approx(two[1])


def test_tree_std_and_jackknife_from_a_fitted_forest() -> None:
    feats, perf = frames([[float(k), 1.0] for k in range(1, 31)])
    for kind in ("tree_std", "jackknife"):
        sel = make_selector(uncertainty=kind)
        sel.fit(feats, perf)
        model = sel.models_["a"]
        X = model.imputer.transform(feats.iloc[[0]].to_numpy())
        trees = np.array([t.predict(X)[0] for t in model.estimator.estimators_])
        mean, sd = sel._predict_model(model, feats.iloc[[0]].to_numpy())
        assert mean == pytest.approx(trees.mean())
        if kind == "tree_std":
            assert sd == pytest.approx(trees.std(ddof=1))
        else:
            # oob mask matches the bootstrap draws of each tree
            for b, sample in enumerate(model.estimator.estimators_samples_):
                assert not model.oob[sample, b].any()
                assert model.oob[:, b].sum() == 30 - len(np.unique(sample))
            assert sd >= 0.0


# --------------------------------------------------------------- selection
@pytest.fixture
def fitted() -> OnlineEPMSelector:
    sel = make_selector()
    sel.fit(*frames([[5.0, 50.0], [6.0, 40.0]]))
    return sel


def test_greedy_minimize_and_maximize() -> None:
    feats, perf = frames([[5.0, 50.0], [6.0, 40.0]])
    sel = make_selector(budget=None)
    sel.fit(feats, perf)
    inject(sel, {"a": (10.0, 0.0), "b": (12.0, 3.0)})
    record = sel._select(one_row())
    assert record.chosen == "a"
    assert record.predicted_runtime == 10.0
    assert record.score == -10.0
    assert record.scores == {"a": -10.0, "b": -12.0}
    assert record.candidates == ("b",)

    sel = make_selector(budget=None, maximize=True)
    sel.fit(feats, perf)
    inject(sel, {"a": (10.0, 0.0), "b": (12.0, 3.0)})
    assert sel._select(one_row()).chosen == "b"


def test_ucb_gamma_flips_the_choice(fitted: OnlineEPMSelector) -> None:
    # a: 10 - 0 = 10; b: 12 - gamma*3 -> 9 for gamma=1, 10.5 for gamma=0.5
    for gamma, expected in ((1.0, "b"), (0.5, "a")):
        sel = make_selector(strategy="ucb", gamma=gamma)
        sel.fit(*frames([[5.0, 50.0], [6.0, 40.0]]))
        inject(sel, {"a": (10.0, 0.0), "b": (12.0, 3.0)})
        record = sel._select(one_row())
        assert record.chosen == expected
        assert record.scores == pytest.approx({"a": -10.0, "b": -12.0 + gamma * 3.0})
        assert record.uncertainty == (0.0 if expected == "a" else 3.0)
    # maximize: a: 10 + 0, b: 12 + 3 -> b
    sel = make_selector(strategy="ucb", maximize=True, budget=None)
    sel.fit(*frames([[5.0, 50.0], [6.0, 40.0]]))
    inject(sel, {"a": (10.0, 0.0), "b": (12.0, 3.0)})
    assert sel._select(one_row()).scores == {"a": 10.0, "b": 15.0}


def test_epsilon_greedy_probabilities() -> None:
    def frequency_of_a(epsilon: float) -> float:
        sel = make_selector(strategy="epsilon_greedy", epsilon=epsilon)
        sel.fit(*frames([[5.0, 50.0], [6.0, 40.0]]))
        inject(sel, {"a": (1.0, 0.0), "b": (2.0, 0.0)})
        return float(
            np.mean([sel._select(one_row()).chosen == "a" for _ in range(4000)])
        )

    assert frequency_of_a(0.0) == 1.0
    assert frequency_of_a(1.0) == pytest.approx(0.5, abs=0.03)
    # best with probability 1 - eps + eps / |A| = 0.8 + 0.1
    assert frequency_of_a(0.2) == pytest.approx(0.9, abs=0.03)


def test_ties_are_random_but_reproducible() -> None:
    def picks(seed: int) -> list[str]:
        sel = make_selector(random_state=seed)
        sel.fit(*frames([[5.0, 50.0], [6.0, 40.0]]))
        inject(sel, {"a": (3.0, 0.0), "b": (3.0, 0.0)})
        return [sel._select(one_row()).chosen for _ in range(50)]

    assert set(picks(0)) == {"a", "b"}
    assert picks(0) == picks(0)


def test_candidates_put_unmodelled_first() -> None:
    feats, perf = frames([[5.0, 50.0, np.nan, 7.0]] * 3, algos=("a", "b", "c", "d"))
    sel = make_selector()
    sel.fit(feats, perf)
    assert set(sel.models_) == {"a", "b", "d"}
    record = sel._select(one_row())
    assert record.chosen == "c"  # the only unmodelled algorithm
    assert record.predicted_runtime is None and record.uncertainty is None
    assert record.score is None
    assert record.candidates == ("a", "d", "b")  # by descending score


# ----------------------------------------------------------------- targets
def test_par_targets(fitted: OnlineEPMSelector) -> None:
    perf = pd.DataFrame(
        [[5.0, 100.0], [150.0, np.nan], [100.0, 7.0]], columns=["a", "b"]
    )
    cens = pd.DataFrame(
        [[False, False], [False, False], [True, False]], columns=["a", "b"]
    )
    out = fitted._training_targets(perf, cens)
    expected = np.array([[5.0, 100.0], [1000.0, np.nan], [1000.0, 7.0]])
    np.testing.assert_array_equal(out, expected)  # == cutoff counts as solved

    fitted.target_transform = "log10"
    np.testing.assert_allclose(fitted._training_targets(perf, cens), np.log10(expected))

    fitted.cutoff_ = math.inf  # no cutoff: values stay raw
    fitted.target_transform = "none"
    np.testing.assert_array_equal(fitted._training_targets(perf, cens), perf.to_numpy())


def test_log10_predictions_are_back_transformed() -> None:
    feats, perf = frames([[10.0, 1000.0]] * 4)
    sel = make_selector(target_transform="log10", uncertainty=None)
    sel.fit(feats, perf)
    record = sel._select(one_row())
    assert record.chosen == "a"
    assert record.predicted_runtime == pytest.approx(10.0)
    assert record.scores == pytest.approx({"a": -1.0, "b": -3.0})


def test_censored_observation_is_learned_as_par10(fitted: OnlineEPMSelector) -> None:
    fitted.retrain_every = 1
    fitted.observe(obs(0, {"a": CUTOFF}, censored=("a",)))
    _, Y, _ = fitted._training_data()
    assert Y[-1, 0] == 10 * CUTOFF


# ---------------------------------------------------------- learning loop
def test_cold_start_tries_every_algorithm_once() -> None:
    feats, perf = frames([], algos=("a", "b", "c"))
    sel = make_selector()
    sel.fit(feats.iloc[:0], perf.iloc[:0])
    assert sel.models_ == {}
    chosen = []
    for i in range(3):
        sel.propose(one_row(i))
        record = sel.last_prediction_
        assert record is not None and record.predicted_runtime is None
        chosen.append(record.chosen)
        sel.observe(obs(i, {record.chosen: 5.0 + i}))
        assert record.chosen in sel.models_  # fitted on its first observation
    assert sorted(chosen) == ["a", "b", "c"]
    sel.propose(one_row(3))
    assert sel.last_prediction_.predicted_runtime is not None  # type: ignore[union-attr]


def test_update_schedule_counts_data_points(
    fitted: OnlineEPMSelector, monkeypatch: pytest.MonkeyPatch
) -> None:
    fits: list[str] = []
    original = fitted._fit_model

    def counting(index: int, *args: Any) -> Any:
        fits.append(fitted.algorithms[index])
        return original(index, *args)

    monkeypatch.setattr(fitted, "_fit_model", counting)
    for i in range(9):
        fitted.observe(obs(i, {"a": 5.0}))
    assert fits == []
    fitted.observe(obs(9, {"a": 5.0}))  # 10th data point: only "a" changed
    assert fits == ["a"]
    for i in range(10, 15):  # 5 instances with an extra run = 10 data points
        fitted.observe(obs(i, {"a": 5.0, "b": 50.0}))
    assert fits == ["a", "a", "b"]
    fitted.retrain()  # nothing changed since the last update
    assert fits == ["a", "a", "b"]


def test_model_follows_new_observations(fitted: OnlineEPMSelector) -> None:
    fitted.retrain_every = 1
    before = fitted._select(one_row()).predicted_runtime
    for i in range(20):
        fitted.observe(obs(i, {"a": 80.0}))
    after = fitted._predict_model(fitted.models_["a"], one_row().to_numpy())[0]
    assert before == pytest.approx(5.5, abs=0.6)
    assert after > 50.0


def test_window_drops_models_without_data() -> None:
    sel = make_selector(window_size=3, retrain_every=1)
    sel.fit(*frames([[5.0, 50.0]] * 3))
    for i in range(3):
        sel.observe(obs(i, {"a": 5.0}))
    assert set(sel.models_) == {"a"}
    assert sel.models_["a"].n_rows == 3


def test_reset_forgets_everything(fitted: OnlineEPMSelector) -> None:
    fitted.reset_model()
    assert fitted.models_ == {}
    fitted.observe(obs(0, {"a": 7.0}))
    assert set(fitted.models_) == {"a"}
    assert fitted.models_["a"].n_rows == 1  # warm-start rows are forgotten
    X, Y, _ = fitted._training_data()
    assert len(X) == 1


# ------------------------------------------------- framework integration
def test_capability_hooks(stream: InstanceStream) -> None:
    sel = make_selector()
    assert sel.supports("_retrain") and sel.supports("_reset_model")
    assert not sel.supports("_discount_history")
    with pytest.raises(ValueError, match="_discount_history"):
        make_selector(drift_response=ReweightResponse(0.5)).fit(*frames([[5.0, 50.0]]))
    make_selector(drift_response=ResetResponse())  # accepted

    sel = make_selector(retrain_policy=PeriodicRetrain(every=25))
    result = run_prequential(sel, stream, n_warmstart=20)
    assert len(result.retrain_steps) == 4
    assert set(np.diff(result.retrain_steps)) == {25}
    assert result.steps["retrained"].sum() == 4


def test_prequential_warm_and_cold(stream: InstanceStream, tmp_path: Any) -> None:
    for n_warm in (20, 0):
        sel = make_selector(strategy="ucb")
        result = run_prequential(sel, stream, n_warmstart=n_warm)
        steps = result.steps
        assert (steps["n_runs"] == 1).all()  # default gate: one run
        modelled = steps["pred_runtime"].notna()
        assert modelled.iloc[2:].all()  # at most the first 2 steps are unmodelled
        assert steps.loc[modelled, "pred_uncertainty"].notna().all()
        assert steps["pred_score"].notna().sum() == modelled.sum()
    result.save(tmp_path / "run")
    loaded = OnlineResult.load(tmp_path / "run")
    pd.testing.assert_series_equal(loaded.steps["pred_runtime"], steps["pred_runtime"])


def test_all_data_upper_bound_runs_everything(stream: InstanceStream) -> None:
    sel = make_selector(gate=AlwaysProbe())
    result = run_prequential(sel, stream, n_warmstart=20, result_mode="primary")
    assert (result.steps["n_runs"] == 2).all()
    assert sel.buffer is not None and sel.buffer.performance().notna().all().all()


def test_learns_on_the_toy_stream(stream: InstanceStream) -> None:
    sel = make_selector(retrain_every=1)
    result = run_prequential(sel, stream, n_warmstart=40)
    before_drift = result.steps.iloc[: 80 - 40]
    assert (before_drift["user_cost"] == before_drift["vbs_cost"]).mean() > 0.8


def test_determinism(stream: InstanceStream) -> None:
    def run(seed: int) -> pd.DataFrame:
        sel = make_selector(strategy="epsilon_greedy", epsilon=0.3, random_state=seed)
        cols = ["primary", "pred_runtime", "pred_uncertainty", "pred_score"]
        return run_prequential(sel, stream, n_warmstart=10).steps[cols]

    pd.testing.assert_frame_equal(run(3), run(3))
    assert not run(3)["primary"].equals(run(4)["primary"])


class _Recorder(AbstractScalarDriftDetector):
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float, censored: bool = False) -> DriftSignal:
        self.values.append(value)
        return DriftSignal(drift=False)

    def reset(self) -> None:
        self.values = []


def test_feeds_the_runtime_residual(stream: InstanceStream) -> None:
    detector = _Recorder()
    sel = make_selector(detector=detector, drift_statistic=LogRuntimeResidual())
    result = run_prequential(sel, stream, n_warmstart=20)
    assert len(detector.values) == len(result.steps)
    assert result.steps["drift_statistic"].notna().all()


def test_extra_trees_and_point_estimators(stream: InstanceStream) -> None:
    sel = make_selector(
        estimator=ExtraTreesRegressor(n_estimators=10), uncertainty="tree_std"
    )
    run_prequential(sel, stream, n_warmstart=20)
    sel = make_selector(estimator=Ridge(), uncertainty=None)
    steps = run_prequential(sel, stream, n_warmstart=20).steps
    assert steps["pred_uncertainty"].isna().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "thompson"},
        {"epsilon": 1.5},
        {"gamma": -1.0},
        {"uncertainty": "bootstrap"},
        {"strategy": "ucb", "uncertainty": None},
        {"target_transform": "log10", "maximize": True},
        {"par_factor": 0.0},
        {"retrain_every": 0},
        {"estimator": Ridge()},  # uncertainty needs a forest
        {"estimator": ExtraTreesRegressor()},  # jackknife needs bootstrap
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_selector(**kwargs)
