"""Tests for censored-runtime imputation (Tornede et al., 2022).

Covers the imputation rules, their use in the bandits (rewards) and in the
online EPM selector (training targets).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from asf.online.base import Observation
from asf.online.evaluation import run_prequential
from asf.online.imputation import (
    CensoredImputer,
    CutoffImputer,
    DropCensored,
    ExpectedImputer,
    SampledImputer,
)
from asf.online.selectors.bandits import (
    AbstractContextualBandit,
    GaussianPosterior,
    LinTSSelector,
    RewardTransform,
    SlidingWindow,
    ThompsonSamplingSelector,
)
from asf.online.selectors.online_epm import OnlineEPMSelector
from asf.online.streams.stream import InstanceStream
from asf.preprocessing.schmee_hahn_imputer import schmee_hahn_impute


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ================================================================== rules
def test_cutoff_returns_bound_and_drop_returns_nan() -> None:
    out = CutoffImputer()([1.0, 2.5], [10.0, -3.0], [1.0, 0.0], _rng())
    np.testing.assert_array_equal(out, [1.0, 2.5])
    out = DropCensored()([1.0, 2.0, 3.0], 0.0, 1.0, _rng())
    assert out.shape == (3,) and np.isnan(out).all()


@pytest.mark.parametrize(
    "bound, mean, sd, expected",
    [
        # phi(0) / (1 - Phi(0)) = 0.398942 / 0.5
        (0.0, 0.0, 1.0, 0.7978845608),
        # phi(1) / (1 - Phi(1)) = 0.241971 / 0.158655
        (1.0, 0.0, 1.0, 1.5251352762),
        # 2 + phi(2) / Phi(2) = 2 + 0.053991 / 0.977250
        (0.0, 2.0, 1.0, 2.0552478627),
        # scaling: bound 10, mean 6, sd 4 -> 6 + 4 * 1.525135
        (10.0, 6.0, 4.0, 6.0 + 4.0 * 1.5251352762),
    ],
)
def test_expected_hand_computed(
    bound: float, mean: float, sd: float, expected: float
) -> None:
    out = ExpectedImputer()(bound, mean, sd, _rng())
    assert out[0] == pytest.approx(expected, rel=1e-9)


def test_expected_edge_cases() -> None:
    imp = ExpectedImputer()
    # sd == 0: no uncertainty -> max(mean, bound)
    np.testing.assert_array_equal(imp([0.0, 5.0], [3.0, 3.0], 0.0, _rng()), [3.0, 5.0])
    # sd == inf: no information -> bound
    assert imp(2.0, 0.0, np.inf, _rng())[0] == 2.0
    # far tail stays finite and close to bound + sd^2 / (bound - mean)
    far = imp(40.0, 0.0, 1.0, _rng())[0]
    assert math.isfinite(far) and 40.0 < far < 40.0 + 1.0 / 40.0 + 1e-9
    with pytest.raises(ValueError):
        ExpectedImputer()(0.0, 0.0, -1.0, _rng())  # negative sd


def test_expected_never_below_bound_and_deterministic() -> None:
    rng = _rng(1)
    bound, mean, sd = rng.normal(size=200), rng.normal(size=200) * 5, rng.random(200)
    a = ExpectedImputer()(bound, mean, sd, _rng(0))
    b = ExpectedImputer()(bound, mean, sd, _rng(99))
    assert np.all(a >= bound)
    np.testing.assert_array_equal(a, b)


def test_sampled_above_bound_and_seeded() -> None:
    bound = np.zeros(500)
    a = SampledImputer()(bound, 0.0, 1.0, _rng(3))
    b = SampledImputer()(bound, 0.0, 1.0, _rng(3))
    np.testing.assert_array_equal(a, b)
    assert np.all(a > 0.0)
    assert not np.array_equal(a, SampledImputer()(bound, 0.0, 1.0, _rng(4)))


def test_sampled_falls_back_to_bound() -> None:
    # Mean 100 sd below the bound: no draw is accepted in 20 tries.
    out = SampledImputer()([0.0, 1.0], -100.0, 1.0, _rng())
    np.testing.assert_array_equal(out, [0.0, 1.0])
    # One try with P(accept) = 1/2: some entries fall back, the rest are above.
    out = SampledImputer(max_tries=1)(np.zeros(1000), 0.0, 1.0, _rng())
    fallback = out == 0.0
    assert 400 < fallback.sum() < 600 and np.all(out[~fallback] > 0.0)


def test_sampled_mean_matches_truncated_mean() -> None:
    # With enough tries the accepted draws follow N(0, 1) truncated at 0.5.
    out = SampledImputer(max_tries=200)(np.full(20000, 0.5), 0.0, 1.0, _rng(7))
    truncated_mean = norm.pdf(0.5) / norm.sf(0.5)
    assert out.mean() == pytest.approx(truncated_mean, abs=0.02)


def test_sampled_edge_cases() -> None:
    imp = SampledImputer()
    assert imp(1.0, 0.0, np.inf, _rng())[0] == 1.0  # no information
    np.testing.assert_array_equal(imp([0.0, 5.0], 3.0, 0.0, _rng()), [3.0, 5.0])
    with pytest.raises(ValueError):
        SampledImputer(max_tries=0)


def test_rule_flags() -> None:
    assert not CutoffImputer.USES_MODEL and not DropCensored.USES_MODEL
    assert ExpectedImputer.USES_MODEL and ExpectedImputer.PREDICTIVE
    assert SampledImputer.USES_MODEL and not SampledImputer.PREDICTIVE


# ================================================================ bandits
# Context-free Gaussian Thompson Sampling (prior precision 1, noise variance 1)
# keeps one running sum per arm, so posteriors can be checked by hand:
# precision = 1 + n, mean = sum(rewards) / precision.

BANDIT_CUTOFF = 100.0


def _bandit_data(perf_a: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(perf_a)
    idx = pd.Index([f"w{k}" for k in range(n)])
    features = pd.DataFrame({"f0": np.arange(n, dtype=float)}, index=idx)
    performance = pd.DataFrame({"a": perf_a, "b": [50.0] * n}, index=idx)
    return features, performance


def _bandit(
    imputation: CensoredImputer | None, perf_a: list[float], **kw: Any
) -> ThompsonSamplingSelector:
    sel = ThompsonSamplingSelector(
        budget=BANDIT_CUTOFF, imputation=imputation, random_state=0, **kw
    )
    sel.fit(*_bandit_data(perf_a))
    return sel


def _arm(sel: AbstractContextualBandit, arm: str) -> GaussianPosterior:
    (models,) = sel.posteriors_.values()  # one cell, one trace
    return models[arm]


def _censored_obs(runtime: float = BANDIT_CUTOFF) -> Observation:
    features = pd.DataFrame({"f0": [0.5]}, index=pd.Index(["s0"]))
    return Observation("s0", features, {"a": runtime}, frozenset({"a"}), BANDIT_CUTOFF)


# Warm start for arm a: runtimes 9 and 99 -> rewards -1 and -2.
WARM_A = [9.0, 99.0]


def test_bandit_rules_update_the_posterior_by_hand() -> None:
    # default keeps the PAR10 reward; cutoff imputes the bound; drop skips it
    sel = _bandit(None, WARM_A)
    sel.observe(_censored_obs())
    post = _arm(sel, "a")
    assert post.precision == 4.0
    assert post.weighted_sum == pytest.approx(-3.0 - math.log10(10 * 100 + 1))

    sel = _bandit(CutoffImputer(), WARM_A)
    sel.observe(_censored_obs())
    post = _arm(sel, "a")
    assert post.precision == 4.0
    assert post.weighted_sum == pytest.approx(-3.0 - math.log10(100 + 1))

    sel = _bandit(DropCensored(), WARM_A)
    sel.observe(_censored_obs())
    post = _arm(sel, "a")
    assert post.precision == 3.0 and post.weighted_sum == pytest.approx(-3.0)
    assert _arm(sel, "b").precision == 3.0  # not executed: untouched as well


def test_bandit_expected_uses_posterior_before_update() -> None:
    sel = _bandit(ExpectedImputer(), WARM_A)
    # Before the update: mean reward -3/3 = -1, width sqrt(1/3). The rule
    # sees the negated reward: belief N(1, 1/3 + noise 1), bound log10(101).
    bound, mean, sd = math.log10(101.0), 1.0, math.sqrt(1.0 / 3.0 + 1.0)
    a = (bound - mean) / sd
    expected = mean + sd * norm.pdf(a) / norm.sf(a)
    sel.observe(_censored_obs())
    post = _arm(sel, "a")
    assert post.weighted_sum == pytest.approx(-3.0 - expected)
    assert expected > bound  # imputed reward is below the cutoff's reward


def test_bandit_sampled_below_bound_and_seeded() -> None:
    sums = []
    for _ in range(2):
        sel = _bandit(SampledImputer(), WARM_A)
        sel.observe(_censored_obs())
        sums.append(_arm(sel, "a").weighted_sum)
    assert sums[0] == sums[1]
    imputed = -(sums[0] + 3.0)  # negated reward that was stored
    assert imputed > math.log10(101.0)


def test_bandit_sampled_falls_back_to_bound() -> None:
    # 400 instant warm-start runs: mean reward 0 (scaled by 400/401), width
    # 1/sqrt(401) = 0.05, so log10(101) = 2 is 40 sd away -> no accepted draw.
    sel = _bandit(SampledImputer(), [0.0] * 400)
    before = _arm(sel, "a").weighted_sum
    sel.observe(_censored_obs())
    assert _arm(sel, "a").weighted_sum - before == pytest.approx(-math.log10(101.0))


def test_bandit_bound_is_the_run_budget_and_par_reward() -> None:
    sel = _bandit(CutoffImputer(), WARM_A)
    sel.observe(_censored_obs(runtime=30.0))  # stopped at a budget of 30 s
    assert _arm(sel, "a").weighted_sum == pytest.approx(-3.0 - math.log10(31.0))

    sel = _bandit(CutoffImputer(), WARM_A, reward=RewardTransform("par"))
    before = _arm(sel, "a").weighted_sum
    sel.observe(_censored_obs(runtime=30.0))
    assert _arm(sel, "a").weighted_sum - before == pytest.approx(-30.0 / 1000.0)


def test_bandit_warm_start_timeouts_are_imputed() -> None:
    # 500 s >= cutoff in the warm start: counted as censored at the cutoff.
    cutoff_sel = _bandit(CutoffImputer(), [9.0, 500.0])
    assert _arm(cutoff_sel, "a").weighted_sum == pytest.approx(-1.0 - math.log10(101))
    drop_sel = _bandit(DropCensored(), [9.0, 500.0])
    assert _arm(drop_sel, "a").precision == 2.0
    default_sel = _bandit(None, [9.0, 500.0])
    assert _arm(default_sel, "a").weighted_sum == pytest.approx(-1.0 - math.log10(1001))


def test_bandit_rejects_unsupported_reward() -> None:
    for kind in ("solved", "unit", "raw"):
        with pytest.raises(ValueError, match="'log' or 'par'"):
            _bandit(CutoffImputer(), WARM_A, reward=RewardTransform(kind))


def test_bandit_rejects_maximize_and_wrong_type() -> None:
    with pytest.raises(ValueError, match="minimization"):
        _bandit(CutoffImputer(), WARM_A, maximize=True)
    with pytest.raises(TypeError):
        ThompsonSamplingSelector(imputation="cutoff")  # type: ignore[arg-type]


def _timeout_stream(n: int = 150, seed: int = 0) -> InstanceStream:
    """Best arm depends on sign(f0); the wrong arm times out (cutoff 100)."""
    rng = np.random.default_rng(seed)
    f0 = rng.choice([-2.0, 2.0], n) + rng.normal(0, 0.3, n)
    x = np.column_stack([f0, rng.normal(size=n)])
    perf = np.where((f0 < 0)[:, None], [[5.0, 900.0]], [[900.0, 5.0]])
    idx = pd.Index([f"i{k}" for k in range(n)])
    return InstanceStream(
        pd.DataFrame(x, index=idx, columns=["f0", "f1"]),
        pd.DataFrame(perf, index=idx, columns=["a", "b"]),
        cutoff=BANDIT_CUTOFF,
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda imp: LinTSSelector(budget=BANDIT_CUTOFF, imputation=imp, random_state=3),
        lambda imp: LinTSSelector(
            weighting=SlidingWindow(100),
            budget=BANDIT_CUTOFF,
            imputation=imp,
            random_state=3,
        ),
    ],
    ids=["lints_incremental", "lints_window_refit"],
)
@pytest.mark.parametrize(
    "imputation", [CutoffImputer, DropCensored, ExpectedImputer, SampledImputer]
)
def test_bandit_stream_runs_and_is_deterministic(make: Any, imputation: Any) -> None:
    stream = _timeout_stream()
    runs = [
        run_prequential(make(imputation()), stream, n_warmstart=20) for _ in range(2)
    ]
    timing = [c for c in runs[0].steps.columns if "seconds" in c]
    pd.testing.assert_frame_equal(
        runs[0].steps.drop(columns=timing), runs[1].steps.drop(columns=timing)
    )
    timeouts = (runs[0].steps["user_cost"] > BANDIT_CUTOFF).mean()
    if imputation is DropCensored:
        # Dropping every timeout hides that the wrong arm is bad (BlindUCB).
        assert timeouts > 0.25
    else:
        # It learns: far fewer timeouts than random choice (about half).
        assert timeouts < 0.1


def test_bandit_imputation_is_a_recorded_parameter() -> None:
    sel = LinTSSelector(imputation=SampledImputer(max_tries=5))
    assert isinstance(sel.describe_params()["imputation"], SampledImputer)


# ===================================================== online EPM (A1)
EPM_CUTOFF = 100.0


def _epm(**kwargs: Any) -> OnlineEPMSelector:
    kwargs.setdefault("estimator", RandomForestRegressor(n_estimators=10))
    kwargs.setdefault("budget", EPM_CUTOFF)
    kwargs.setdefault("random_state", 0)
    return OnlineEPMSelector(**kwargs)


def _epm_frames(
    perf: np.ndarray, features: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.Index([f"w{k}" for k in range(len(perf))])
    if features is None:
        features = np.random.default_rng(1).normal(size=(len(perf), 2))
    feats = pd.DataFrame(features, index=idx).add_prefix("f")
    return feats, pd.DataFrame(perf, index=idx, columns=["a", "b"], dtype=float)


# a: solved 5; b: 100 at the cutoff (solved, as in the evaluator's PAR-k);
# a: 150 above the cutoff (unsolved); a: censored at the cutoff;
# a: censored at a budget of 30.
TARGET_PERF = pd.DataFrame(
    [[5.0, 100.0], [150.0, np.nan], [100.0, 7.0], [30.0, 8.0]], columns=["a", "b"]
)
TARGET_CENS = pd.DataFrame(
    [[False, False], [False, False], [True, False], [True, False]],
    columns=["a", "b"],
)


def _fitted_epm(**kwargs: Any) -> OnlineEPMSelector:
    sel = _epm(**kwargs)
    sel.fit(*_epm_frames(np.array([[5.0, 50.0]] * 4)))
    return sel


def test_epm_cutoff_and_drop_targets() -> None:
    sel = _fitted_epm(imputation=CutoffImputer())
    out = sel._training_targets(TARGET_PERF, TARGET_CENS)
    expected = np.array([[5.0, 100.0], [100.0, np.nan], [100.0, 7.0], [30.0, 8.0]])
    np.testing.assert_array_equal(out, expected)
    sel.target_transform = "log10"
    np.testing.assert_allclose(
        sel._training_targets(TARGET_PERF, TARGET_CENS), np.log10(expected)
    )

    sel = _fitted_epm(imputation=DropCensored())
    out = sel._training_targets(TARGET_PERF, TARGET_CENS)
    expected = np.array([[5.0, np.nan], [np.nan, np.nan], [np.nan, 7.0], [np.nan, 8.0]])
    np.testing.assert_array_equal(out, expected)


def test_epm_drop_does_not_learn_from_timeouts() -> None:
    sel = _fitted_epm(imputation=DropCensored(), update_every=1)
    features = pd.DataFrame([[0.0, 0.0]], index=["o0"], columns=["f0", "f1"])
    sel.observe(
        Observation("o0", features, {"a": EPM_CUTOFF}, frozenset({"a"}), EPM_CUTOFF)
    )
    assert sel.models_["a"].n_rows == 4  # the censored run was not used
    sel.observe(Observation("o1", features.rename(index={"o0": "o1"}), {"a": 3.0}))
    assert sel.models_["a"].n_rows == 5


def _censored_linear_data(
    n: int = 80, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """log10 runtime of a = 1 + x (+ small noise); runs above 100 s time out."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 2.0, n)
    true_log = 1.0 + x + rng.normal(0, 0.1, n)
    runtime = 10.0**true_log
    perf = np.column_stack([runtime, np.full(n, 20.0)])
    feats, frame = _epm_frames(perf, np.column_stack([x, np.zeros(n)]))
    return feats, frame, true_log


def _capture_targets(sel: OnlineEPMSelector) -> dict[str, np.ndarray]:
    """Record the targets each model is finally fitted on."""
    seen: dict[str, np.ndarray] = {}
    original = sel._fit_model

    def wrapper(index: int, X: np.ndarray, y: np.ndarray, sig: Any) -> Any:
        seen[sel.algorithms[index]] = y.copy()
        return original(index, X, y, sig)

    sel._fit_model = wrapper  # type: ignore[method-assign]
    return seen


def test_epm_expected_matches_schmee_hahn_impute() -> None:
    feats, perf, _ = _censored_linear_data()
    sel = _epm(
        imputation=ExpectedImputer(),
        estimator=LinearRegression(),
        uncertainty=None,
        target_transform="log10",
    )
    seen = _capture_targets(sel)
    sel.fit(feats, perf)

    y = np.log10(np.minimum(perf["a"].to_numpy(), EPM_CUTOFF))
    reference, _ = schmee_hahn_impute(
        LinearRegression, y, feats.to_numpy(), budget=math.log10(EPM_CUTOFF)
    )
    np.testing.assert_allclose(seen["a"], reference)
    censored = perf["a"].to_numpy() >= EPM_CUTOFF
    assert censored.sum() >= 10
    assert np.all(seen["a"][censored] > 2.0)  # above log10(cutoff)
    np.testing.assert_array_equal(seen["a"][~censored], y[~censored])
    np.testing.assert_array_equal(seen["b"], np.log10(20.0))  # nothing to impute


def test_epm_expected_is_closer_to_the_truth_than_the_cutoff() -> None:
    feats, perf, true_log = _censored_linear_data(n=200)
    sel = _epm(
        imputation=ExpectedImputer(),
        estimator=LinearRegression(),
        uncertainty=None,
        target_transform="log10",
    )
    seen = _capture_targets(sel)
    sel.fit(feats, perf)
    censored = perf["a"].to_numpy() >= EPM_CUTOFF
    error_expected = np.abs(seen["a"][censored] - true_log[censored]).mean()
    error_cutoff = np.abs(2.0 - true_log[censored]).mean()
    assert error_expected < 0.5 * error_cutoff


def test_epm_expected_is_deterministic_with_a_forest() -> None:
    feats, perf, _ = _censored_linear_data()

    def fitted_targets(seed: int) -> np.ndarray:
        sel = _epm(
            imputation=ExpectedImputer(), target_transform="log10", random_state=seed
        )
        seen = _capture_targets(sel)
        sel.fit(feats, perf)
        return seen["a"]

    np.testing.assert_array_equal(fitted_targets(0), fitted_targets(0))
    assert not np.array_equal(fitted_targets(0), fitted_targets(1))


@pytest.mark.parametrize(
    "imputation", [None, CutoffImputer(), DropCensored(), ExpectedImputer()]
)
def test_epm_stream_runs(imputation: CensoredImputer | None) -> None:
    stream = _timeout_stream(n=120)
    result = run_prequential(
        _epm(imputation=imputation, target_transform="log10"), stream, n_warmstart=20
    )
    assert len(result.steps) == 100


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"imputation": SampledImputer()}, ValueError),
        ({"imputation": CutoffImputer(), "maximize": True}, ValueError),
        ({"imputation": "cutoff"}, TypeError),
        ({"imputation_max_iter": 0}, ValueError),
        ({"imputation_tol": 0.0}, ValueError),
    ],
)
def test_epm_invalid_imputation(kwargs: dict[str, Any], error: type) -> None:
    with pytest.raises(error):
        _epm(**kwargs)
