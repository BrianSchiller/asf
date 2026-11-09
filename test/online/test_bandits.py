"""Tests for the composable bandit selectors (UCB, TS, LinUCB, LinTS + variants)."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from asf.online.base import Observation
from asf.online.evaluation import run_prequential
from asf.online.selectors.bandits import (
    AbstractContextualBandit,
    BayesianLinearRegression,
    ConfidenceWeighting,
    DiscountedTSSelector,
    DiscountedUCBSelector,
    ExponentialDiscount,
    FDSWTSSelector,
    GaussianConfidence,
    GaussianPosterior,
    LinTSSelector,
    LinUCBSelector,
    NearestNeighbours,
    RewardTransform,
    SlidingWindow,
    SlidingWindowTSSelector,
    SlidingWindowUCBSelector,
    ThompsonPolicy,
    ThompsonSamplingSelector,
    UCBSelector,
    Uniform,
    Weighting,
)
from asf.online.streams.stream import InstanceStream

ALGOS = ["a", "b", "c"]
CUTOFF = 1000.0
SELECTORS = [
    UCBSelector,
    ThompsonSamplingSelector,
    LinUCBSelector,
    LinTSSelector,
    DiscountedTSSelector,
    SlidingWindowTSSelector,
    FDSWTSSelector,
    DiscountedUCBSelector,
    SlidingWindowUCBSelector,
]


# ---------------------------------------------------------------- helpers
def _stream(
    features: np.ndarray, perf: np.ndarray, drift_points: tuple[int, ...] = ()
) -> InstanceStream:
    idx = pd.Index([f"i{k}" for k in range(len(features))])
    cols = [f"f{j}" for j in range(features.shape[1])]
    return InstanceStream(
        pd.DataFrame(features, index=idx, columns=cols),
        pd.DataFrame(perf, index=idx, columns=ALGOS),
        cutoff=CUTOFF,
        drift_points=tuple(drift_points),
    )


def region_stream(n: int = 600, seed: int = 0) -> InstanceStream:
    """Stationary; best arm depends on the sign of f0 (a if f0<0, b otherwise).

    Wrong choices time out (2000 > cutoff), c is mediocre everywhere.
    """
    rng = np.random.default_rng(seed)
    f0 = rng.choice([-3.0, 3.0], n) + rng.normal(0, 0.5, n)
    x = np.column_stack([f0, rng.normal(size=n)])
    perf = np.where((f0 < 0)[:, None], [[5, 2000, 300]], [[2000, 5, 300]])
    return _stream(x, perf.astype(float))


def abrupt_drift_stream(
    n: int = 800, drift: int = 400, seed: int = 0
) -> InstanceStream:
    """Context carries no information; best arm switches a -> b at ``drift``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    before = np.arange(n) < drift
    perf = np.where(before[:, None], [[5, 2000, 300]], [[2000, 5, 300]])
    return _stream(x, perf.astype(float), drift_points=(drift,))


def _obs(stream: InstanceStream, i: int, algos: list[str]) -> Observation:
    item_id = stream.features.index[i]
    truth = stream.true_runtimes(item_id)
    runs = {a: float(min(truth[a], CUTOFF)) for a in algos}
    censored = frozenset(a for a in algos if truth[a] > CUTOFF)
    return Observation(item_id, stream.features.iloc[[i]], runs, censored, CUTOFF)


class _ForceRefit(Weighting):
    """Same weights as ``inner`` but forces the bandit into refit mode."""

    incremental = False

    def __init__(self, inner: Weighting) -> None:
        self.inner = inner

    def weights(
        self, z: np.ndarray, t: int, steps: np.ndarray, Z: np.ndarray
    ) -> np.ndarray:
        return self.inner.weights(z, t, steps, Z)


# ---------------------------------------------------------------- posteriors
def test_linear_posterior_matches_weighted_ridge() -> None:
    rng = np.random.default_rng(0)
    X, r, w = rng.normal(size=(30, 4)), rng.normal(size=30), rng.uniform(0, 2, 30)
    post = BayesianLinearRegression(4, prior_precision=0.5)
    post.fit(X, r, w)
    expected = np.linalg.solve(0.5 * np.eye(4) + X.T @ (w[:, None] * X), X.T @ (w * r))
    np.testing.assert_allclose(post.theta, expected)

    inc = BayesianLinearRegression(4, prior_precision=0.5)
    for xi, ri, wi in zip(X, r, w):
        inc.update(xi, ri, wi)
    np.testing.assert_allclose(inc.theta, expected)


def test_weight_two_equals_duplicated_observation() -> None:
    x, r = np.array([1.0, 2.0]), 0.7
    a, b = BayesianLinearRegression(2), BayesianLinearRegression(2)
    a.update(x, r, 2.0)
    b.update(x, r)
    b.update(x, r)
    np.testing.assert_allclose(a.theta, b.theta)
    assert a.width(x) == pytest.approx(b.width(x))


def test_gaussian_posterior_is_linear_posterior_on_constant_input() -> None:
    rng = np.random.default_rng(1)
    g = GaussianPosterior(prior_precision=0.8, noise_var=0.3)
    lin = BayesianLinearRegression(1, prior_precision=0.8, noise_var=0.3)
    one = np.ones(1)
    for _ in range(20):
        r, w = rng.normal(), rng.uniform(0.1, 2)
        g.update(one, r, w)
        lin.update(one, r, w)
        g.decay(0.9)
        lin.decay(0.9)
    assert g.mean(one) == pytest.approx(lin.mean(one))
    assert g.width(one) == pytest.approx(lin.width(one))


def test_decay_one_is_noop_and_reset_restores_prior() -> None:
    post = BayesianLinearRegression(3)
    post.update(np.ones(3), 1.0)
    A = post.A.copy()
    post.decay(1.0)
    np.testing.assert_allclose(post.A, A)
    post.reset()
    np.testing.assert_allclose(post.A, np.eye(3))
    assert GaussianPosterior(prior_precision=0).width() == float("inf")


def test_full_decay_keeps_prior() -> None:
    post = BayesianLinearRegression(2, prior_precision=2.0)
    post.update(np.ones(2), 1.0)
    post.decay(1e-300)
    np.testing.assert_allclose(post.A, 2.0 * np.eye(2), atol=1e-12)


# ---------------------------------------------------------------- weightings
def test_basic_weightings() -> None:
    steps = np.array([0, 1, 2, 3, 4])
    Z = np.array([[0.0], [10.0], [1.0], [-1.0], [9.0]])
    z, t = np.array([0.2]), 5
    np.testing.assert_allclose(Uniform().weights(z, t, steps, Z), np.ones(5))
    np.testing.assert_allclose(
        ExponentialDiscount(0.5).weights(z, t, steps, Z),
        [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1],
    )
    np.testing.assert_allclose(
        SlidingWindow(2).weights(z, t, steps, Z), [0, 0, 0, 1, 1]
    )
    np.testing.assert_allclose(
        NearestNeighbours(3).weights(z, t, steps, Z), [1, 0, 1, 1, 0]
    )
    combined = ExponentialDiscount(0.5) * NearestNeighbours(3)
    np.testing.assert_allclose(
        combined.weights(z, t, steps, Z), [1 / 16, 0, 1 / 4, 1 / 2, 0]
    )
    assert not combined.incremental
    both_inc = ExponentialDiscount(0.5) * ExponentialDiscount(0.8)
    assert both_inc.incremental and both_inc.decay_factor == pytest.approx(0.4)


# ---------------------------------------------------------------- equivalence
@pytest.mark.parametrize("cls", [ThompsonSamplingSelector, LinTSSelector])
@pytest.mark.parametrize(
    "make_weighting, use_confidence",
    [
        (Uniform, False),
        (lambda: ExponentialDiscount(0.9), False),
        (lambda: ExponentialDiscount(0.95), True),
    ],
)
def test_incremental_and_refit_modes_agree(cls, make_weighting, use_confidence) -> None:  # noqa: ANN001
    stream = region_stream(n=120)
    sels = []
    for refit in (False, True):
        w = make_weighting()
        sel = cls(
            weighting=_ForceRefit(w) if refit else w,
            confidence=GaussianConfidence() if use_confidence else None,
            budget=CUTOFF,
            random_state=0,
        )
        assert sel.incremental is not refit
        sel.fit(stream.features.iloc[:40], stream.performance.iloc[:40])
        rng = np.random.default_rng(3)
        for i in range(40, 120):  # same scripted feedback for both
            sel.observe(_obs(stream, i, [str(rng.choice(ALGOS))]))
        sels.append(sel)

    z = sels[0]._feature_vector(stream.features.iloc[[0]])
    x = np.append(z, 1.0) if sels[0].contextual else np.ones(1)
    for arm in ALGOS:
        p_inc = sels[0]._posterior_for(arm, z, 0)
        p_ref = sels[1]._posterior_for(arm, z, 0)
        assert p_inc.mean(x) == pytest.approx(p_ref.mean(x), rel=1e-8, abs=1e-12)
        assert p_inc.width(x) == pytest.approx(p_ref.width(x), rel=1e-8)


# ---------------------------------------------------------------- behaviour
def test_ucb_tries_every_arm_before_repeating() -> None:
    stream = region_stream(n=10)
    sel = UCBSelector(random_state=0)
    sel.fit(stream.features.iloc[:0], stream.performance.iloc[:0])
    played = []
    for i in range(3):
        arm = sel.propose(stream.features.iloc[[i]])[0][0]
        played.append(arm)
        sel.observe(_obs(stream, i, [arm]))
    assert sorted(played) == ALGOS


def test_thompson_sampling_is_reproducible() -> None:
    def run() -> list[str]:
        stream = region_stream(n=60)
        sel = LinTSSelector(random_state=7, budget=CUTOFF)
        sel.fit(stream.features.iloc[:10], stream.performance.iloc[:10])
        out = []
        for i in range(10, 60):
            arm = sel.propose(stream.features.iloc[[i]])[0][0]
            out.append(arm)
            sel.observe(_obs(stream, i, [arm]))
        return out

    assert run() == run()


def test_reset_arms_forgets_everything() -> None:
    stream = region_stream(n=30)
    sel = ThompsonSamplingSelector(budget=CUTOFF)
    sel.fit(stream.features.iloc[:30], stream.performance.iloc[:30])
    assert sel.history_.n_observations("a") == 30
    sel.reset_arms()
    assert sel.history_.n_observations("a") == 0
    assert sel.posteriors_ == {}
    assert sel.propose(stream.features.iloc[[0]])[0][0] in ALGOS


def test_rewards() -> None:
    assert RewardTransform().resolve(1000.0, False) == "log"
    rt = RewardTransform("par")
    out = rt({"a": 10.0, "b": 1000.0}, {"b"}, 1000.0, False)
    assert out == {"a": pytest.approx(-0.001), "b": -1.0}
    log = RewardTransform("log")({"a": 9.0}, set(), float("inf"), False)
    assert log["a"] == pytest.approx(-1.0)
    assert RewardTransform()({"a": 3.0}, set(), float("inf"), True) == {"a": 3.0}
    with pytest.raises(ValueError):
        RewardTransform("par")({"a": 1.0}, set(), float("inf"), False)


def test_warm_start_without_budget_warns() -> None:
    stream = region_stream(n=20)
    sel = UCBSelector()
    sel.fit(stream.features.iloc[:10], stream.performance.iloc[:10])
    with pytest.warns(UserWarning, match="budget"):
        sel.observe(_obs(stream, 10, ["a"]))


def test_confidence_fixed_at_arrival_and_re_estimated() -> None:
    model = GaussianConfidence()
    model.fit(np.random.default_rng(0).normal(size=(100, 2)))
    near, far = model.score(np.array([[0.0, 0.0], [6.0, 6.0]]))
    assert near > 0.9 and far < 0.01

    stream = region_stream(n=80)
    fixed = LinTSSelector(confidence=GaussianConfidence(), budget=CUTOFF)
    fixed.fit(stream.features.iloc[:40], stream.performance.iloc[:40])
    w = fixed.history_.get("a").weights
    assert np.all((w > 0) & (w <= 1)) and w.std() > 0

    reest = LinTSSelector(
        weighting=ConfidenceWeighting(GaussianConfidence(forgetting=0.5)),
        budget=CUTOFF,
    )
    reest.fit(stream.features.iloc[:40], stream.performance.iloc[:40])
    data = reest.history_.get("a")
    before = reest.weighting.weights(None, 0, data.steps, data.Z)
    for i in range(40, 45):
        reest.observe(_obs(stream, i, ["a"]))
    data = reest.history_.get("a")
    after = reest.weighting.weights(None, 0, data.steps, data.Z)
    assert not np.allclose(before, after[: len(before)])  # history re-scored


def test_custom_assembly_from_components() -> None:
    sel = AbstractContextualBandit(
        posterior=partial(GaussianPosterior, prior_precision=1.0, noise_var=0.1),
        policy=ThompsonPolicy(),
        contextual=False,
        weighting=ExponentialDiscount(0.98) * NearestNeighbours(30),
        budget=CUTOFF,
        random_state=0,
    )
    result = run_prequential(sel, region_stream(n=200), n_warmstart=20)
    assert len(result.steps) == 180


# ---------------------------------------------------------------- learning
def _tail(
    sel: AbstractContextualBandit, stream: InstanceStream, n_warm: int, last: int
) -> pd.DataFrame:
    return run_prequential(sel, stream, n_warmstart=n_warm).steps.tail(last)


# Rewards lie in [-1, 0]; noise_var=1 (the textbook default) would make the
# posteriors far too wide for these short streams.
KW = dict(budget=CUTOFF, random_state=0, noise_var=0.05)


def test_linear_bandits_learn_feature_dependent_best_arm() -> None:
    stream = region_stream(n=500)
    ts = _tail(ThompsonSamplingSelector(**KW), stream, 0, 200)
    for cls in (LinUCBSelector, LinTSSelector):
        lin = _tail(cls(**KW), stream, 0, 200)
        # A linear model of a step function is misspecified, so c is still
        # picked near the boundary, but a / b take over in their regions.
        assert lin["user_cost"].mean() < 0.6 * ts["user_cost"].mean()


def test_non_stationary_weightings_switch_arm_after_abrupt_drift() -> None:
    stream = abrupt_drift_stream()  # best arm a -> b at step 400
    stationary = _tail(ThompsonSamplingSelector(**KW), stream, 20, 300)
    assert (stationary["primary"] == "b").mean() < 0.05  # never finds b again
    for weighting in (
        ExponentialDiscount(0.95),
        ExponentialDiscount(0.99),
        SlidingWindow(50),
    ):
        adapted = _tail(
            ThompsonSamplingSelector(weighting=weighting, **KW), stream, 20, 300
        )
        assert (adapted["primary"] == "b").mean() > 0.5


def test_nearest_neighbour_window_makes_plain_ts_contextual() -> None:
    stream = region_stream(n=500)
    plain = _tail(ThompsonSamplingSelector(**KW), stream, 20, 200)
    nn = _tail(
        ThompsonSamplingSelector(weighting=NearestNeighbours(5), **KW),
        stream,
        20,
        200,
    )
    assert nn["user_cost"].mean() < 0.6 * plain["user_cost"].mean()


# ---------------------------------------------------------------- integration
@pytest.mark.parametrize("cls", SELECTORS)
@pytest.mark.parametrize("n_warm", [0, 30])
def test_prequential_integration(cls, n_warm) -> None:  # noqa: ANN001
    stream = region_stream(n=120)
    sel = cls(budget=CUTOFF, random_state=0)
    result = run_prequential(sel, stream, n_warmstart=n_warm)
    assert len(result.steps) == 120 - n_warm
    assert (result.steps["n_runs"] == 1).all()
    assert set(result.steps["primary"]) <= set(ALGOS)
    batch = sel.predict(stream.features.iloc[:5])
    assert all(len(s) == 1 for s in batch.values())


def test_maximize_mode_uses_raw_quality() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 2))
    quality = np.tile([[0.9, 0.1, 0.5]], (n, 1))
    idx = pd.Index([f"i{k}" for k in range(n)])
    stream = InstanceStream(
        pd.DataFrame(x, index=idx, columns=["f0", "f1"]),
        pd.DataFrame(quality, index=idx, columns=ALGOS),
        cutoff=float("inf"),
    )
    sel = ThompsonSamplingSelector(maximize=True, noise_var=0.05, random_state=0)
    res = run_prequential(sel, stream, n_warmstart=10)
    assert (res.steps["primary"].tail(50) == "a").mean() > 0.9


# ---------------------------------------------------------------- DS-TS, SW-TS, f-dsw
def _scripted(sel: AbstractContextualBandit, n: int = 60) -> AbstractContextualBandit:
    """Cold start, then ``n`` observations of randomly chosen arms."""
    stream = region_stream(n=n)
    sel.fit(stream.features.iloc[:0], stream.performance.iloc[:0])
    rng = np.random.default_rng(0)
    for i in range(n):
        sel.observe(_obs(stream, i, [str(rng.choice(ALGOS))]))
    return sel


def test_ds_ts_and_sw_ts_match_paper_formulas() -> None:
    gamma, tau, sigma2 = 0.9, 20, 0.5
    for sel, paper_weights in [
        # DS-TS: N = sum_j gamma^(t-j); round j = step + 1, t = clock + 1
        (
            DiscountedTSSelector(gamma, sigma2, budget=CUTOFF),
            lambda s, t: gamma ** (t - s),
        ),
        # SW-TS: only the last tau steps
        (
            SlidingWindowTSSelector(tau, sigma2, budget=CUTOFF),
            lambda s, t: (t - s <= tau) * 1.0,
        ),
    ]:
        _scripted(sel)
        data = sel.history_.get("a")
        w = paper_weights(data.steps, sel.clock_)
        post = sel._posterior_for("a", np.zeros(2), 0)
        assert post.mean() == pytest.approx(w @ data.rewards / w.sum())
        assert post.width() ** 2 == pytest.approx(sigma2 / w.sum())


def test_fdsw_traces() -> None:
    sel = _scripted(FDSWTSSelector(gamma=0.9, window=5, budget=CUTOFF))
    data = sel.history_.get("a")
    hist = sel._posterior_for("a", np.zeros(2), 0, "historic")
    hot = sel._posterior_for("a", np.zeros(2), 0, "hot")
    discount = 0.9 ** (sel.clock_ - 1 - data.steps)
    assert hist.alpha == pytest.approx(discount @ data.rewards)
    assert hot.alpha == pytest.approx(data.rewards[-5:].sum())
    assert hot.alpha + hot.beta == pytest.approx(5)
    unit = RewardTransform("unit")({"a": 0.0, "b": 5.0}, {"b"}, 5.0, False)
    assert unit == {"a": 1.0, "b": 0.0}
    with pytest.raises(ValueError):
        ThompsonSamplingSelector(weighting=Uniform(), traces={"x": Uniform()})


def test_non_stationary_ts_presets_switch_arm_after_drift() -> None:
    stream = abrupt_drift_stream()
    for sel in (
        DiscountedTSSelector(0.99, noise_var=0.05, budget=CUTOFF, random_state=0),
        SlidingWindowTSSelector(100, noise_var=0.05, budget=CUTOFF, random_state=0),
        FDSWTSSelector(budget=CUTOFF, random_state=0),
    ):
        steps = _tail(sel, stream, 20, 300)
        assert (steps["primary"] == "b").mean() > 0.5, type(sel).__name__


def test_ds_ucb_and_sw_ucb_match_paper_indices() -> None:
    gamma, tau, xi = 0.9, 20, 0.6
    ds = _scripted(DiscountedUCBSelector(gamma, xi=xi, budget=CUTOFF))
    sw = _scripted(SlidingWindowUCBSelector(tau, xi=xi, budget=CUTOFF))
    for sel, weights, bonus in [
        (
            ds,
            lambda s, t: gamma ** (t - s),
            lambda n, N: 2 * np.sqrt(xi * np.log(n) / N),
        ),
        (
            sw,
            lambda s, t: (t - s <= tau) * 1.0,
            lambda n, N: np.sqrt(xi * np.log(min(sel.clock_ + 1, tau)) / N),
        ),
    ]:
        data = {a: sel.history_.get(a) for a in ALGOS}
        N = {a: weights(d.steps, sel.clock_).sum() for a, d in data.items()}
        n = sum(N.values())
        d, w = data["a"], weights(data["a"].steps, sel.clock_)
        expected = w @ d.rewards / N["a"] + bonus(n, N["a"])
        z = np.zeros(2)
        t = sel._policy_time(z, 0)
        got = sel.policy.score(sel._posterior_for("a", z, 0), np.ones(1), t, sel.rng_)
        assert got == pytest.approx(expected)
