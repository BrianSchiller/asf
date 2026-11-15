"""Tests for the KS subspace ensemble (EDFS-style) covariate drift detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from asf.online.drift import KSSubspaceEnsembleDetector
from asf.online.drift.ks_subspace_ensemble import (
    build_subspaces,
    ks_critical_value,
    ks_statistic,
)
from asf.online.evaluation import run_prequential
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams.stream import InstanceStream

D = 12
NAMES = [f"f{i}" for i in range(D)]
BLOCKS = [NAMES[i : i + 3] for i in range(0, D, 3)]
BLOCK0 = np.r_[np.full(3, 2.0), np.zeros(D - 3)]


def sample(
    n: int, seed: int, shift: np.ndarray | None = None, noise: float = 0.5
) -> pd.DataFrame:
    """Four blocks of three correlated features (one latent factor per block)."""
    rng = np.random.default_rng(seed)
    x = np.repeat(rng.normal(size=(n, 4)), 3, axis=1) + noise * rng.normal(size=(n, D))
    if shift is not None:
        x = x + shift
    return pd.DataFrame(x, columns=NAMES)


def shift_of(*features: str, by: float = 5.0) -> np.ndarray:
    return np.array([by if name in features else 0.0 for name in NAMES])


def feed(det: KSSubspaceEnsembleDetector, data: pd.DataFrame) -> list[int]:
    return [i for i in range(len(data)) if det.update(data.iloc[[i]]).drift]


def pair_corr(strength: float) -> np.ndarray:
    """Features (0, 1) and (2, 3) are correlated with ``strength``, else 0."""
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = corr[2, 3] = corr[3, 2] = strength
    return corr


# ------------------------------------------------------------------ KS test
def test_ks_statistic_hand_computed() -> None:
    # F_a - F_b at 1..6: .25, .5, .5, .5, .25, 0
    assert ks_statistic(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])) == 0.5
    # ties: at 1 -> 2/3 - 1/3, at 2 -> 1 - 1
    assert ks_statistic(np.array([1, 1, 2]), np.array([1, 2, 2])) == pytest.approx(
        1 / 3
    )
    assert ks_statistic(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert ks_statistic(np.array([1.0, 2.0]), np.array([3.0])) == 1.0


def test_ks_statistic_matches_scipy_and_drops_nan() -> None:
    rng = np.random.default_rng(0)
    for n, m in [(30, 30), (200, 50), (7, 91)]:
        a, b = rng.normal(size=n), rng.normal(0.3, 2.0, size=m).round(1)
        assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)
    with_nan = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    assert ks_statistic(with_nan, np.array([3.0, 4.0, 5.0, 6.0])) == 0.5
    with pytest.raises(ValueError):
        ks_statistic(np.array([np.nan]), np.array([1.0]))


def test_ks_critical_value() -> None:
    # c(0.05) = sqrt(-ln(0.025) / 2) = 1.3581, times sqrt(100 / 2500) = 0.2
    # c(0.01) = sqrt(-ln(0.005) / 2) = 1.6276, times sqrt(250 / 10000) = 0.1581
    assert ks_critical_value(0.05, 50, 50) == pytest.approx(0.27162, abs=1e-5)
    assert ks_critical_value(0.01, 200, 50) == pytest.approx(0.25735, abs=1e-5)
    assert ks_critical_value(0.01, 50, 200) == ks_critical_value(0.01, 200, 50)
    with pytest.raises(ValueError):
        ks_critical_value(0.0, 10, 10)
    with pytest.raises(ValueError):
        ks_critical_value(0.05, 0, 10)


# ---------------------------------------------------------------- subspaces
@pytest.mark.parametrize("seed", range(10))
def test_subspaces_diversity_spreads_uncorrelated_features(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # no correlation: only the diversity terms act, so nothing is reused
    subs = build_subspaces(np.eye(4), 2, 2, rng)
    assert sorted(subs[0] + subs[1]) == [0, 1, 2, 3]
    assert sorted(f for s in build_subspaces(np.eye(4), 4, 1, rng) for f in s) == [
        0,
        1,
        2,
        3,
    ]
    # more subspaces than features: the first two still take different ones
    subs = build_subspaces(np.eye(2), 3, 1, rng)
    assert subs[0] != subs[1] and len(subs) == 3


@pytest.mark.parametrize("seed", range(10))
def test_subspaces_quality_picks_the_most_correlated_feature(seed: int) -> None:
    corr = np.array([[1.0, 0.9, 0.2], [0.9, 1.0, 0.5], [0.2, 0.5, 1.0]])
    first, second = build_subspaces(corr, 1, 2, np.random.default_rng(seed))[0]
    assert second == {0: 1, 1: 0, 2: 1}[first]


@pytest.mark.parametrize("seed", range(10))
def test_subspaces_quality_against_diversity(seed: int) -> None:
    """
    Two correlated pairs, k = 2, n = 2. With the first features a (S0), b (S1):

    * strength 0.9: the partner always wins. For S0 it scores at least
      0.9 + (0.5 + 0.5) / 2 = 1.4, an unrelated unused feature 0 + 1 = 1.
      For S1 the partner scores at least 0.9 + (0.5 + 0) / 2 = 1.15 > 1.
    * strength 0.3: diversity wins. If b is a's partner, b scores
      0.3 + (0.5 + 0.5) / 2 = 0.8 < 1 for S0, and a scores 0.8 < 1 for S1,
      so both take a feature of the other pair; otherwise each takes its
      own, unused partner (0.3 + 1). Either way no feature is used twice.
    """
    rng = np.random.default_rng(seed)
    strong = build_subspaces(pair_corr(0.9), 2, 2, rng)
    assert all(sorted(s) in ([0, 1], [2, 3]) for s in strong)
    weak = build_subspaces(pair_corr(0.3), 2, 2, rng)
    assert sorted(weak[0] + weak[1]) == [0, 1, 2, 3]


def test_build_subspaces_validation_and_nan() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_subspaces(np.eye(3), 2, 4, rng)
    with pytest.raises(ValueError):
        build_subspaces(np.eye(3), 0, 1, rng)
    with pytest.raises(ValueError):
        build_subspaces(np.ones((2, 3)), 1, 1, rng)
    corr = np.full((3, 3), np.nan)  # e.g. constant features
    assert sorted(build_subspaces(corr, 1, 3, rng)[0]) == [0, 1, 2]


def test_guided_subspaces_follow_the_correlation_blocks() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(300, 0)
    )
    assert len(det.subspaces_) == 10
    assert all(set(s) in map(set, BLOCKS) for s in det.subspaces_)
    assert {f for s in det.subspaces_ for f in s} == set(NAMES)  # 10 starts, 4 blocks


def test_subspace_size_and_random_subspaces() -> None:
    ref = sample(100, 0)
    det = KSSubspaceEnsembleDetector().fit(ref)  # ceil(0.1 * 12) = 2
    assert [len(s) for s in det.subspaces_] == [2] * 10
    det = KSSubspaceEnsembleDetector(n_subspaces=3, subspace_size=1.0).fit(ref)
    assert all(sorted(s) == sorted(NAMES) for s in det.subspaces_)
    det = KSSubspaceEnsembleDetector(n_subspaces=3, subspace_size=50).fit(ref)
    assert [len(s) for s in det.subspaces_] == [D] * 3  # capped
    det = KSSubspaceEnsembleDetector(
        n_subspaces=5, subspace_size=4, subspaces="random", random_state=1
    ).fit(ref)
    assert all(len(set(s)) == 4 and set(s) <= set(NAMES) for s in det.subspaces_)


# ------------------------------------------------------------------- voting
@pytest.mark.parametrize(
    ("shifted", "expected_score", "expected_drift"),
    [
        ((), 0.0, False),
        (("f0",), 1 / 3, False),  # 1 of 3 tests rejects: no majority
        (("f0", "f3"), 2 / 3, True),  # 2 of 3
        (("f0", "f3", "f6"), 1.0, True),
    ],
)
def test_majority_vote_inside_a_subspace(
    shifted: tuple[str, ...], expected_score: float, expected_drift: bool
) -> None:
    det = KSSubspaceEnsembleDetector(subspaces=[["f0", "f3", "f6"]], random_state=0)
    det.fit(sample(200, 0))
    signal = det.update(sample(50, 1, shift=shift_of(*shifted)))
    assert signal.drift is expected_drift
    assert signal.score == pytest.approx(expected_score)
    assert det.last_votes_ == {f: f in shifted for f in ("f0", "f3", "f6")}


def test_majority_is_strict() -> None:
    det = KSSubspaceEnsembleDetector(subspaces=[["f0", "f3"]]).fit(sample(200, 0))
    signal = det.update(sample(50, 1, shift=shift_of("f0")))
    assert not signal.drift and signal.score == 0.5


def test_one_drifting_subspace_is_enough() -> None:
    groups = [["f0", "f3", "f6"], ["f1", "f4", "f7"], ["f2", "f5", "f8"]]
    det = KSSubspaceEnsembleDetector(subspaces=groups).fit(sample(200, 0))
    # one rejecting test in each subspace: three rejections, but no majority
    assert not det.update(sample(50, 1, shift=shift_of("f0", "f1", "f2"))).drift
    # two rejections, both in the second subspace
    signal = det.update(sample(50, 2, shift=shift_of("f1", "f4")))
    assert signal.drift and signal.score == pytest.approx(2 / 3)


def test_unmonitored_features_are_ignored() -> None:
    det = KSSubspaceEnsembleDetector(subspaces=[["f0", "f3"]]).fit(sample(200, 0))
    signal = det.update(sample(50, 1, shift=shift_of("f1", "f2", "f4", "f5")))
    assert not signal.drift and signal.score == 0.0
    assert set(det.last_votes_ or {}) == {"f0", "f3"}


# ------------------------------------------------------------------- timing
def test_signals_only_when_a_window_completes() -> None:
    det = KSSubspaceEnsembleDetector(window_size=20, random_state=0).fit(sample(200, 0))
    drifted = sample(40, 1, shift=np.full(D, 5.0))
    signals = [det.update(drifted.iloc[[i]]) for i in range(40)]
    assert all(not s.drift and s.score is None for s in signals[:19])
    assert [i for i, s in enumerate(signals) if s.drift] == [19, 39]


def test_stride_checks_more_often() -> None:
    kw = dict(window_size=20, subspace_size=3, random_state=0)
    data = pd.concat(
        [sample(20, 1), sample(40, 2, shift=np.full(D, 5.0))], ignore_index=True
    )
    slow = feed(KSSubspaceEnsembleDetector(**kw).fit(sample(200, 0)), data)
    fast = feed(KSSubspaceEnsembleDetector(stride=5, **kw).fit(sample(200, 0)), data)
    assert slow[0] == 39  # first full window after the shift at position 20
    assert 20 <= fast[0] < 39 and (fast[0] + 1) % 5 == 0


def test_multi_row_update_reports_any_drift() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(200, 0)
    )
    data = pd.concat([sample(50, 1, shift=BLOCK0), sample(50, 2)], ignore_index=True)
    assert det.update(data).drift  # the first of the two windows drifted


# ---------------------------------------------------------------- behaviour
def test_stationary_data_rarely_alarms() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(400, 0)
    )
    assert len(feed(det, sample(1000, 1))) <= 1  # 20 checks


def test_shift_in_one_block_is_detected_within_two_windows() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(300, 0)
    )
    pre = sample(120, 2)
    hits = feed(det, pd.concat([pre, sample(300, 3, shift=BLOCK0)], ignore_index=True))
    after = [h for h in hits if h >= len(pre)]
    assert after and after[0] - len(pre) <= 2 * 50
    votes = det.last_votes_ or {}
    assert all(votes[f] for f in BLOCKS[0])


def test_larger_spread_is_detected() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(300, 0)
    )
    assert feed(det, sample(200, 1, noise=3.0))


def test_reset_takes_new_reference_and_clears_state() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(300, 0)
    )
    drifted = sample(300, 1, shift=BLOCK0)
    assert feed(det, drifted)
    det.reset(drifted)  # new reference = the drifted regime
    assert det.last_score_ is None and det.last_votes_ is None
    assert len(feed(det, sample(300, 2, shift=BLOCK0))) <= 1


def test_mixed_reference_alarms_again_unless_reference_is_recent() -> None:
    """After a drift the buffer still holds old data: 150 old + 50 new rows."""
    mixed = pd.concat([sample(150, 0), sample(50, 1, shift=BLOCK0)], ignore_index=True)
    new = sample(200, 2, shift=BLOCK0)
    kw = dict(subspace_size=3, random_state=0)
    whole = KSSubspaceEnsembleDetector(**kw).fit(mixed)
    recent = KSSubspaceEnsembleDetector(reference_size=50, **kw).fit(mixed)
    assert len(feed(whole, new)) == 4  # every check: new data != mixed reference
    assert len(feed(recent, new)) == 0


# -------------------------------------------------------------------- score
def test_instance_score_is_share_of_features_outside_seen_range() -> None:
    ref = pd.DataFrame(
        {"a": [0.0, 1.0, 0.5], "b": [0.0, 2.0, 1.0], "c": [-1.0, 1.0, 0.0]}
    )
    det = KSSubspaceEnsembleDetector(subspaces=[["a", "b"], ["c"]]).fit(ref)
    new = pd.DataFrame(
        {
            "a": [0.5, 0.0, 1.5, 1.5, -0.1, 0.5, np.nan],
            "b": [1.0, 2.0, 1.0, 2.5, -0.1, 1.0, 5.0],
            "c": [0.0, 1.0, 0.0, 0.0, 0.0, -2.0, np.nan],
        }
    )
    np.testing.assert_allclose(det.score(new), [0, 0, 0.5, 1, 1, 1, 0.5])


def test_instance_score_on_shifted_data() -> None:
    det = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(
        sample(300, 0)
    )
    inside, outside = det.score(sample(100, 1)), det.score(sample(100, 1, by_far()))
    assert inside.shape == (100,) and inside.mean() < 0.1
    assert (outside == 1.0).all()


def by_far() -> np.ndarray:
    return np.r_[np.full(3, 50.0), np.zeros(D - 3)]


# --------------------------------------------------------------- edge cases
def test_few_features() -> None:
    ref, new = sample(200, 0)[["f0", "f3", "f6"]], sample(50, 1, shift=np.full(D, 5.0))
    det = KSSubspaceEnsembleDetector(random_state=0).fit(ref)  # k = 10 > 3 features
    assert len(det.subspaces_) == 10 and all(len(s) == 1 for s in det.subspaces_)
    assert {s[0] for s in det.subspaces_} == {"f0", "f3", "f6"}
    assert det.update(new).drift
    single = KSSubspaceEnsembleDetector(random_state=0).fit(ref[["f0"]])
    assert single.update(new).drift and not single.update(sample(50, 2)).drift


def test_missing_values_constant_columns_and_column_order() -> None:
    ref = sample(200, 0)
    ref["const"] = 7.0
    ref["empty"] = np.nan
    ref.iloc[::7, 2] = np.nan
    groups = [["f0", "f1", "f2"], ["const"], ["empty"], ["f3", "f4", "f5"]]
    det = KSSubspaceEnsembleDetector(subspaces=groups).fit(ref)
    new = sample(50, 1)
    new["const"] = 7.0
    new["empty"] = 1.0
    new.iloc[::5, 1] = np.nan
    new = new[list(reversed(new.columns))]  # different column order
    signal = det.update(new)
    assert not signal.drift and signal.score == 0.0
    assert np.isfinite(det.score(new)).all()

    new["const"] = 8.0  # a constant that changes is a drift of its subspace
    assert det.update(new).drift
    assert det.last_votes_ == {
        **{f: False for f in NAMES[:6]},
        "const": True,
        "empty": False,
    }

    guided = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(ref)
    assert not guided.update(sample(50, 1).assign(const=7.0, empty=np.nan)).drift
    with pytest.raises(ValueError):
        det.update(new.drop(columns=["f0"]))


def test_validation() -> None:
    with pytest.raises(RuntimeError):
        KSSubspaceEnsembleDetector().update(sample(1, 0))
    with pytest.raises(RuntimeError):
        KSSubspaceEnsembleDetector().score(sample(1, 0))
    with pytest.raises(ValueError):
        KSSubspaceEnsembleDetector().fit(sample(1, 0))
    with pytest.raises(ValueError):
        KSSubspaceEnsembleDetector(subspaces=[["nope"]]).fit(sample(10, 0))
    for kwargs in (
        {"n_subspaces": 0},
        {"subspace_size": 0},
        {"subspace_size": 1.5},
        {"subspace_size": True},
        {"subspaces": "nope"},
        {"subspaces": [[]]},
        {"window_size": 1},
        {"window_size": 10, "stride": 11},
        {"alpha": 1.0},
        {"reference_size": 1},
    ):
        with pytest.raises(ValueError):
            KSSubspaceEnsembleDetector(**kwargs)


# -------------------------------------------------------------- determinism
@pytest.mark.parametrize("subspaces", ["guided", "random"])
def test_same_random_state_gives_same_subspaces_and_signals(subspaces: str) -> None:
    data = pd.concat([sample(100, 1), sample(200, 2, shift=BLOCK0)], ignore_index=True)
    runs = []
    for _ in range(2):
        det = KSSubspaceEnsembleDetector(
            subspace_size=3, subspaces=subspaces, random_state=7
        ).fit(sample(300, 0))
        signals = [det.update(data.iloc[[i]]) for i in range(len(data))]
        runs.append((det.subspaces_, signals))
    assert runs[0] == runs[1]
    refit = det.fit(sample(300, 0)).subspaces_  # fit restarts the generator
    assert refit == runs[0][0]


def test_performance_is_ignored() -> None:
    ref, new = sample(200, 0), sample(50, 1, shift=BLOCK0)
    perf = pd.DataFrame({"a": np.arange(200.0)})
    a = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(ref, perf)
    b = KSSubspaceEnsembleDetector(subspace_size=3, random_state=0).fit(ref)
    assert a.subspaces_ == b.subspaces_
    assert a.update(new, perf.iloc[:50]) == b.update(new)


# -------------------------------------------------------------- integration
def shift_stream(n_before: int = 300, n_after: int = 300) -> InstanceStream:
    """Covariate shift in one feature block; the best algorithm changes with it."""
    x = pd.concat(
        [sample(n_before, 5), sample(n_after, 6, shift=BLOCK0)], ignore_index=True
    )
    perf = np.where(
        (np.arange(len(x)) < n_before)[:, None], [[5, 50, 20]], [[50, 5, 20]]
    ).astype(float)
    idx = pd.Index([f"i{k}" for k in range(len(x))])
    x.index = idx
    return InstanceStream(
        x,
        pd.DataFrame(perf, index=idx, columns=["a", "b", "c"]),
        cutoff=1000.0,
        drift_points=(n_before,),
    )


def test_integration_with_online_pairwise_selector() -> None:
    stream = shift_stream()
    kw = dict(reference_period=80, random_state=0)
    detector = KSSubspaceEnsembleDetector(
        subspace_size=3, window_size=25, random_state=0
    )
    adaptive = run_prequential(OnlinePairwiseSelector(detector=detector, **kw), stream)
    static = run_prequential(OnlinePairwiseSelector(**kw), stream)

    summary = adaptive.summary()
    assert summary["n_missed"] == 0 and summary["mean_delay"] <= 2 * 25
    assert len(adaptive.retrain_steps) >= 1
    assert sum(step < 300 for step in adaptive.drift_steps) <= 1  # false alarms
    after = slice(400, None)  # drift at position 300
    a, s = (
        adaptive.steps["user_cost"].iloc[after],
        static.steps["user_cost"].iloc[after],
    )
    assert a.mean() < s.mean()
