"""Tests for the one-class drift detector (OCDD, Gözüaçık and Can 2021)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from asf.online.drift import DriftSignal, OneClassDriftDetector
from asf.online.evaluation import run_prequential
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams.stream import InstanceStream

D = 5
NAMES = [f"f{i}" for i in range(D)]


def sample(n: int, seed: int, shift: float = 0.0, d: int = D) -> pd.DataFrame:
    """Independent standard normal features, optionally shifted."""
    x = np.random.default_rng(seed).normal(size=(n, d)) + shift
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])


def feed(det: OneClassDriftDetector, data: pd.DataFrame) -> list[int]:
    """Positions in ``data`` at which the detector signals drift."""
    return [i for i in range(len(data)) if det.update(data.iloc[[i]]).drift]


# ------------------------------------------- Algorithm 1 with known labels
class RuleOCDD(OneClassDriftDetector):
    """
    OCDD whose one-class model is a fixed rule, so labels can be told by hand:
    an instance is an outlier iff it is further than 10 from the mean of the
    data the model was last trained on. ``expected`` replaces the
    cross-validated baseline share.
    """

    expected: float | None = None

    def _train(self, X: np.ndarray) -> None:
        self.center = float(X[:, 0].mean())
        self.trained_on = X[:, 0].tolist()
        super()._train(X)

    def _outliers(self, X: np.ndarray) -> np.ndarray:
        return np.abs(X[:, 0] - self.center) > 10

    def _expected_outlier_share(self, X: np.ndarray) -> float:
        return (
            super()._expected_outlier_share(X)
            if self.expected is None
            else (self.expected)
        )


def values(*xs: float) -> pd.DataFrame:
    return pd.DataFrame({"x": [float(x) for x in xs]})


def rule_detector(**kwargs: object) -> RuleOCDD:
    """w = 10, rho = 0.3 (keep 3); trained on values around 0."""
    det = RuleOCDD(window_size=10, threshold=0.3, **kwargs)  # type: ignore[arg-type]
    return det.fit(values(-1, 0, 1, 0))


def signals(det: OneClassDriftDetector, *xs: float) -> list[DriftSignal]:
    return [det.update(values(x)) for x in xs]


def test_drift_when_outlier_share_reaches_threshold() -> None:
    det = rule_detector()
    out = signals(det, 0, 0, 0, 0, 0, 0, 0, 50, 51, 52)
    # No check before the window is full; then 3 of 10 outliers = rho: drift.
    assert all(s == DriftSignal(drift=False, score=None) for s in out[:9])
    assert out[9] == DriftSignal(drift=True, score=0.3)
    assert det.last_outlier_share_ == 0.3 and det.n_retrains_ == 1


def test_no_drift_below_threshold_then_window_slides_by_one() -> None:
    det = rule_detector()
    out = signals(det, 0, 0, 0, 0, 0, 0, 0, 0, 50, 51)
    assert out[9] == DriftSignal(drift=False, score=0.2)  # 2 of 10
    # The oldest instance left, so the next instance completes a window again.
    assert det.update(values(52)) == DriftSignal(drift=True, score=0.3)


def test_no_check_before_the_window_is_full() -> None:
    det = rule_detector()
    out = signals(det, *[50] * 9)  # 9 outliers, but only 9 instances
    assert all(not s.drift and s.score is None for s in out)
    assert det.last_outlier_share_ is None
    assert det.update(values(50)).drift


def test_outliers_leave_the_window() -> None:
    det = rule_detector()
    shares = [s.score for s in signals(det, 50, 50, *[0] * 10)]
    assert shares[9:] == [0.2, 0.1, 0.0]
    assert det.n_retrains_ == 0


def test_after_drift_newest_instances_are_kept_and_retrained_on() -> None:
    det = rule_detector()
    signals(det, 0, 0, 0, 0, 0, 0, 0, 50, 51, 52)
    # Kept: the newest rho * w = 3 instances; the model now describes them.
    assert det.trained_on == [50.0, 51.0, 52.0] and det.center == 51.0
    # They were outliers for the old model but are typical for the new one,
    # so 7 more typical instances fill the window without a new drift.
    out = signals(det, *[51] * 7)
    assert [s.drift for s in out] == [False] * 7
    assert [s.score for s in out[:6]] == [0.3] * 6  # no check: last result
    assert out[6].score == 0.0
    # The old concept is now the unfamiliar one: checks at every instance.
    back = signals(det, 0, 0, 0)
    assert [s.score for s in back] == [0.1, 0.2, 0.3]
    assert [s.drift for s in back] == [False, False, True]
    assert det.trained_on == [0.0, 0.0, 0.0] and det.n_retrains_ == 2


def test_several_rows_in_one_update() -> None:
    one_by_one, batch = rule_detector(), rule_detector()
    xs = [0, 0, 0, 0, 0, 0, 0, 50, 51, 52, 51, 51]
    last = signals(one_by_one, *xs)[-1]
    assert batch.update(values(*xs)) == DriftSignal(drift=True, score=last.score)
    assert batch.trained_on == one_by_one.trained_on


def test_baseline_shifts_the_threshold() -> None:
    # Expected outlier share 0.2: drift needs 0.2 + 0.3 = 5 of 10 outliers.
    det = rule_detector(baseline="cross_validated")
    det.expected = 0.2
    det.fit(values(-1, 0, 1, 0))
    assert det.baseline_share_ == 0.2
    out = signals(det, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50)
    assert not out[9].drift and out[9].score == pytest.approx(0.2)  # 4 of 10
    assert out[10].drift and out[10].score == pytest.approx(0.3)  # 5 of 10


@pytest.mark.parametrize(
    ("window_size", "threshold", "n_keep"),
    [(250, 0.3, 75), (100, 0.3, 30), (10, 0.25, 3), (10, 1.0, 10), (7, 0.3, 2)],
)
def test_number_of_kept_instances(
    window_size: int, threshold: float, n_keep: int
) -> None:
    assert OneClassDriftDetector(window_size, threshold).n_keep_ == n_keep


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 1},
        {"threshold": 0.0},
        {"threshold": 1.1},
        {"window_size": 10, "threshold": 0.1},  # would keep one instance
        {"nu": 0.0},
        {"nu": 1.5},
        {"baseline": "bootstrap"},
        {"n_folds": 1},
    ],
)
def test_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        OneClassDriftDetector(**kwargs)


def test_defaults_follow_the_paper() -> None:
    det = OneClassDriftDetector()
    assert (det.window_size, det.threshold, det.nu) == (250, 0.3, 0.5)
    assert det.baseline == "none" and det.kernel == "rbf"


# ------------------------------------------------------ one-class SVM model
def test_fit_requirements_and_use_before_fit() -> None:
    det = OneClassDriftDetector()
    with pytest.raises(RuntimeError):
        det.update(sample(1, 0))
    with pytest.raises(RuntimeError):
        det.score(sample(1, 0))
    with pytest.raises(ValueError):
        det.fit(sample(1, 0))
    with pytest.raises(ValueError):
        det.fit(pd.DataFrame(index=range(5)))


def test_score_is_positive_exactly_for_outliers() -> None:
    det = OneClassDriftDetector(nu=0.1).fit(sample(200, 0))
    new = pd.concat([sample(100, 1), sample(100, 2, shift=4.0)], ignore_index=True)
    score = det.score(new)
    assert score.shape == (200,)
    assert ((score > 0) == det._outliers(new.to_numpy())).all()
    # At most about nu of the training data is outside the boundary ...
    assert (det.score(sample(200, 0)) > 0).mean() <= 0.1 + 0.02
    # ... but shifted data is, and further out than unshifted data.
    assert (score[100:] > 0).all()
    assert score[100:].min() > np.median(score[:100])
    centre = pd.DataFrame(np.zeros((1, D)), columns=NAMES)
    assert det.score(centre)[0] < 0


def test_baseline_share() -> None:
    ref = sample(200, 0)
    assert OneClassDriftDetector().fit(ref).baseline_share_ == 0.0
    det = OneClassDriftDetector(nu=0.1, baseline="cross_validated").fit(ref)
    in_sample = float(det._outliers(ref.to_numpy()).mean())
    # New data is called an outlier more often than the training data.
    assert in_sample <= 0.12 < det.baseline_share_ < 0.5
    fresh = float((det.score(sample(2000, 1)) > 0).mean())
    assert abs(det.baseline_share_ - fresh) < 0.1
    # Too few instances for the folds: share on the training data itself.
    small = sample(9, 3)
    det = OneClassDriftDetector(baseline="cross_validated").fit(small)
    assert det.baseline_share_ == float(det._outliers(small.to_numpy()).mean())


def test_paper_settings_signal_without_drift() -> None:
    # nu = 0.5 labels about half of all typical instances as outliers, which
    # is above rho = 0.3: a signal every time the window is full again.
    det = OneClassDriftDetector(window_size=50, threshold=0.3).fit(sample(50, 0))
    hits = feed(det, sample(400, 1))
    assert hits == list(range(49, 400, 50 - det.n_keep_))


def test_baseline_is_quiet_without_drift_and_detects_a_shift() -> None:
    kwargs = dict(window_size=100, threshold=0.3, baseline="cross_validated")
    det = OneClassDriftDetector(**kwargs).fit(sample(100, 0))  # type: ignore[arg-type]
    assert feed(det, sample(1000, 1)) == []

    det = OneClassDriftDetector(**kwargs).fit(sample(100, 0))  # type: ignore[arg-type]
    stream = pd.concat([sample(300, 2), sample(700, 3, shift=2.0)], ignore_index=True)
    hits = feed(det, stream)
    # One signal, within one window after the shift at 300; the detector then
    # describes the new data and stays quiet.
    assert len(hits) == 1 and 300 <= hits[0] < 400
    assert det.n_retrains_ == 1
    assert det.score(sample(50, 4, shift=2.0)).mean() < det.score(sample(50, 4)).mean()


def test_deterministic() -> None:
    stream = pd.concat([sample(200, 2), sample(300, 3, shift=2.0)], ignore_index=True)

    def run() -> list[DriftSignal]:
        det = OneClassDriftDetector(window_size=60, baseline="cross_validated")
        det.fit(sample(100, 0))
        return [det.update(stream.iloc[[i]]) for i in range(len(stream))]

    first = run()
    assert first == run() and any(s.drift for s in first)


def test_fit_restarts_the_detector() -> None:
    det = OneClassDriftDetector(window_size=20).fit(sample(50, 0))
    assert feed(det, sample(40, 1, shift=3.0))
    det.fit(sample(50, 0))
    assert det.n_retrains_ == 0 and det.last_outlier_share_ is None
    out = [det.update(sample(1, i)) for i in range(19)]
    assert all(s.score is None for s in out)  # the window was emptied


def test_reset_does_nothing() -> None:
    det = OneClassDriftDetector(window_size=20, nu=0.1).fit(sample(50, 0))
    twin = OneClassDriftDetector(window_size=20, nu=0.1).fit(sample(50, 0))
    head, tail = sample(15, 1), sample(30, 2, shift=3.0)
    feed(det, head), feed(twin, head)
    det.reset(sample(200, 5, shift=10.0))  # e.g. the selector's buffer
    assert feed(det, tail) == feed(twin, tail)
    probe = sample(20, 6)
    assert (det.score(probe) == twin.score(probe)).all()


def test_missing_values_and_degenerate_features() -> None:
    def messy(n: int, seed: int) -> pd.DataFrame:
        x = sample(n, seed)
        x[x > 1.5] = np.nan  # missing values
        x["constant"] = 1.0
        x["empty"] = np.nan
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        det = OneClassDriftDetector(window_size=30, baseline="cross_validated")
        det.fit(messy(80, 0))
        hits = feed(det, pd.concat([messy(60, 1), messy(60, 2) + 3.0]))
        assert np.isfinite(det.score(messy(10, 3))).all()
    assert hits and hits[0] >= 60
    # A row without any value is imputed with the training medians.
    blank = pd.DataFrame(np.nan, index=[0], columns=messy(1, 0).columns)
    assert np.isfinite(det.score(blank)).all()


def test_columns_are_matched_by_name() -> None:
    det = OneClassDriftDetector().fit(sample(100, 0))
    new = sample(10, 1)
    assert (det.score(new[NAMES[::-1]]) == det.score(new)).all()
    with pytest.raises(ValueError):
        det.score(new.drop(columns="f0"))


def test_performance_is_ignored() -> None:
    ref, new = sample(100, 0), sample(60, 1, shift=3.0)
    perf = pd.DataFrame({"a": np.arange(100.0)})
    a = OneClassDriftDetector(window_size=40).fit(ref, perf)
    b = OneClassDriftDetector(window_size=40).fit(ref)
    assert a.update(new, perf.iloc[:60]) == b.update(new)


# -------------------------------------------------------------- integration
def shift_stream(n_before: int = 300, n_after: int = 300) -> InstanceStream:
    """Covariate shift in all features; the best algorithm changes with it."""
    x = pd.concat(
        [sample(n_before, 5), sample(n_after, 6, shift=2.0)], ignore_index=True
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
    detector = OneClassDriftDetector(window_size=50, baseline="cross_validated")
    adaptive = run_prequential(OnlinePairwiseSelector(detector=detector, **kw), stream)
    static = run_prequential(OnlinePairwiseSelector(**kw), stream)

    assert len(adaptive.drift_steps) == 1 and 300 <= adaptive.drift_steps[0] < 350
    assert adaptive.retrain_steps == adaptive.drift_steps
    assert detector.n_retrains_ == 1
    steps = adaptive.steps
    assert steps["drift_signal"].sum() == 1
    assert steps["detector_score"].notna().sum() > 0
    after = slice(400, None)  # drift at position 300
    a, s = steps["user_cost"].iloc[after], static.steps["user_cost"].iloc[after]
    assert a.mean() < s.mean()
