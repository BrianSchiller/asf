"""ADWIN and EDDM on scalar streams, also with censored values (B3)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.drift import (
    ADWINDetector,
    DriftSignal,
    EDDMDetector,
    LogRuntimeResidual,
)
from asf.online.drift.abstract_detector import AbstractScalarDriftDetector
from asf.online.evaluation import run_prequential
from asf.online.imputation import (
    CensoredImputer,
    CutoffImputer,
    DropCensored,
    ExpectedImputer,
    SampledImputer,
)
from asf.online.prediction import PredictionRecord
from asf.online.retraining import ResetResponse
from asf.online.streams.stream import InstanceStream

WIDE = (-10.0, 10.0)


# ---------------------------------------------------------------- helpers
def drift_steps(detector: AbstractScalarDriftDetector, values: Any) -> list[int]:
    return [i for i, v in enumerate(values) if detector.update(float(v)).drift]


def reference_adwin(values: Any, delta: float, bound: str) -> tuple[list[int], int]:
    """Figure 1 of the paper with plain loops: drift steps and final width."""
    window: list[float] = []
    drifts = []
    for t, x in enumerate(values):
        window.append(float(x))
        shrunk = False
        while len(window) >= 2 and _some_split_differs(window, delta, bound):
            window.pop(0)
            shrunk = True
        if shrunk:
            drifts.append(t)
    return drifts, len(window)


def _some_split_differs(window: list[float], delta: float, bound: str) -> bool:
    n = len(window)
    for n0 in range(1, n):
        n1 = n - n0
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        gap = abs(sum(window[n0:]) / n1 - sum(window[:n0]) / n0)
        if bound == "hoeffding":
            eps = math.sqrt(math.log(4.0 / (delta / n)) / (2.0 * m))
        else:
            log_term = math.log(2.0 / (delta / math.log(n)))
            eps = math.sqrt(2.0 / m * float(np.var(window)) * log_term)
            eps += 2.0 / (3.0 * m) * log_term
        if gap >= eps:
            return True
    return False


def bernoulli(rng: np.random.Generator, p: float, n: int) -> np.ndarray:
    return (rng.random(n) < p).astype(float)


# ------------------------------------------------------------------ ADWIN
def test_adwin_hand_computed_hoeffding_case() -> None:
    # 8 zeros, then ones; delta = 0.05. Split "8 zeros | j ones": the gap is 1,
    # eps = sqrt(ln(4 n / delta) / (2 m)), n = 8 + j, m = 8 j / (8 + j).
    #   j = 6: sqrt(ln(1120) / 6.857) = 1.012 > 1   -> no cut
    #   j = 7: sqrt(ln(1200) / 7.467) = 0.974 <= 1  -> cut at step 14
    # After dropping one zero (7 | 7): sqrt(ln(1120) / 7) = 1.0015 > 1 -> stop.
    detector = ADWINDetector(delta=0.05, bound="hoeffding")
    signals = [detector.update(x) for x in [0.0] * 8 + [1.0] * 7]
    assert [s.drift for s in signals] == [False] * 14 + [True]
    assert not any(s.warning for s in signals)
    assert detector.width_ == 14
    assert signals[-1].score == pytest.approx(0.5)
    assert signals[7].score == 0.0 and signals[8].score == pytest.approx(1 / 9)


@pytest.mark.parametrize("bound", ["variance", "hoeffding"])
def test_adwin_equals_figure_1(bound: str) -> None:
    seed = 0
    rng = np.random.default_rng(seed)
    values = np.concatenate(
        [bernoulli(rng, 0.2, 120), bernoulli(rng, 0.8, 80), rng.random(60) * 0.3]
    )
    detector = ADWINDetector(delta=0.05, bound=bound)
    drifts = drift_steps(detector, values)
    expected, width = reference_adwin(values, 0.05, bound)
    assert drifts == expected and len(drifts) >= 2
    assert detector.width_ == width


def test_adwin_abrupt_change_as_in_figure_2() -> None:
    # mu jumps from 0.8 to 0.4 at t = 1000 (Figure 2 of the paper)
    rng = np.random.default_rng(0)
    values = np.concatenate([bernoulli(rng, 0.8, 1000), bernoulli(rng, 0.4, 1000)])
    detector = ADWINDetector()
    drifts = drift_steps(detector, values)
    assert drifts and 1000 <= drifts[0] < 1100
    # the window ends up holding (almost) only values after the change
    assert 800 < detector.width_ <= 1100
    assert detector.estimate_ == pytest.approx(0.4, abs=0.08)


def test_adwin_stationary_stream_keeps_growing() -> None:
    values = bernoulli(np.random.default_rng(1), 0.3, 3000)
    detector = ADWINDetector(delta=0.002)
    assert drift_steps(detector, values) == []
    assert detector.width_ == 3000
    assert detector.estimate_ == pytest.approx(values.mean())
    constant = ADWINDetector(delta=0.5)
    assert drift_steps(constant, [0.7] * 300) == []


def test_adwin_direction_up_only_signals_increases() -> None:
    rng = np.random.default_rng(0)
    low, high = bernoulli(rng, 0.2, 500), bernoulli(rng, 0.7, 500)
    falling = ADWINDetector(direction="up")
    assert drift_steps(falling, np.concatenate([high, low])) == []
    assert falling.width_ < 700  # the window still follows the change
    rising = ADWINDetector(direction="up")
    both = ADWINDetector()
    values = np.concatenate([low, high])
    assert drift_steps(rising, values) == drift_steps(both, values) != []


def test_adwin_value_range_rescales_and_clips() -> None:
    rng = np.random.default_rng(3)
    unit = np.concatenate([rng.random(200) * 0.4, 0.5 + rng.random(200) * 0.5])
    a, b = ADWINDetector(), ADWINDetector(value_range=(-2.0, 6.0))
    for x in unit:
        sa, sb = a.update(x), b.update(-2.0 + 8.0 * x)
        assert sa.drift == sb.drift
        assert sb.score == pytest.approx(-2.0 + 8.0 * sa.score)
    clipped = ADWINDetector()
    assert clipped.update(5.0).score == 1.0
    assert clipped.update(-3.0).score == 0.5


def test_adwin_reset_forgets_the_window() -> None:
    detector = ADWINDetector(delta=0.05, bound="hoeffding")
    first = [detector.update(x).drift for x in [0.0] * 8 + [1.0] * 7]
    detector.reset()
    assert detector.width_ == 0 and detector.estimate_ is None
    assert [detector.update(x).drift for x in [0.0] * 8 + [1.0] * 7] == first


# ------------------------------------------------------------------- EDDM
def eddm_signals(detector: EDDMDetector, error_positions: set[int], n: int) -> list:
    """Signals for a 0/1 stream with errors at the given 1-based positions."""
    return [detector.update(float(k in error_positions)) for k in range(1, n + 1)]


def test_eddm_hand_computed_case() -> None:
    # Distances between errors: 10, 10, 10, 2, 2, 2, 2 (sample sd, level p + 2 s)
    #   10, 10, 10       p = 10    s = 0      level 10      max 10
    #   + 2              p = 8     s = 4      level 16      max 16     ratio 1
    #   + 2              p = 6.8   s = 4.382  level 15.564             ratio 0.9727
    #   + 2              p = 6     s = 4.382  level 14.764             ratio 0.9227
    #   + 2              p = 5.429 s = 4.276  level 13.981             ratio 0.8738
    signals = eddm_signals(EDDMDetector(min_errors=3), {10, 20, 30, 32, 34, 36, 38}, 38)
    assert signals[8].score is None  # no error yet
    scores = [signals[k - 1].score for k in (10, 20, 30, 32, 34, 36, 38)]
    assert scores == pytest.approx(
        [1, 1, 1, 1, (6.8 + 2 * math.sqrt(19.2)) / 16, (6 + 2 * math.sqrt(19.2)) / 16]
        + [(38 / 7 + 2 * math.sqrt(768 / 42)) / 16]
    )
    assert [s.warning for s in signals] == [False] * 35 + [True, True, False]
    assert [s.drift for s in signals] == [False] * 37 + [True]
    assert signals[36].score == signals[35].score  # held between errors


def test_eddm_waits_for_min_errors() -> None:
    errors = {10, 20, 30, 32, 34, 36, 38}
    late = eddm_signals(EDDMDetector(min_errors=8), errors, 38)
    assert not any(s.drift for s in late)
    last = eddm_signals(EDDMDetector(min_errors=7), errors, 38)[-1]
    assert last.drift and not last.warning
    # with the paper's 30 errors nothing can be tested on this short stream
    quiet = eddm_signals(EDDMDetector(), errors, 38)
    assert not any(s.drift or s.warning for s in quiet)
    assert quiet[-1].score == pytest.approx(0.8738081981)


def test_eddm_warning_ends_when_errors_spread_out_again() -> None:
    detector = EDDMDetector(min_errors=3)
    signals = eddm_signals(detector, {10, 20, 30, 32, 34, 36, 66}, 66)
    assert signals[35].warning and signals[64].warning
    # distance 30: p = 66 / 7, new maximum of p + 2 s -> ratio 1, back to normal
    assert signals[65].score == 1.0 and not signals[65].warning


def test_eddm_restarts_after_a_drift() -> None:
    errors = {10, 20, 30, 32, 34, 36, 38}
    detector = EDDMDetector(min_errors=3)
    first = eddm_signals(detector, errors, 38)
    assert first[-1].drift
    assert eddm_signals(detector, errors, 38) == first  # as if freshly built
    detector.update(1.0)
    detector.reset()
    assert eddm_signals(detector, errors, 38) == first


def test_eddm_regular_errors_never_drift() -> None:
    detector = EDDMDetector(min_errors=3)
    signals = eddm_signals(detector, set(range(5, 1001, 5)), 1000)
    assert not any(s.drift or s.warning for s in signals)
    assert signals[-1].score == 1.0


def test_eddm_detects_errors_getting_closer() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([bernoulli(rng, 0.05, 2000), bernoulli(rng, 0.4, 500)])
    drifts = drift_steps(EDDMDetector(), values)
    assert any(2000 <= t < 2300 for t in drifts)


def test_eddm_error_threshold_on_real_values() -> None:
    # default: more than twice as slow as predicted (log residual > log 2)
    twice = math.log(2.0)
    detector = EDDMDetector(min_errors=1)
    assert detector.update(twice).score is None  # not above the threshold
    assert detector.update(twice + 1e-9).score == 1.0
    assert detector.update(-5.0).score == 1.0  # much faster: no error
    strict = EDDMDetector(error_threshold=0.0, min_errors=1)
    assert strict.update(0.1).score == 1.0


# -------------------------------------------------------- censored values
def exact_then_bound(imputation: CensoredImputer | None, bound: float) -> float:
    """Value used for a bound after the exact values 0, 1, 2, 3."""
    detector = ADWINDetector(delta=1e-9, value_range=WIDE, imputation=imputation)
    for x in (0.0, 1.0, 2.0, 3.0):
        detector.update(x)
    score = detector.update(bound, censored=True).score
    assert detector.width_ == 5 and score is not None
    return 5.0 * score - 6.0


def test_default_and_expected_rules_on_a_censored_bound() -> None:
    assert exact_then_bound(None, 2.0) == pytest.approx(2.0)
    assert exact_then_bound(CutoffImputer(), 2.0) == pytest.approx(2.0)
    # belief: mean 1.5 and sample sd sqrt(5 / 3) of the exact values
    sd = math.sqrt(5.0 / 3.0)
    a = (2.0 - 1.5) / sd
    expected = 1.5 + sd * norm.pdf(a) / norm.sf(a)
    assert exact_then_bound(ExpectedImputer(), 2.0) == pytest.approx(expected)
    assert expected > 2.0
    # a bound far below the belief changes little: E[y | y >= -50] = mean
    assert exact_then_bound(ExpectedImputer(), -50.0) == pytest.approx(1.5)


@pytest.mark.parametrize("rule", [ExpectedImputer(), SampledImputer()])
def test_imputing_rules_fall_back_to_the_bound_without_a_belief(
    rule: CensoredImputer,
) -> None:
    detector = ADWINDetector(value_range=WIDE, imputation=rule, random_state=0)
    assert detector.update(3.0, censored=True).score == pytest.approx(3.0)
    one_exact = ADWINDetector(value_range=WIDE, imputation=rule, random_state=0)
    one_exact.update(0.0)
    assert one_exact.update(3.0, censored=True).score == pytest.approx(1.5)


def test_drop_rule_skips_the_instance() -> None:
    detector = ADWINDetector(value_range=WIDE, imputation=DropCensored())
    assert detector.update(4.0, censored=True) == detector.update(9.0, censored=True)
    assert detector.width_ == 0
    detector.update(1.0)
    signal = detector.update(9.0, censored=True)
    assert detector.width_ == 1 and not signal.drift
    assert signal.score == pytest.approx(1.0)

    eddm = EDDMDetector(min_errors=3, imputation=DropCensored())
    errors = {10, 20, 30, 32, 34, 36, 38}
    signals = []
    for k in range(1, 39):
        signals.append(eddm.update(float(k in errors)))
        dropped = eddm.update(9.0, censored=True)  # would be an error
        if not signals[-1].drift:  # (after a drift the detector starts anew)
            assert dropped == DriftSignal(False, signals[-1].warning, signals[-1].score)
    assert signals == eddm_signals(EDDMDetector(min_errors=3), errors, 38)


def test_sampled_rule_is_seeded_and_stays_above_the_bound() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=300)
    censored = rng.random(300) < 0.4

    def run(seed: int) -> list[float]:
        detector = ADWINDetector(
            delta=1e-9, value_range=WIDE, imputation=SampledImputer(), random_state=seed
        )
        return [detector.update(v, c).score for v, c in zip(values, censored)]

    assert run(0) == run(0)
    assert run(0) != run(1)

    # one detector, restarted: the same draws again
    detector = ADWINDetector(
        delta=1e-9, value_range=WIDE, imputation=SampledImputer(), random_state=0
    )
    first = [detector.update(v, c).score for v, c in zip(values, censored)]
    detector.reset()
    assert [detector.update(v, c).score for v, c in zip(values, censored)] == first

    bound_only = ADWINDetector(delta=1e-9, value_range=WIDE)
    lower = [bound_only.update(v, c).score for v, c in zip(values, censored)]
    assert all(s >= b - 1e-12 for s, b in zip(first, lower))
    assert first[-1] > lower[-1]


def test_eddm_applies_the_threshold_after_imputation() -> None:
    # bound rule: a timeout is an error only if its bound is above the threshold
    plain = EDDMDetector(min_errors=1)
    assert plain.update(0.5, censored=True).score is None
    assert plain.update(0.8, censored=True).score == 1.0
    # exact values -3 and 0.6 are no errors, but their spread is large: the
    # conditional mean above a bound of 0.5 is about 1.5, an error
    imputing = EDDMDetector(min_errors=1, imputation=ExpectedImputer())
    bound_rule = EDDMDetector(min_errors=1)
    for x in [-3.0, 0.6] * 5:
        assert imputing.update(x).score is None
        bound_rule.update(x)
    assert bound_rule.update(0.5, censored=True).score is None
    assert imputing.update(0.5, censored=True).score == 1.0


def test_invalid_arguments() -> None:
    for kwargs in (
        {"delta": 0.0},
        {"delta": 1.0},
        {"value_range": (1.0, 1.0)},
        {"value_range": (0.0, math.inf)},
        {"bound": "normal"},
        {"direction": "down"},
    ):
        with pytest.raises(ValueError):
            ADWINDetector(**kwargs)  # type: ignore[arg-type]
    for kwargs in (
        {"alpha": 0.8, "beta": 0.9},
        {"beta": 0.0},
        {"alpha": 1.1},
        {"min_errors": 0},
        {"error_threshold": math.nan},
    ):
        with pytest.raises(ValueError):
            EDDMDetector(**kwargs)  # type: ignore[arg-type]
    for cls in (ADWINDetector, EDDMDetector):
        with pytest.raises(TypeError):
            cls(imputation="cutoff")  # type: ignore[arg-type]
        for bad in (math.nan, math.inf):
            with pytest.raises(ValueError, match="finite"):
                cls().update(bad)


# ------------------------------------------------- with a selector (runtime)
class _Predicting(AbstractOnlineSelector):
    """Always picks ``a`` and predicts a runtime of 5 for it."""

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kw: Any
    ) -> None:
        self.n_resets = 0

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        return PredictionRecord(chosen="a", predicted_runtime=5.0)

    def _observe(self, observation: Observation) -> None:
        pass

    def _reset_model(self) -> None:
        self.n_resets += 1


JUMP = 300  # position in the stream where ``a`` gets 4 times slower
N_WARM = 10
RESIDUALS = (-math.log(10.0), math.log(10.0))


def runtime_stream(cutoff: float, n: int = 500, seed: int = 0) -> InstanceStream:
    """``a`` takes about 5 s (as predicted) before ``JUMP`` and about 20 s after."""
    rng = np.random.default_rng(seed)
    a = np.where(np.arange(n) < JUMP, 5.0, 20.0) * np.exp(rng.normal(0, 0.1, n))
    index = pd.Index([f"i{k}" for k in range(n)])
    return InstanceStream(
        pd.DataFrame({"f0": rng.normal(size=n)}, index=index),
        pd.DataFrame({"a": a, "b": 50.0}, index=index),
        cutoff=cutoff,
        drift_points=(JUMP,),
        name="jump",
    )


def detected(detector: AbstractScalarDriftDetector, cutoff: float) -> list[int]:
    """Stream positions at which the detector signalled drift."""
    selector = _Predicting(detector=detector, drift_statistic=LogRuntimeResidual())
    result = run_prequential(selector, runtime_stream(cutoff), n_warmstart=N_WARM)
    steps = result.steps
    assert steps["drift_signal"].sum() == len(selector.drift_steps)
    return [int(s) + N_WARM for s in steps.index[steps["drift_signal"]]]


@pytest.mark.parametrize(
    "detector",
    [
        ADWINDetector(value_range=RESIDUALS),
        ADWINDetector(value_range=RESIDUALS, direction="up"),
        EDDMDetector(min_errors=5),
    ],
)
def test_runtime_jump_is_detected(detector: AbstractScalarDriftDetector) -> None:
    drifts = detected(detector, cutoff=100.0)
    assert drifts and JUMP <= drifts[0] < JUMP + 40


@pytest.mark.parametrize(
    "rule", [None, ExpectedImputer(), SampledImputer(), DropCensored()]
)
def test_runtime_jump_into_timeouts(rule: CensoredImputer | None) -> None:
    # cutoff 10: after the jump every run times out, the residual is the
    # bound log(10 / 5) instead of about log(20 / 5)
    stream = runtime_stream(cutoff=10.0)
    assert (stream.performance["a"].iloc[JUMP:] > 10.0).all()
    adwin = ADWINDetector(value_range=RESIDUALS, imputation=rule, random_state=0)
    drifts = detected(adwin, cutoff=10.0)
    if isinstance(rule, DropCensored):
        assert drifts == []  # the detector never sees the slow runs
        return
    assert drifts and JUMP <= drifts[0] < JUMP + 80
    bound_rule = detected(ADWINDetector(value_range=RESIDUALS), cutoff=10.0)
    assert drifts[0] <= bound_rule[0]


def test_eddm_counts_timeouts_above_the_threshold_as_errors() -> None:
    # the bound log(10 / 5) = log 2 is not above a threshold of log 3 ...
    slow = EDDMDetector(error_threshold=math.log(3.0), min_errors=5)
    assert detected(slow, cutoff=10.0) == []
    # ... but above log 1.5
    drifts = detected(
        EDDMDetector(error_threshold=math.log(1.5), min_errors=5), cutoff=10.0
    )
    assert drifts and JUMP <= drifts[0] < JUMP + 40


def test_detector_restarts_when_the_model_is_reset() -> None:
    detector = ADWINDetector(value_range=RESIDUALS)
    selector = _Predicting(
        detector=detector,
        drift_statistic=LogRuntimeResidual(),
        drift_response=ResetResponse(),
    )
    stream = runtime_stream(cutoff=100.0)
    steps = run_prequential(selector, stream, n_warmstart=N_WARM).steps
    drifts = list(steps.index[steps["drift_signal"]])
    assert len(drifts) == selector.n_resets == 1
    assert list(steps["response"][steps["drift_signal"]]) == ["reset"]
    # the window was emptied at the drift step and grew by one per step since
    assert detector.width_ == len(steps) - 1 - drifts[0]
