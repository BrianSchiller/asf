"""A4: bandits fill the prediction record, so extra-run gates work on them."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from asf.online.drift.statistics import LogRuntimeResidual
from asf.online.evaluation import run_prequential
from asf.online.gate import AlwaysProbe, NeverProbe, UncertaintyGate
from asf.online.selectors.bandits import (
    AbstractContextualBandit,
    FDSWTSSelector,
    LinTSSelector,
    LinUCBSelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from asf.online.selectors.bandits.policies import UCBPolicy
from asf.online.selectors.bandits.posteriors import GaussianPosterior
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.weighting import ExponentialDiscount, LastPulls

from .test_bandits import ALGOS, CUTOFF, _obs, _stream, region_stream

STREAM = region_stream(n=200)
ROW = STREAM.features.iloc[[0]]


def _with_arms(
    sel: AbstractContextualBandit, arms: dict[str, tuple[float, float]]
) -> AbstractContextualBandit:
    """Cold start, then give each arm a Gaussian posterior ``(mean, width)``.

    A width of ``inf`` leaves the arm without data; 0 is a (nearly) exact value.
    """
    sel.fit(STREAM.features.iloc[:0], STREAM.performance.iloc[:0])
    for arm, (mean, width) in arms.items():
        post = sel._cell_posteriors(0, "main")[arm]
        post.precision = 1e30 if width == 0 else post.noise_var / width**2
        post.weighted_sum = mean * post.precision
    return sel


def _ucb(**kwargs: Any) -> UCBSelector:
    """UCB with interval ``mean +- width`` (no ln t factor), always gated."""
    kwargs.setdefault("gate", AlwaysProbe(1))
    return UCBSelector(alpha=1.0, ucb1=False, budget=CUTOFF, random_state=0, **kwargs)


def _ts(**kwargs: Any) -> ThompsonSamplingSelector:
    kwargs.setdefault("gate", AlwaysProbe(1))
    return ThompsonSamplingSelector(
        prior_precision=0.0, budget=CUTOFF, random_state=0, **kwargs
    )


# ---------------------------------------------------------------- UCB
def test_ucb_uncertainty_is_interval_overlap_with_the_runner_up() -> None:
    # a: [0.75, 0.85], b: [0.16, 0.84], c far below -> overlap 0.09, union 0.69
    sel = _with_arms(_ucb(), {"a": (0.8, 0.05), "b": (0.5, 0.34), "c": (-5, 0.05)})
    record = sel._select(ROW)
    assert record.chosen == "a"
    assert record.candidates == ("b", "c")  # runner-up first, by UCB index
    assert record.uncertainty == pytest.approx(0.09 / 0.69)
    assert sel.propose(ROW) == [("a", 0.0), ("b", 0.0)]


@pytest.mark.parametrize(
    "rival, expected",
    [
        ((0.0, 0.1), 0.0),  # disjoint: [-0.1, 0.1] vs [0.75, 0.85]
        ((0.8, 0.05), 1.0),  # identical intervals
        ((0.84, 0.0), 0.0),  # a point inside the chosen interval: no length
        ((0.0, np.inf), 1.0),  # rival without data
    ],
)
def test_ucb_uncertainty_edge_cases(
    rival: tuple[float, float], expected: float
) -> None:
    sel = _with_arms(_ucb(), {"a": (0.8, 0.05), "b": rival, "c": (-5, 0.05)})
    assert sel._select(ROW).uncertainty == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------- Thompson
def test_thompson_uncertainty_is_share_of_draws_the_chosen_arm_loses() -> None:
    arms = {"a": (0.0, 0.1), "b": (0.0, 0.1), "c": (-100.0, 0.1)}
    sel = _with_arms(_ts(uncertainty_samples=4000), arms)
    assert sel._select(ROW).uncertainty == pytest.approx(0.5, abs=0.03)

    arms = {"a": (0.0, 0.1), "b": (-100.0, 0.1), "c": (-100.0, 0.1)}
    record = _with_arms(_ts(), arms)._select(ROW)
    assert record.chosen == "a" and record.uncertainty == 0.0


def test_thompson_uncertainty_without_data_shares_ties() -> None:
    record = _with_arms(_ts(), {})._select(ROW)  # every draw is inf for all arms
    assert record.uncertainty == pytest.approx(2 / 3)
    assert record.predicted_runtime is None


def test_thompson_uncertainty_matches_a_manual_count() -> None:
    arms = {"a": (0.0, 1.0), "b": (0.2, 1.0), "c": (-0.5, 2.0)}
    n = 50
    record = _with_arms(_ts(uncertainty_samples=n), arms)._select(ROW)

    rng = np.random.default_rng(0)
    decision = [m + w * rng.standard_normal() for m, w in arms.values()]
    chosen = int(rng.choice(np.flatnonzero(decision == np.max(decision))))
    draws = np.column_stack([m + w * rng.standard_normal(n) for m, w in arms.values()])
    assert record.chosen == ALGOS[chosen]
    assert record.candidates[0] == ALGOS[int(np.argsort(decision)[-2])]
    assert record.uncertainty == pytest.approx(np.mean(draws.argmax(axis=1) != chosen))


def test_uncertainty_samples_zero_and_validation() -> None:
    arms = {"a": (0.0, 1.0), "b": (0.2, 1.0), "c": (-0.5, 2.0)}
    assert _with_arms(_ts(uncertainty_samples=0), arms)._select(ROW).uncertainty is None
    with pytest.raises(ValueError):
        _ts(uncertainty_samples=-1)
    with pytest.raises(ValueError):
        _ts(primary="best")


# ---------------------------------------------------------------- default
def test_ungated_bandit_is_unchanged_and_draws_nothing_extra() -> None:
    arms = {"a": (0.0, 1.0), "b": (0.2, 1.0), "c": (-0.5, 2.0)}
    sel = _with_arms(_ts(gate=None), arms)
    assert isinstance(sel.gate_, NeverProbe)
    before = np.random.default_rng(0)
    [before.standard_normal() for _ in ALGOS]
    before.choice(np.array([0]))
    record = sel._select(ROW)
    assert record.uncertainty is None and len(record.candidates) == 2
    # only the three decision draws and the tie-break were consumed
    assert sel.rng_.standard_normal() == before.standard_normal()

    runs = [
        run_prequential(
            ThompsonSamplingSelector(
                budget=CUTOFF, random_state=0, uncertainty_samples=m
            ),
            STREAM,
        ).steps
        for m in (0, 32)
    ]
    assert (runs[0]["n_runs"] == 1).all()
    assert runs[0]["primary"].equals(runs[1]["primary"])
    assert runs[1]["pred_uncertainty"].isna().all()


# ---------------------------------------------------------------- gates
def test_uncertainty_gate_fires_only_when_intervals_overlap() -> None:
    for rival, extras in [((0.8, 0.05), 1), ((0.0, 0.1), 0)]:
        sel = _ucb(gate=UncertaintyGate(threshold=0.5))
        _with_arms(sel, {"a": (0.8, 0.05), "b": rival, "c": (-5, 0.05)})
        assert len(sel.propose(ROW)) == 1 + extras


@pytest.mark.parametrize("make", [_ucb, _ts])
def test_bandit_learns_from_the_extra_arm(
    make: Callable[[], AbstractContextualBandit],
) -> None:
    sel = make()
    sel.fit(STREAM.features.iloc[:0], STREAM.performance.iloc[:0])
    schedule = [a for a, _ in sel.propose(ROW)]
    assert len(schedule) == 2
    sel.observe(_obs(STREAM, 0, schedule))
    assert sel.clock_ == 1  # one instance, not two
    for arm in ALGOS:
        assert len(sel.history_.get(arm).rewards) == (arm in schedule)


# ---------------------------------------------------------------- greedy
def test_greedy_primary_exploits_and_nominates_the_policy_arm() -> None:
    # UCB index: a 0.85, b 1.0, c 0.05 -> the policy plays b, greedy plays a
    arms = {"a": (0.8, 0.05), "b": (0.5, 0.5), "c": (0.0, 0.05)}
    assert _with_arms(_ucb(), arms)._select(ROW).chosen == "b"
    record = _with_arms(_ucb(primary="greedy"), arms)._select(ROW)
    assert record.chosen == "a"
    assert record.candidates == ("b", "c")
    assert record.scores == pytest.approx({"a": 0.8, "b": 0.5, "c": 0.0})  # means
    assert record.uncertainty == pytest.approx(0.1 / 1.0)  # [.75,.85] in [0,1]

    # policy arm == greedy arm: the extra is the policy's runner-up
    arms["b"] = (0.5, 0.34)
    record = _with_arms(_ucb(primary="greedy"), arms)._select(ROW)
    assert record.chosen == "a" and record.candidates == ("b", "c")

    # arms without data come first
    arms["c"] = (0.0, np.inf)
    assert _with_arms(_ucb(primary="greedy"), arms)._select(ROW).chosen == "c"


# ---------------------------------------------------------------- runtime
@pytest.mark.parametrize("kind", ["log", "par", "unit"])
def test_to_performance_inverts_the_runtime_rewards(kind: str) -> None:
    reward = RewardTransform(kind)
    for runtime, censored in [(37.0, ()), (100.0, ("a",))]:  # solved / PAR10
        r = reward({"a": runtime}, censored, 100.0, False)["a"]
        expected = 1000.0 if censored else runtime
        assert reward.to_performance(r, 100.0, False) == pytest.approx(expected)


def test_to_performance_other_kinds() -> None:
    raw = RewardTransform("raw")
    assert raw.to_performance(raw({"a": 3.0}, (), 9.0, True)["a"], 9.0, True) == 3.0
    assert raw.to_performance(raw({"a": 3.0}, (), 9.0, False)["a"], 9.0, False) == 3.0
    assert RewardTransform("solved").to_performance(1.0, 100.0, False) is None
    assert RewardTransform("par").to_performance(-0.5, float("inf"), False) is None
    assert RewardTransform("log").to_performance(0.5, 100.0, False) == 0.0  # clipped


def test_predicted_runtime_is_the_inverted_posterior_mean() -> None:
    perf = np.array([[9.0, 999.0, 999.0], [99.0, 999.0, 999.0], [1.0, 1.0, 1.0]])
    stream = _stream(np.zeros((3, 2)), perf)
    sel = UCBSelector(budget=CUTOFF, random_state=0)
    sel.fit(stream.features.iloc[:0], stream.performance.iloc[:0])
    assert sel._select(stream.features.iloc[[0]]).predicted_runtime is None
    for i in range(2):
        sel.observe(_obs(stream, i, ALGOS))
    sel.propose(stream.features.iloc[[2]])
    record = sel.last_prediction_
    # rewards of a: -log10(10) = -1 and -log10(100) = -2 -> mean -1.5
    assert record.chosen == "a"
    assert record.predicted_runtime == pytest.approx(10**1.5 - 1)

    sample = LogRuntimeResidual()(_obs(stream, 2, ["a"]), record)  # B4's signal
    assert sample is not None
    assert sample.value == pytest.approx(np.log(1.0) - np.log(10**1.5 - 1))


# ---------------------------------------------------------------- end to end
def _two_trace_ucb(**kwargs: Any) -> AbstractContextualBandit:
    return AbstractContextualBandit(
        posterior=partial(GaussianPosterior, prior_precision=1.0),
        policy=UCBPolicy(),
        contextual=False,
        traces={"historic": ExponentialDiscount(0.99), "hot": LastPulls(30)},
        **kwargs,
    )


@pytest.mark.parametrize(
    "cls",
    [
        ThompsonSamplingSelector,
        UCBSelector,
        LinTSSelector,
        LinUCBSelector,
        FDSWTSSelector,
        _two_trace_ucb,
    ],
)
@pytest.mark.parametrize("n_warm", [0, 50])
def test_gated_bandits_run_and_are_deterministic(cls: Any, n_warm: int) -> None:
    def run() -> pd.DataFrame:
        gate = UncertaintyGate(rate=0.3, random_state=0)
        sel = cls(budget=CUTOFF, gate=gate, random_state=0)
        return run_prequential(sel, STREAM, n_warmstart=n_warm).steps

    steps = run()
    pd.testing.assert_frame_equal(
        steps.drop(columns=["selector_seconds", "detector_seconds"]),
        run().drop(columns=["selector_seconds", "detector_seconds"]),
    )
    assert steps["pred_uncertainty"].between(0, 1).all()
    assert set(steps["n_runs"]) == {1, 2}


@pytest.mark.parametrize("cls", [ThompsonSamplingSelector, LinUCBSelector])
def test_greedy_primary_with_exploring_extra_runs_end_to_end(cls: Any) -> None:
    sel = cls(budget=CUTOFF, primary="greedy", gate=AlwaysProbe(1), random_state=0)
    result = run_prequential(sel, STREAM, result_mode="best_of_scheduled")
    assert (result.steps["n_runs"] == 2).all()
    assert result.steps["pred_runtime"].notna().mean() > 0.9
