"""Properties of controlled covariate-only synthetic streams."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asf.online.evaluation import run_prequential
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams import (
    FeatureRegime,
    PerformanceModelConfig,
    StreamPhase,
    make_synthetic_scenario,
    make_synthetic_stream,
)
from asf.online.streams.stream import InstanceStream


def _stream(**kwargs: object) -> InstanceStream:
    settings = dict(
        regimes={"A": FeatureRegime([0.0, 0.0]), "B": FeatureRegime([8.0, 0.0])},
        phases=[StreamPhase("A", 30), StreamPhase("B", 30)],
        performance_model=PerformanceModelConfig(
            [np.log(5.0), np.log(10.0)], [[0.2, -0.2], [0.0, 0.0]]
        ),
        cutoff=30.0,
        random_state=7,
    )
    settings.update(kwargs)
    return make_synthetic_stream(**settings)


def test_reproducibility_and_independent_feature_noise() -> None:
    first, same, other = _stream(), _stream(), _stream(random_state=8)
    pd.testing.assert_frame_equal(first.features, same.features)
    pd.testing.assert_frame_equal(first.performance, same.performance)
    assert not first.features.equals(other.features)
    changed = _stream(performance_noise=1.0)
    pd.testing.assert_frame_equal(first.features, changed.features)
    assert not first.performance.equals(changed.performance)
    assert first.drift_points == (30,)
    assert first.features.index.is_unique
    assert first.features.index.equals(first.performance.index)


def test_global_performance_model_and_custom_names() -> None:
    # Both regimes change features only: the same formula holds on every row.
    model = PerformanceModelConfig([0.0, 1.0], [[0.2, -0.1], [0.3, 0.4]])
    stream = _stream(
        performance_model=model,
        phases=[StreamPhase("A", 10), StreamPhase("B", 10)],
        feature_names=["size", "density"],
        algorithm_names=["solver_a", "solver_b"],
    )
    expected = np.exp(
        np.asarray(model.intercepts)
        + stream.features.to_numpy() @ np.asarray(model.coefficients)
    )
    np.testing.assert_allclose(stream.performance.to_numpy(), expected)
    assert list(stream.features) == ["size", "density"]
    assert list(stream.performance) == ["solver_a", "solver_b"]
    assert stream.features.iloc[10:].mean().iloc[0] > 5
    assert stream.features.iloc[:10].mean().iloc[0] < 3


def test_shared_loading_correlates_features_without_changing_runtime_rule() -> None:
    model = PerformanceModelConfig([0.0, 0.3], [[0.2, -0.3], [0.4, 0.1]])
    stream = _stream(
        phases=[StreamPhase("A", 200)],
        performance_model=model,
        loading=[[1.0, 2.0]],
    )
    np.testing.assert_allclose(
        stream.features.iloc[:, 1], 2 * stream.features.iloc[:, 0]
    )
    np.testing.assert_allclose(
        stream.performance.to_numpy(),
        np.exp(
            np.asarray(model.intercepts)
            + stream.features.to_numpy() @ np.asarray(model.coefficients)
        ),
    )


def test_adjacent_same_regime_and_recurring_draws() -> None:
    stream = _stream(
        phases=[
            StreamPhase("A", 10),
            StreamPhase("A", 10),
            StreamPhase("B", 10),
            StreamPhase("A", 10),
        ]
    )
    assert stream.drift_points == (20, 30)
    assert list(stream.features.index) == [f"inst_{i}" for i in range(40)]
    assert not np.array_equal(stream.features.iloc[:10], stream.features.iloc[30:])
    assert stream.features.iloc[30:].mean().iloc[0] < 3


def test_gradual_transition_uses_whole_regime_draws_and_records_interval() -> None:
    stream = _stream(
        regimes={
            "A": FeatureRegime([0.0, 0.0], scale=0),
            "B": FeatureRegime([8.0, 0.0], scale=0),
        },
        phases=[StreamPhase("A", 6), StreamPhase("B", 8, transition_length=5)],
    )
    assert stream.drift_points == (6,)
    assert stream.transition_intervals == ((6, 11),)
    assert stream.features.iloc[6, 0] == 0
    assert stream.features.iloc[10, 0] == 8
    assert set(stream.features.iloc[6:11, 0]) <= {0.0, 8.0}
    assert (stream.features.iloc[11:, 0] == 8).all()
    _, mid = stream.split(8)
    assert mid.drift_points == ()
    assert mid.transition_intervals == ((0, 3),)
    head, tail = stream.split(7)
    assert head.transition_intervals == ((6, 7),)
    assert tail.transition_intervals == ((0, 4),)


@pytest.mark.parametrize(
    "kind",
    ["stationary", "sudden_covariate", "gradual_covariate", "recurring_covariate"],
)
@pytest.mark.parametrize("dimensions", [(1, 2), (5, 4)])
def test_preset_dimensions_and_breakpoints(
    kind: str, dimensions: tuple[int, int]
) -> None:
    d, m = dimensions
    stream = make_synthetic_scenario(
        kind, n_instances=180, n_features=d, n_algorithms=m, random_state=3
    )
    assert stream.features.shape == (180, d)
    assert stream.performance.shape == (180, m)
    assert len(stream.drift_points) == (
        0 if kind == "stationary" else 2 if kind == "recurring_covariate" else 1
    )
    assert len(stream.transition_intervals) == (1 if kind == "gradual_covariate" else 0)
    assert (stream.performance.to_numpy() > 0).all()


def test_preset_breakpoint_placement_and_recurrence() -> None:
    even = make_synthetic_scenario(
        "recurring_covariate",
        n_instances=180,
        n_breakpoints=2,
        random_state=2,
    )
    assert even.drift_points == (60, 120)
    assert even.transition_intervals == ()
    random_kwargs = dict(
        kind="recurring_covariate",
        n_instances=180,
        n_breakpoints=3,
        min_segment_length=20,
        breakpoint_distribution="uniform",
        random_state=12,
    )
    random_a = make_synthetic_scenario(**random_kwargs)
    random_b = make_synthetic_scenario(**random_kwargs)
    assert random_a.drift_points == random_b.drift_points
    assert min(np.diff([0, *random_a.drift_points, 180])) >= 20
    explicit = make_synthetic_scenario(
        "gradual_covariate",
        n_instances=180,
        n_breakpoints=2,
        breakpoints=[60, 120],
        min_segment_length=40,
        transition_length=30,
        random_state=2,
    )
    assert explicit.transition_intervals == ((60, 90), (120, 150))
    assert explicit.drift_points == (60, 120)
    assert not np.array_equal(even.features.iloc[:60], even.features.iloc[120:])


def test_preset_rejects_unknown_scenario_kind() -> None:
    with pytest.raises(ValueError, match="Unknown synthetic scenario"):
        make_synthetic_scenario("pca_off_subspace")  # type: ignore[arg-type]


def test_additional_algorithms_can_win_with_same_runtime_model() -> None:
    stream = make_synthetic_scenario(
        "stationary",
        n_instances=4000,
        n_algorithms=5,
        min_segment_length=20,
        random_state=4,
    )
    winners = stream.performance.idxmin(axis=1)
    assert set(winners) == set(stream.performance.columns)


def test_automatic_gradual_breakpoints_fit_transition_and_retain_draws() -> None:
    settings = dict(
        kind="gradual_covariate",
        n_instances=450,
        n_breakpoints=2,
        min_segment_length=20,
        transition_length=100,
        random_state=3,
    )
    even = make_synthetic_scenario(**settings)
    uniform = make_synthetic_scenario(**settings, breakpoint_distribution="uniform")
    for stream in (even, uniform):
        lengths = np.diff([0, *stream.drift_points, 450])
        assert min(lengths) >= 100
        assert stream.transition_intervals == tuple(
            (point, point + 100) for point in stream.drift_points
        )
    # Breakpoint placement must not replace the per-row Gaussian draws.
    left = make_synthetic_scenario(**settings, shift_strength=0.5)
    right = make_synthetic_scenario(**settings, shift_strength=4)
    np.testing.assert_allclose(left.features.iloc[:, 1:], right.features.iloc[:, 1:])


def test_gradual_result_rebases_intervals_after_warm_start() -> None:
    stream = make_synthetic_scenario(
        "gradual_covariate",
        n_instances=130,
        breakpoints=[60],
        min_segment_length=20,
        transition_length=30,
        random_state=2,
    )
    result = run_prequential(
        OnlinePairwiseSelector(reference_period=5, random_state=2),
        stream,
        n_warmstart=65,
    )
    assert result.drift_points == ()
    assert result.transition_intervals == ((0, 25),)
    head, tail = stream.split(65)
    assert head.drift_points == (60,)
    assert tail.transition_intervals == result.transition_intervals


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"phases": [StreamPhase("A", 0)]}, "phase length"),
        ({"phases": [StreamPhase("unknown", 5)]}, "Unknown feature regime"),
        ({"phases": [StreamPhase("A", 5, 2)]}, "preceding different"),
        (
            {"phases": [StreamPhase("A", 5), StreamPhase("B", 5, 6)]},
            "transition length",
        ),
        ({"performance_noise": -1}, "performance_noise"),
        ({"feature_noise": float("nan")}, "feature_noise"),
        ({"cutoff": 0}, "cutoff"),
        ({"regimes": {"A": FeatureRegime([0])}}, "mean"),
        ({"algorithm_names": ["a", "a"]}, "distinct"),
        ({"loading": [[0.0]]}, "loading"),
        (
            {"performance_model": PerformanceModelConfig([1000, 0], [[0, 0], [0, 0]])},
            "overflowed",
        ),
    ],
)
def test_rejects_bad_core_config(changes: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _stream(**changes)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"breakpoints": [5], "n_instances": 80}, "minimum segment"),
        ({"n_instances": 40, "n_breakpoints": 2}, "too small"),
        ({"breakpoints": [80, 40], "n_breakpoints": 2}, "minimum segment"),
        ({"breakpoints": [60], "breakpoint_distribution": "uniform"}, "conflict"),
        ({"transition_length": 400}, "too small"),
        ({"transition_length": 0.0}, "transition_length.*integer"),
        ({"transition_length": "50"}, "transition_length.*integer"),
        ({"n_features": 0}, "n_features"),
        ({"n_algorithms": 1}, "n_algorithms"),
    ],
)
def test_rejects_bad_scenario_config(changes: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_synthetic_scenario("gradual_covariate", **changes)


def test_raw_timeouts_and_prequential_integration() -> None:
    stream = make_synthetic_stream(
        {"A": FeatureRegime([0.0])},
        [StreamPhase("A", 35)],
        PerformanceModelConfig([np.log(80.0), np.log(120.0)], [[0.0, 0.0]]),
        cutoff=10.0,
        random_state=0,
    )
    assert (stream.performance.to_numpy() > stream.cutoff).all()
    for warm in (0, 10):
        selector = OnlinePairwiseSelector(reference_period=5, random_state=0)
        result = run_prequential(selector, stream, n_warmstart=warm)
        assert len(result.steps) == 35 - warm
        assert (result.steps["user_cost"] == 100.0).all()
        assert selector.buffer is not None
        assert selector.buffer.censored_mask().to_numpy().any()
