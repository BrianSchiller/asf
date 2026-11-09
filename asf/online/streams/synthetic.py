"""Controlled covariate-only streams for developing online selectors.

Feature distributions can change; the conditional runtime model is shared by
every regime and every instance. All generated values are raw runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from asf.online.streams.stream import InstanceStream


@dataclass(frozen=True)
class FeatureRegime:
    """Feature mean and scale of the shared latent Gaussian projection."""

    mean: Sequence[float]
    scale: float = 1.0


@dataclass(frozen=True)
class PerformanceModelConfig:
    """One fixed model: log runtime = intercepts + features @ coefficients + noise."""

    intercepts: Sequence[float]
    coefficients: Sequence[Sequence[float]]


@dataclass(frozen=True)
class StreamPhase:
    """Draw ``length`` rows from ``regime``; optionally transition into it.

    ``transition_length`` belongs to the start of this phase, not the end of
    the preceding phase. A value of zero gives a sudden change.
    """

    regime: str
    length: int
    transition_length: int = 0


def _int_param(value: int, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _array(value: object, name: str, ndim: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers.") from exc
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a {ndim}-D array of finite numbers.")
    return result


def _finite_number(value: float, name: str, positive: bool = False) -> float:
    requirement = "positive" if positive else "nonnegative"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and {requirement}.") from exc
    if not np.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {requirement}.")
    return number


def _names(names: Sequence[str] | None, prefix: str, count: int) -> list[str]:
    result = [f"{prefix}{i}" for i in range(count)] if names is None else list(names)
    if (
        len(result) != count
        or any(not isinstance(name, str) or not name for name in result)
        or len(set(result)) != count
    ):
        raise ValueError(f"{prefix} names must be {count} distinct nonempty strings.")
    return result


def make_synthetic_stream(
    regimes: Mapping[str, FeatureRegime],
    phases: Sequence[StreamPhase],
    performance_model: PerformanceModelConfig,
    cutoff: float,
    *,
    random_state: int | None = None,
    feature_noise: float = 0.0,
    performance_noise: float = 0.0,
    loading: Sequence[Sequence[float]] | np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    algorithm_names: Sequence[str] | None = None,
    name: str = "synthetic",
) -> InstanceStream:
    """Generate a runtime-minimization stream with a fixed P(runtime | features).

    A gradual transition draws each *whole* instance from the old or new
    feature regime. During its ``L`` rows, the probability of the new regime
    goes linearly from zero (first row) to one (last row). It does not blend
    runtime vectors or alter the global performance model.

    The return value retains raw runtimes above ``cutoff`` for the evaluator to
    censor. Separate RNGs for feature draws, transition membership, and
    performance noise make controlled comparisons reproducible.
    """
    if not regimes or any(not isinstance(k, str) or not k for k in regimes):
        raise ValueError("regimes must have nonempty string names.")
    if not phases:
        raise ValueError("At least one phase is required.")
    feature_noise = _finite_number(feature_noise, "feature_noise")
    performance_noise = _finite_number(performance_noise, "performance_noise")
    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be positive (or infinity).") from exc
    if np.isnan(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be positive (or infinity).")

    intercepts = _array(performance_model.intercepts, "intercepts", 1)
    coefficients = _array(performance_model.coefficients, "coefficients", 2)
    n_algorithms = intercepts.size
    n_features = coefficients.shape[0]
    if n_algorithms < 2 or n_features < 1 or coefficients.shape[1] != n_algorithms:
        raise ValueError(
            "coefficients must have shape (n_features, n_algorithms >= 2)."
        )
    feature_columns = _names(feature_names, "f", n_features)
    algorithm_columns = _names(algorithm_names, "a", n_algorithms)

    projection = (
        np.eye(n_features) if loading is None else _array(loading, "loading", 2)
    )
    if projection.shape[0] < 1 or projection.shape[1] != n_features:
        raise ValueError("loading must have shape (n_latent >= 1, n_features).")

    means: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for regime_name, regime in regimes.items():
        if not isinstance(regime, FeatureRegime):
            raise ValueError(f"{regime_name!r} must be a FeatureRegime.")
        means[regime_name] = _array(regime.mean, f"mean for {regime_name}", 1)
        if means[regime_name].shape != (n_features,):
            raise ValueError(f"mean for {regime_name!r} must have {n_features} values.")
        scales[regime_name] = _finite_number(regime.scale, f"scale for {regime_name!r}")

    n = 0
    boundaries: list[tuple[int, str, str, int]] = []
    for i, phase in enumerate(phases):
        if not isinstance(phase.regime, str) or not phase.regime:
            raise ValueError("Phase regime names must be nonempty strings.")
        if phase.regime not in regimes:
            raise ValueError(f"Unknown feature regime: {phase.regime!r}.")
        length = _int_param(phase.length, "phase length")
        transition = _int_param(phase.transition_length, "transition length", 0)
        if i == 0 or phase.regime == phases[i - 1].regime:
            if transition:
                raise ValueError("Transitions require a preceding different regime.")
        elif transition:
            if transition < 2 or transition > length:
                raise ValueError(
                    "Gradual transition length must be in [2, phase length]."
                )
            boundaries.append((n, phases[i - 1].regime, phase.regime, transition))
        else:
            boundaries.append((n, phases[i - 1].regime, phase.regime, 0))
        n += length

    seeds = np.random.SeedSequence(random_state).spawn(3)
    feature_rng, mixture_rng, performance_rng = (
        np.random.default_rng(seed) for seed in seeds
    )
    # Draw for all rows before assigning their regimes: changes to phase
    # boundaries do not change the underlying Gaussian draws at each position.
    latent_draws = feature_rng.normal(size=(n, projection.shape[0]))
    feature_errors = feature_rng.normal(size=(n, n_features))
    performance_errors = performance_rng.normal(size=(n, n_algorithms))

    memberships = [phase.regime for phase in phases for _ in range(phase.length)]
    intervals: list[tuple[int, int]] = []
    for start, old, new, duration in boundaries:
        if duration:
            probabilities = np.linspace(0.0, 1.0, duration)
            use_new = mixture_rng.random(duration) < probabilities
            for offset, chosen in enumerate(use_new):
                memberships[start + offset] = new if chosen else old
            intervals.append((start, start + duration))

    row_means = np.stack([means[regime] for regime in memberships])
    row_scales = np.array([scales[regime] for regime in memberships])
    features = (
        row_means
        + row_scales[:, None] * (latent_draws @ projection)
        + feature_noise * feature_errors
    )
    log_runtimes = (
        intercepts + features @ coefficients + performance_noise * performance_errors
    )
    if not np.isfinite(features).all() or not np.isfinite(log_runtimes).all():
        raise ValueError("Generated features or log runtimes are not finite.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        runtimes = np.exp(log_runtimes)
    if not np.isfinite(runtimes).all() or (runtimes <= 0).any():
        raise ValueError("Generated runtimes overflowed or underflowed; reduce scale.")

    index = pd.Index([f"inst_{i}" for i in range(n)])
    return InstanceStream(
        features=pd.DataFrame(features, index=index, columns=feature_columns),
        performance=pd.DataFrame(runtimes, index=index, columns=algorithm_columns),
        cutoff=float(cutoff),
        drift_points=tuple(start for start, _, _, _ in boundaries),
        name=name,
        transition_intervals=tuple(intervals),
    )


def _breakpoints(
    n_instances: int,
    count: int,
    minimum: int,
    positions: Sequence[int] | None,
    distribution: Literal["even", "uniform"] | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    segments = count + 1
    slack = n_instances - segments * minimum
    if slack < 0:
        raise ValueError(
            "n_instances is too small for the requested minimum segment length."
        )
    if positions is not None:
        if distribution is not None:
            raise ValueError(
                "Explicit breakpoints conflict with breakpoint_distribution."
            )
        pts = tuple(positions)
        if len(pts) != count or any(
            isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer))
            for p in pts
        ):
            raise ValueError("breakpoints must contain exactly n_breakpoints integers.")
        lengths = np.diff([0, *pts, n_instances])
        if np.any(lengths < minimum):
            raise ValueError(
                "breakpoints must be ordered with minimum segment spacing."
            )
        return tuple(int(p) for p in pts)

    if distribution not in (None, "even", "uniform"):
        raise ValueError("breakpoint_distribution must be 'even' or 'uniform'.")
    if distribution == "uniform" and count:
        # Stars and bars: uniformly sample feasible integer segment lengths.
        dividers = np.sort(rng.choice(slack + count, size=count, replace=False))
        extras = np.diff(np.r_[-1, dividers, slack + count]) - 1
        lengths = minimum + extras
    else:
        lengths = np.array(
            [
                minimum + slack // segments + (i < slack % segments)
                for i in range(segments)
            ]
        )
    return tuple(int(p) for p in np.cumsum(lengths)[:-1])


def make_synthetic_scenario(
    kind: Literal[
        "stationary",
        "sudden_covariate",
        "gradual_covariate",
        "recurring_covariate",
    ],
    *,
    n_instances: int = 600,
    n_features: int = 3,
    n_algorithms: int = 3,
    n_breakpoints: int | None = None,
    breakpoints: Sequence[int] | None = None,
    breakpoint_distribution: Literal["even", "uniform"] | None = None,
    min_segment_length: int = 50,
    transition_length: int | None = None,
    shift_strength: float = 3.0,
    cutoff: float = 100.0,
    feature_noise: float = 0.3,
    performance_noise: float = 0.1,
    random_state: int | None = None,
) -> InstanceStream:
    """Ready-to-use stationary or sudden/gradual/recurring covariate stream.

    Regimes alternate A/B when more than one breakpoint is requested. Every
    scenario keeps the *same* runtime model, even when winner frequencies shift.
    Scenario kinds are drift schedules only; correlated features or an
    off-subspace shift need an explicit ``make_synthetic_stream`` configuration.
    """
    kinds = (
        "stationary",
        "sudden_covariate",
        "gradual_covariate",
        "recurring_covariate",
    )
    if kind not in kinds:
        raise ValueError(f"Unknown synthetic scenario: {kind!r}.")
    n = _int_param(n_instances, "n_instances")
    d = _int_param(n_features, "n_features")
    m = _int_param(n_algorithms, "n_algorithms", 2)
    minimum = _int_param(min_segment_length, "min_segment_length")
    shift_strength = _finite_number(shift_strength, "shift_strength", positive=True)
    default_count = (
        0 if kind == "stationary" else 2 if kind == "recurring_covariate" else 1
    )
    requested_count = (
        n_breakpoints
        if n_breakpoints is not None
        else len(breakpoints)
        if breakpoints is not None
        else default_count
    )
    count = _int_param(requested_count, "n_breakpoints", 0)
    if kind == "stationary" and count:
        raise ValueError("stationary scenario cannot have breakpoints.")
    if kind != "stationary" and count < (2 if kind == "recurring_covariate" else 1):
        raise ValueError("This scenario requires more breakpoints for a feature shift.")

    requested_transition = (
        None
        if transition_length is None
        else _int_param(transition_length, "transition_length", 0)
    )
    gradual = kind == "gradual_covariate" or (
        kind == "recurring_covariate"
        and requested_transition is not None
        and requested_transition > 0
    )
    duration = (
        (50 if requested_transition is None else requested_transition) if gradual else 0
    )
    if kind == "gradual_covariate" and duration < 2:
        raise ValueError("gradual_covariate requires transition_length >= 2.")
    if duration == 1 or (not gradual and requested_transition not in (None, 0)):
        raise ValueError("transition_length must be zero or >= 2 for this scenario.")
    # Independent breakpoint randomness: changing placement does not alter the
    # feature or runtime random streams used by make_synthetic_stream.
    rng = np.random.default_rng(np.random.SeedSequence(random_state).spawn(4)[3])
    pts = _breakpoints(
        n,
        count,
        max(minimum, duration) if breakpoints is None else minimum,
        breakpoints,
        breakpoint_distribution,
        rng,
    )
    lengths = np.diff([0, *pts, n])
    if any(length < duration for length in lengths[1:]):
        raise ValueError("Gradual transition length exceeds its phase length.")
    phases = [
        StreamPhase("A" if i % 2 == 0 else "B", int(length), duration if i else 0)
        for i, length in enumerate(lengths)
    ]

    mean_a, mean_b = np.zeros(d), np.zeros(d)
    mean_a[0], mean_b[0] = -shift_strength / 2, shift_strength / 2
    regimes = {"A": FeatureRegime(mean_a), "B": FeatureRegime(mean_b)}
    # A convex intercept/slope envelope gives every algorithm a feature region
    # where it can win, rather than padding larger scenarios with dead columns.
    slopes = np.linspace(0.55, -0.55, m)
    intercepts = np.log(7.0) + 0.8 * (slopes**2 - 0.55**2)
    coefficients = np.zeros((d, m))
    coefficients[0] = slopes

    return make_synthetic_stream(
        regimes,
        phases,
        PerformanceModelConfig(intercepts, coefficients),
        cutoff,
        random_state=random_state,
        feature_noise=feature_noise,
        performance_noise=performance_noise,
        name=kind,
    )
