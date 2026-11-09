# Synthetic streams for online selection

`asf.online.streams` can produce reproducible streams to develop and debug
selectors before using benchmark-backed data. **Only the feature distribution
changes:** the model of algorithm runtime given observed features is shared by
every regime. A change in which algorithm wins *on average* can therefore
follow a feature shift without any concept drift.

## Start with a preset

```python
from asf.online.streams import make_synthetic_scenario

stream = make_synthetic_scenario(
    "gradual_covariate",
    n_instances=600,
    n_features=8,
    n_algorithms=5,
    n_breakpoints=2,
    breakpoint_distribution="uniform",  # seeded, feasible random positions
    min_segment_length=100,
    transition_length=60,
    random_state=42,
)
print(stream.drift_points)  # zero-based onsets
print(stream.transition_intervals)  # half-open [start, end) intervals
```

Presets are `"stationary"` (no change), `"sudden_covariate"`,
`"gradual_covariate"`, and `"recurring_covariate"` (alternates A-B-A by
default). The defaults are 600 instances, three features and three algorithms.
Use `n_breakpoints=0` for stationarity; a recurring stream requires at least two
breaks. Other presets start with one. Multiple breaks alternate distributions
A and B. A returning regime produces *new* samples, not a replay. Scenario kinds
describe drift schedules only; correlated features and off-subspace shifts are
configured explicitly with the core generator below.

Set `breakpoints=[200, 400]` for exact positions, **or** request a count via
`n_breakpoints` and choose `breakpoint_distribution="even"` (the default) or
`"uniform"`. Uniform samples feasible integer segment lengths subject to
`min_segment_length`; it is not a promise of equal lengths. Explicit positions
cannot be combined with a distribution argument. Each phase must fit the
minimum length, and gradual transitions must fit within their phase. The
`shift_strength` parameter controls separation between regime means. A larger
shift does not guarantee a detector alarm or a particular algorithm ranking.

## Custom regimes

```python
import numpy as np
from asf.online.streams import (
    FeatureRegime,
    PerformanceModelConfig,
    StreamPhase,
    make_synthetic_stream,
)

stream = make_synthetic_stream(
    regimes={
        "A": FeatureRegime(mean=[-2.0, 0.0]),
        "B": FeatureRegime(mean=[2.0, 0.0]),
    },
    phases=[
        StreamPhase("A", 200),
        StreamPhase("B", 150, transition_length=50),
        StreamPhase("A", 200),
    ],
    performance_model=PerformanceModelConfig(
        intercepts=[np.log(7.0), np.log(7.0), np.log(20.0)],
        coefficients=[[0.5, -0.5, 0.0], [0.0, 0.0, 0.0]],
    ),
    cutoff=100.0,
    random_state=42,
    feature_noise=0.3,
    performance_noise=0.1,
)
```

The runtime formula, fixed for the entire stream, is

```text
features = regime_mean + regime_scale × (standard_normal @ shared_loading)
           + feature_noise × independent_standard_normal
runtime = exp(intercepts + features @ coefficients
              + performance_noise × independent_standard_normal)
```

The loading defaults to an identity matrix; supply a single matrix of shape
`(n_latent, n_features)` to correlate features. Different regimes can have
different means or scales, but **cannot** have different runtime models or
performance-noise laws. Intercepts are log-runtime baselines; the coefficient
matrix has shape `(n_features, n_algorithms)`. Names default to `f0`, `f1`, ...
and `a0`, `a1`, ...; custom distinct names can be supplied to
`make_synthetic_stream`.

### Off-subspace shifts

A PCA reconstruction detector alarms when new features fall outside the
subspace learned from the reference window. Build such a stream by giving the
loading matrix fewer latent dimensions than features, then shifting the mean
along a coordinate the loading leaves empty:

```python
n_features, n_latent = 10, 8
loading = np.random.default_rng(0).normal(size=(n_latent, n_features))
loading[:, -2:] = 0.0  # last two features carry no latent structure

mean_a = np.zeros(n_features)
mean_b = np.zeros(n_features)
mean_b[-1] = 3.0  # shift outside the retained reference subspace
```

This remains a covariate shift: the shift direction is one of the observed
features, so it can matter to runtimes only through the usual fixed coefficient
matrix. Contrast it with an ordinary shift along a feature that already varies
in the reference window, which changes the region visited without necessarily
leaving the subspace.

Generated performance contains **raw positive runtimes**, including values
above `cutoff`. `run_prequential` decides which runs time out, applies budgets
and reports PAR penalties. The generator models runtime **minimization only**;
quality maximization requires different score and resource semantics. Use a
separate selector seed when comparing training policies on the same stream.

## Interpreting ground truth

For a sudden change, `drift_points` is the first position of the new feature
regime. For gradual drift, it is the **start** of the transition. Within a
gradual interval `[start, end)`, the probability of drawing an entire instance
from the new regime increases linearly from zero on the first row to one on
the last row; after that all rows use the new regime. The separate
`transition_intervals` preserve that range when a stream is split for warm
start, including a split inside the transition. A split inside a gradual
transition has a remaining interval but no *new* drift onset in its tail.

Existing point-based detection metrics measure delay from onset; they do not
fully describe performance during a gradual transition. Nor is an alarm
guaranteed: the shift may be too small or invisible to a given detector. The
stream retains the full performance matrix for evaluation, but a selector
only observes algorithms actually run by the evaluator.
