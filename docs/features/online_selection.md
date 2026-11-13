# Online Algorithm Selection

`asf.online` extends ASF with selectors that keep learning while instances
arrive and whose distribution may drift.

## Interaction contract

```python
schedule = selector.propose(features_of_one_instance)  # [(algo, budget), ...]
# ... run the scheduled algorithms ...
selector.observe(Observation(instance_id, features, runtimes, censored, cutoff))
```

The first entry of a schedule is the selected (primary) algorithm; further
entries are extra runs for information gathering, with the same budget as the
primary run. Only executed algorithms are reported back, so the selector's
data is partially observed.

Inside `propose` and `observe`, `AbstractOnlineSelector` runs the same steps
for every selector:

```text
propose:  record = _select(features)                 # subclass -> PredictionRecord
          extras = gate.extras(GateContext(...))      # extra-run gate
          last_prediction_ = record + extras          # logged per step

observe:  buffer.append(obs); _observe(obs)           # subclass learns
          signal = detector.update(...)               # feature or scalar detector
          if signal.drift: on_drift(signal)           # drift responses
          if retrain_policy.should_retrain(...): retrain()
          re-reference the detector if the model was rebuilt
```

## Writing a selector

Subclass `AbstractOnlineSelector` and implement:

| Hook | Purpose |
|---|---|
| `_fit_initial(features, performance)` | Initial model from the fully observed warm start (may be empty: cold start) |
| `_select(features) -> PredictionRecord` | Decision for one instance |
| `_observe(observation)` | Learn from the feedback (the buffer is already updated) |

Optional capability hooks, used by drift responses and retrain policies:
`_retrain()`, `_reset_model()`, `_discount_history(factor)`,
`_boost_exploration(factor, steps)`. A selector that does not implement a
hook cannot be combined with a response that needs it; this fails at `fit`.

Class attributes: `RESULT_MODE` (default scoring mode),
`SUPPORTS_COLD_START` (`False` for `StaticOnlineSelector`) and
`MANAGES_DETECTOR_FIT` (the selector calls `fit_detector()` itself, as the
Rook selector does at the end of its reference period).

All hyperparameters (discounts, windows, thresholds) are constructor
parameters. Constructor arguments are captured automatically as
`init_params_` and written to every result file.

### Prediction record

`PredictionRecord` (`asf.online.prediction`) is what a selector decided for
one instance. The base class adds the gate's `extras` and the `instance_id`
and keeps it as `selector.last_prediction_`.

| Field | Meaning |
|---|---|
| `chosen`, `budget` | Primary algorithm and its run budget (`<= 0` = cutoff) |
| `predicted_runtime` | Predicted performance of `chosen` on the raw stream scale, or `None` |
| `score`, `scores` | Selector-internal score of `chosen` / of every algorithm (higher = preferred) |
| `uncertainty` | Selector-defined uncertainty (higher = less sure), or `None` |
| `candidates` | The selector's nominees for extra runs, most useful first |
| `extras` | Extra algorithms actually scheduled (set by the base class) |

Built-in selectors: the Rook selector fills `score`/`scores` with pairwise
votes and nominates its `n_additional` sampled algorithms; bandits fill them
with the policy scores (UCB index, Thompson sample) and nominate nothing.
`predicted_runtime` and `uncertainty` are `None` for both.

### Extra-run gate

A gate (`asf.online.gate`) decides which extra algorithms run:

```python
class AbstractExtraRunGate(ABC):
    needs_instance_score: bool = False
    def extras(self, context: GateContext) -> Sequence[str]: ...
```

`GateContext` carries the prediction record, the nominees (`candidates`),
all algorithms, the instance features, the step, and optionally the
detector's per-instance score (`instance_score`, computed only if
`needs_instance_score`), the previous detector signal and the steps since the
last drift. `context.ranked_alternatives()` lists all other algorithms, the
nominees first and then by descending score.

`AlwaysProbe(n=None)` runs the first `n` nominees (all by default) and is the
default gate, so every selector keeps its built-in behaviour: the Rook
selector runs `n_additional` extras, the bandits none. `NeverProbe()` runs
none. The selector rejects unknown algorithms and drops duplicates and the
primary choice from a gate's answer. `UncertaintyGate(threshold, n=1)` runs
the first `n` nominees only if `prediction.uncertainty >= threshold`.

## Scoring and protocols

`run_prequential(selector, stream, n_warmstart, ...)` evaluates
test-then-train. Per instance it logs both result modes:

* `cost_primary`: PAR-k of the primary run,
* `cost_best_of_scheduled`: best PAR-k among all scheduled runs (parallel
  runs, as in Rook et al.),

and `user_cost`, the one selected by `result_mode` (default: the selector's
`RESULT_MODE`). `compute_cost` is the summed capped runtime of everything
that ran. `summary()` reports `mean_user_cost` and `closed_gap` for the
selected mode and both modes separately (`mean_cost_primary`,
`closed_gap_best_of_scheduled`, ...).

SBS is the single best algorithm on the *evaluated* stream (in hindsight), in
both protocols, so closed gaps are comparable across warm and cold starts.
With a warm start, the single best on the prefix is reported as
`sbs_cost_warmstart` / `closed_gap_warmstart`. VBS ties are broken by the
stream's algorithm order.

Two protocols are supported (`protocol=` is checked against `n_warmstart`):

* `"warm"` (`n_warmstart >= 1`): the selector is fitted on a fully observed
  prefix. Offline AS methods (`StaticOnlineSelector`) need this protocol.
* `"cold"` (`n_warmstart == 0`): OAS methods start without data.

By default, lower runtime is better. For `maximize=True`, higher quality
scores are better: SBS, VBS, `best_of_scheduled`, and closed gap all use that
direction. Use `cutoff=inf` and unlimited run budgets, since quality scores
do not reveal whether a time-limited run would finish. In this mode the cost
columns contain quality scores; `compute_cost` is NaN.

## Drift detection and responses

Two kinds of detectors (`asf.online.drift`) can be passed as `detector=`:

* `AbstractDriftDetector`, feature-based: `fit(features, performance)`,
  `update(features, performance) -> DriftSignal`, optional `score(features)`
  per instance, `reset(...)`. It is fitted on the buffer right after a warm
  start, or after `detector_reference_size` instances on a cold start.
* `AbstractScalarDriftDetector`, for ADWIN/EDDM-style methods on one number
  per instance: `update(value, censored=False) -> DriftSignal`, `reset()`.
  It needs a `drift_statistic`, an `AbstractDriftStatistic` that turns
  `(observation, prediction)` into a `ScalarSample(value, censored)`.
  `LogRuntimeResidual` gives `log(observed) - log(predicted)` runtime of the
  primary run; censored runs are passed on as lower bounds
  (`censored="emit"`) or dropped (`"skip"`).

When the detector signals drift, `on_drift(signal)` runs the configured
`drift_response` (one or a list, `asf.online.retraining`):

| Response | Selector hook | Implemented by |
|---|---|---|
| `RetrainResponse()` | `_retrain` | `OnlinePairwiseSelector` (the default with a detector), `OnlineEPMSelector` |
| `ResetResponse()` | `_reset_model` | bandits (`reset_arms`), `OnlineEPMSelector` |
| `ReweightResponse(factor)` | `_discount_history` | planned |
| `RaiseExplorationResponse(factor, steps)` | `_boost_exploration` | planned |

After a retrain or reset the detector is re-referenced (refitted on the
buffer, or reset if scalar). A `retrain_policy` (e.g. `PeriodicRetrain`) is
independent of drift signals: it is checked after every instance; if a drift
response already retrained on that instance, it does not retrain again.

`PCAReconstructionDetector` re-implements the NannyML-style detector used by
Rook et al.: standardise, fit a PCA (65% variance), and flag chunks whose mean
reconstruction error leaves a band of 3 reference standard deviations. Shifts
that stay inside the retained principal subspace do not change the error and
are not detected. `reference_errors="cross_validated"` gives more honest
thresholds than NannyML's in-sample errors when the reference set is small.

## Result files (schema v1)

```python
result = run_prequential(selector, stream, n_warmstart=100, stream_seed=0,
                         metadata={"config": "pilot-01"})
result.save("results/run-0")           # needs pip install 'asf[online]'
result = OnlineResult.load("results/run-0")
```

A run is a directory with `steps.parquet` (one row per evaluated instance)
and `meta.json` (everything else). Parquet columns use only `string`,
`int64`, `float64` and `bool`. JSON never contains NaN or Infinity
(non-finite values are `null`).

| Column | Type | Meaning |
|---|---|---|
| `step` | int64 | 0-based position in the evaluated stream (warm-start prefix excluded) |
| `instance_id` | string | `str()` of the instance id |
| `primary` | string | Algorithm the selector chose |
| `extras` | string | JSON list of extra scheduled algorithms, in order |
| `runs` | string | JSON `{algorithm: {"runtime": float, "censored": bool}}` for every executed algorithm, in schedule order; runtime capped at the budget |
| `n_runs` | int64 | Distinct algorithms executed |
| `pred_runtime`, `pred_score`, `pred_uncertainty` | float64 | Prediction record of the primary choice (NaN if not available) |
| `instance_score` | float64 | Detector's per-instance score passed to the gate (NaN if not requested) |
| `primary_runtime`, `primary_censored` | float64, bool | Observed (capped) runtime of the primary run and whether it was censored |
| `cost_primary`, `cost_best_of_scheduled` | float64 | PAR-k under both result modes |
| `user_cost` | float64 | The one matching `result_mode` |
| `compute_cost` | float64 | Summed capped runtimes of all executed runs |
| `vbs_cost`, `vbs_algorithm` | float64, string | Oracle cost and algorithm (ties by algorithm order) |
| `sbs_cost` | float64 | PAR-k of the single best algorithm on the evaluated stream |
| `sbs_cost_warmstart` | float64 | PAR-k of the single best on the warm-start prefix (NaN for cold) |
| `retrained` | bool | Selector retrained at this step |
| `drift_signal`, `drift_warning`, `detector_score` | bool, bool, float64 | Detector output |
| `drift_statistic` | float64 | Value fed to a scalar detector |
| `response` | string | Drift responses executed, comma-joined |
| `selector_seconds`, `detector_seconds` | float64 | Wall-clock of `propose` + `observe` without detector time, and inside detector calls |

`meta.json` keys: `schema_version`; `asf` (`version`, `commit`, `dirty`,
`commit_source` = `"git"`, `"direct_url"` or `"unknown"`); `selector`
(`class`, `params` with nested components as `{class, params}`); `stream`
(`name`, `seed`, `n_instances`, `n_evaluated`, `cutoff` (null = infinite),
`algorithms`, `n_features`, `drift_points`, `transition_intervals`, all in
`step` indexing); `protocol`; `n_warmstart`; `seed` (the selector's
`random_state`); `result_mode`; `par_factor`; `maximize`;
`sbs_algorithm_stream`; `sbs_algorithm_warmstart` (null for cold);
`retrain_steps`; `drift_steps`; `retrain_seconds`; `created_utc`; `extra`
(from `metadata=`).

Adding columns or keys keeps `schema_version`; renaming, removing or changing
the meaning of one bumps it. `load` rejects newer versions.

## Building blocks

| Module | Purpose |
|---|---|
| `asf.online.base` | `AbstractOnlineSelector`, `Observation`, `StepTrace` |
| `asf.online.prediction` | `PredictionRecord` |
| `asf.online.gate` | extra-run gates |
| `asf.online.buffer` | sliding window with partially observed performance |
| `asf.online.drift` | drift detectors (when to reconfigure) and drift statistics |
| `asf.online.retraining` | retrain policies and drift responses |
| `asf.online.selectors` | online selectors (how to reconfigure) |
| `asf.online.streams` | streams with injected drift |
| `asf.online.evaluation` | prequential evaluation, metrics, result files |

## Selectors

* `StaticOnlineSelector`: any offline ASF selector, never updated (lower
  bound; warm start only).
* `OnlinePairwiseSelector`: partial one-vs-one selector with selectable
  `variant="paper"` and `variant="code"` implementations.
* `FrugalPairwiseSelector`: one-vs-one selector that runs a second algorithm
  only when it is unsure (Kuş et al.); see [Frugal pairwise selector](frugal_pairwise.md).
* `OnlineEPMSelector`: one runtime model per algorithm (Degroote et al.,
  2018), see below.
* Bandits (`UCBSelector`, `ThompsonSamplingSelector`, `LinUCBSelector`,
  `LinTSSelector`) and their non-stationary variants: see
  [Bandit selectors](bandits.md).

The paper variant supports `additional_strategy="uniform"` or
`"error_guided"` (sampling inversely to estimated pairwise accuracy). The
Jeroen-code variant supports `"least_observed"` or `"cost_based"` (weights
`1 +` in-sample F1 error). The latter follows the code's maximizing-quality
assumption, so configure it with `maximize=True`. ASF estimates the paper
variant's pairwise accuracy with cross-validation; the paper's accuracy
weighting does not specify this estimation procedure.

### OnlineEPMSelector (runtime models)

The regression-based method of Degroote et al. (2018). Every algorithm has
its own regression model (default: a random forest with R `randomForest`
defaults, 500 trees) that predicts its PAR10 performance from the features.
A model learns only from instances its algorithm was run on, so the partially
observed online data needs no imputation.

```python
from sklearn.ensemble import RandomForestRegressor
from asf.online.selectors import OnlineEPMSelector

selector = OnlineEPMSelector(
    strategy="ucb",            # "greedy" (default), "epsilon_greedy", "ucb"
    gamma=1.0,                 # UCB weight; epsilon=0.05 for epsilon-greedy
    budget=stream.cutoff,      # cutoff for scoring the warm start
    random_state=0,
    # estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1, ...),
)
```

* **Strategies**: greedy picks the best prediction; epsilon-greedy picks a
  uniformly random algorithm with probability `epsilon`; UCB picks the lowest
  `prediction - gamma * sd` (runtime) or highest `prediction + gamma * sd`
  (`maximize=True`).
* **Uncertainty**: `uncertainty="jackknife"` (default, as in the paper via
  mlr: jackknife-after-bootstrap with bias correction), `"tree_std"` (spread
  across trees) or `None` (faster; not with UCB). It fills the prediction
  record's `uncertainty`, which gates can use.
* **Targets**: unsolved runs (censored, or above the cutoff) count as
  `par_factor * cutoff`. `target_transform="log10"` trains on log runtimes
  instead of the raw scale. `predicted_runtime` is the predicted PAR10 value
  of the chosen algorithm.
* **Updates**: models are refitted every `retrain_every=10` new data points
  (extra runs count), only those whose data changed. Algorithms without data
  are tried first; a model is fitted on an algorithm's first observation. By
  default all data is kept (`window_size=None`).
* **Runs**: one per instance (default gate `NeverProbe`). The paper's
  `allData` upper bound is `gate=AlwaysProbe()` with `result_mode="primary"`.
* **Drift responses**: `RetrainResponse` (refit now) and `ResetResponse`
  (forget all data and models).

A forest with 500 trees on about 1,500 instances takes tens of seconds to
fit on one core. Set `n_jobs` on the estimator, or use fewer trees for
pilots.

See `examples/online/` for runnable examples.
See [Synthetic streams](synthetic_streams.md) to configure stationary, sudden,
gradual and recurring covariate shifts for selector experiments.
