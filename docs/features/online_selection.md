# Online Algorithm Selection

`asf.online` extends ASF with selectors that keep learning while instances
arrive and whose distribution may drift.

## Interaction contract

```python
schedule = selector.propose(features_of_one_instance)  # [(algo, budget), ...]
# ... run the scheduled algorithms ...
selector.observe(Observation(instance_id, features, runtimes, censored, cutoff))
```

The first entry of a schedule is the selected algorithm; further entries are
additional runs for information gathering. Only executed algorithms are
reported back, so the selector's data is partially observed.

## Scoring

`run_prequential` reports, per instance, `user_cost` (PAR-k of what the user
gets) and `compute_cost` (summed capped runtimes of everything that ran). Which
result the user gets depends on the selector's `RESULT_MODE`: `"primary"` (the
first algorithm) or `"best_of_scheduled"` (parallel runs, best result wins).

## Building blocks

| Module | Purpose |
|---|---|
| `asf.online.base` | `AbstractOnlineSelector`, `Observation` |
| `asf.online.buffer` | sliding window with partially observed performance |
| `asf.online.drift` | drift detectors (when to reconfigure) |
| `asf.online.retraining` | retrain policies |
| `asf.online.selectors` | online selectors (how to reconfigure) |
| `asf.online.streams` | streams with injected drift |
| `asf.online.evaluation` | prequential evaluation and metrics |

## Selectors

* `StaticOnlineSelector`: any offline ASF selector, never updated (lower bound).
* `OnlinePairwiseSelector`: partial one-vs-one selector with selectable
  `variant="paper"` and `variant="jeroen_code"` implementations.

The paper variant supports `additional_strategy="uniform"` or
`"error_guided"` (sampling inversely to estimated pairwise accuracy). The
Jeroen-code variant supports `"least_observed"` or `"cost_based"` (weights
`1 +` in-sample F1 error). The latter follows the code's maximizing-quality
assumption, so configure it with `maximize=True`. ASF estimates the paper
variant's pairwise accuracy with cross-validation; the paper's accuracy
weighting does not specify this estimation procedure.

## Drift detectors

`PCAReconstructionDetector` re-implements the NannyML-style detector used by
Rook et al.: standardise, fit a PCA (65% variance), and flag chunks whose mean
reconstruction error leaves a band of 3 reference standard deviations. Shifts
that stay inside the retained principal subspace do not change the error and
are not detected. `reference_errors="cross_validated"` gives more honest
thresholds than NannyML's in-sample errors when the reference set is small.

See `examples/online/` for runnable examples.
