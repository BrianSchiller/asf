# Bandit selectors

`asf.online.selectors.bandits` treats algorithms as the arms of a
multi-armed bandit. Each instance runs exactly one algorithm, and the selector
learns only from that run.

## Baselines

| Selector | Context | Exploration |
|---|---|---|
| `UCBSelector` | none | UCB1 |
| `ThompsonSamplingSelector` | none | Gaussian Thompson Sampling |
| `LinUCBSelector` | linear in the features | LinUCB (disjoint models) |
| `LinTSSelector` | linear in the features | Linear Thompson Sampling |
| `DiscountedUCBSelector` | none | DS-UCB: UCB on discounted counts |
| `SlidingWindowUCBSelector` | none | SW-UCB: UCB over the last `window` steps |
| `DiscountedTSSelector` | none | DS-TS: Gaussian TS, discounted by `gamma` |
| `SlidingWindowTSSelector` | none | SW-TS: Gaussian TS over the last `window` steps |
| `FDSWTSSelector` | none | f-dsw TS: Beta TS, discounted trace + per-arm window, combined by `aggregate` |
| `DTSOCSelector` | cluster of the instance | DTS-OC: discounted Beta TS per online cluster |

```python
from asf.online.evaluation import run_prequential
from asf.online.selectors import LinTSSelector

selector = LinTSSelector(budget=stream.cutoff, random_state=0)
result = run_prequential(selector, stream, n_warmstart=50)
```

Always pass `budget=<cutoff>`: the warm start is scored before the stream's
cutoff is known. `examples/online/3_bandits.py` compares all of them.

## DTS-OC

`DTSOCSelector` (Dzhoha & Rozora, 2023) clusters instances online and runs a
separate discounted Thompson Sampling bandit in every cluster:

1. **Cluster:** `OnlineKMeans` assigns the instance to its nearest centroid.
   Centroids are initialised by k-means on the warm start, or (cold start) by
   the first `n_clusters` distinct instances. After an instance is observed,
   its centroid moves with discounted statistics (`cluster_gamma`).
2. **Select:** one Beta sample per algorithm from the cluster's posteriors.
3. **Update:** the played algorithm's counts in that cluster are discounted by
   `gamma` (`PullDiscount`: per pull, not per step) and the new reward is
   added, scaled by the fuzzy penalty
   `phi = 1 / sum_y (d_assigned / d_y)^(2 / (l - 1))`, which down-weights
   instances that lie between clusters.

Deviations from the paper: the centroid moves in `observe` instead of
`propose`; rewards are the fractional `"unit"` reward (binary: pass
`reward=RewardTransform("solved")`); `per_step_discount=True` decays every arm
in every cluster each step instead. With `posterior=partial(BayesianLinearRegression, ...)`
and `contextual=True` it becomes LinTS per cluster.

## Building blocks

Every selector is an `AbstractContextualBandit` assembled from parts that can
be swapped independently:

| Part | Default | Alternatives |
|---|---|---|
| `posterior` (per-arm reward model) | `GaussianPosterior` / `BayesianLinearRegression` | `BetaPosterior`, your own `Posterior` subclass |
| `policy` | `UCBPolicy` / `ThompsonPolicy` | your own `ExplorationPolicy` |
| `weighting` (which past observations count) | `Uniform` | `ExponentialDiscount`, `PullDiscount`, `SlidingWindow`, `LastPulls`, `NearestNeighbours`, `ConfidenceWeighting`, combined with `*` |
| `traces` + `aggregate` (several weightings, scores combined per arm) | one trace | e.g. `{"historic": ExponentialDiscount(0.99), "hot": LastPulls(30)}` with `"mean"` / `"min"` / `"max"` |
| `partitioner` (separate models per region) | `SinglePartition` | `OnlineKMeans`, your own `Partitioner` (trees, ...) |
| `confidence` (weight fixed when an observation arrives) | none | `GaussianConfidence`, your own `ConfidenceModel` |
| `reward` | `RewardTransform("auto")`: `-log10(PAR10 + 1)` | `"unit"` (the same in `[0, 1]`, for Beta posteriors), `"solved"` (binary), `"par"`, `"raw"` |

Non-stationary variants are compositions, not new classes:

```python
from asf.online.selectors.bandits import ExponentialDiscount, NearestNeighbours

LinTSSelector(weighting=ExponentialDiscount(0.99))  # discounted LinTS
ThompsonSamplingSelector(  # discounted kNN window
    weighting=ExponentialDiscount(0.99) * NearestNeighbours(10)
)
```

When a trace's weighting is uniform or an exponential discount, each arm's
posterior is updated incrementally. Otherwise it is refitted from the stored
history at every decision. Both give the same result where both apply.

## Tuning notes

* `noise_var` (and `alpha` for UCB, `scale` for TS) sets how much the bandit
  explores relative to the reward scale. The defaults are the textbook values;
  tune them per scenario.
* The `"par"` reward gives runs well below the cutoff almost the same reward,
  so the bandit mostly learns whether an algorithm solves an instance, not
  how fast. Prefer the default `"log"` reward unless that is what you want.
* Under PAR10 scoring, every timeout during exploration is expensive.
  Strong discounting (small `gamma`) forgets past timeouts and re-explores
  them, so check `gamma` against the cost of a timeout.
