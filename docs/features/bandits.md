# Bandit selectors

`asf.online.selectors.bandits` treats algorithms as the arms of a
multi-armed bandit. By default each instance runs exactly one algorithm. With
an extra-run gate a second algorithm can run too, and the bandit learns from
every run it sees (see [Extra runs](#extra-runs)).

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
| `imputation` (how timed-out runs are learnt from) | none: the PAR10 penalty of `reward` | `CutoffImputer`, `DropCensored`, `ExpectedImputer`, `SampledImputer`; see [Censored runs](#censored-runs) |

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

## Extra runs

Bandits fill the prediction record, so the gates of
[online selection](online_selection.md#extra-run-gate) work on them:

```python
from asf.online.gate import AllNomineesGate, UncertaintyGate
from asf.online.selectors import LinTSSelector

# a second arm on the 20% most uncertain recent instances
LinTSSelector(gate=UncertaintyGate(rate=0.2, random_state=0), budget=cutoff, random_state=0)
```

| Field | Bandit value |
|---|---|
| `candidates` | all other arms by descending policy score (UCB index, Thompson sample): the runner-up first |
| `uncertainty` | how likely another arm is better, in `[0, 1]` (below) |
| `predicted_runtime` | the chosen arm's expected reward turned back into a runtime (`RewardTransform.to_performance`) |

* **Thompson Sampling:** `uncertainty_samples=32` joint posterior draws; the
  uncertainty is the share of draws in which the chosen arm is not the best.
  Ties share the win, so `K` arms without data give `(K - 1) / K`.
* **UCB:** the overlap of the confidence intervals `mean ± bonus` of the
  chosen arm and the runner-up, as intersection over union: 0 for disjoint
  intervals, 1 for identical ones or if either arm has no data.
* The uncertainty is only computed when a gate is set. Without one
  (default: `NoExtrasGate`) the bandit draws nothing extra and behaves as
  before. The draws are one numpy call per arm on the posteriors the decision
  already built; `uncertainty_samples=0` turns them off.
* `predicted_runtime` is a predicted PAR10 value, since timeouts are learnt
  with their penalty, and through the log reward a geometric mean. It is
  `None` for an arm without data and for the `"solved"` reward. With it,
  `TimeoutRiskGate` and `LogRuntimeResidual` work on bandits.
* Both arms update their posteriors and the clock advances once per
  instance. A timed-out extra run is penalised or imputed like any other.
* With several traces the uncertainty uses the combined scores and
  `predicted_runtime` the first trace.

### Exploring with the extra run

When the user gets the best result of all scheduled runs, the primary run
does not have to explore. `primary="greedy"` plays the arm with the best
posterior mean (arms without data first) and leaves exploration to the extra
run: the nominees stay ordered by the policy score, so the first extra is the
arm Thompson Sampling or UCB would have played, or its runner-up if that is
the greedy arm.

```python
ThompsonSamplingSelector(primary="greedy", gate=AllNomineesGate(1), budget=cutoff, random_state=0)
```

`scores` then holds the posterior means. Without a gate this never explores.

## Censored runs

A run that hits its budget is censored: the true runtime is unknown, only that
it is at least the budget. By default the bandit learns the PAR10 penalty for
such a run. With `imputation`, it learns an imputed reward instead, following
Tornede et al. (2022):

```python
from asf.online.imputation import SampledImputer
from asf.online.selectors.bandits import LinTSSelector

# Their bj_thompson: censored runs get a posterior draw above the cutoff.
LinTSSelector(imputation=SampledImputer(), budget=stream.cutoff, random_state=0)
```

| Rule | Learnt reward of a censored run | Source |
|---|---|---|
| `CutoffImputer()` | reward of solving exactly at the budget | eq. 11 |
| `DropCensored()` | nothing (the arm is not updated) | BlindUCB, eq. 9 |
| `ExpectedImputer()` | mean of the arm's predictive distribution beyond the budget | Buckley-James, as described in their text |
| `SampledImputer(max_tries=20)` | a posterior draw beyond the budget, after 20 failed draws the budget | Algorithm 1, lines 17-20, and their code |

* The rule uses the arm's posterior *before* the update, at the current
  instance. `SampledImputer` draws from the posterior of the mean (their code
  samples `theta ~ N(theta_hat, sigma A^-1)`); `ExpectedImputer` adds the
  noise variance, since it fills in an observation.
* The bound is the budget the run was stopped at (at most the cutoff).
  Warm-start runs at or above `budget` count as censored too.
* Imputation works on the reward scale (`"log"` or `"par"`), not on the
  natural-log runtime of the paper. The direction is the same (a longer
  runtime is a lower reward), but `noise_var` and `prior_precision` are not
  on the paper's scale, so their `sigma=1, lambda=0.5` do not carry over.
* With several traces or cells, the first trace's posterior in the
  instance's cell is used.
* `DropCensored` hides that an algorithm times out, so the bandit keeps
  choosing it in regions where it fails; it is a baseline, not a fix.
* Scoring stays PAR10: imputation only changes what the bandit learns.
* Not implemented from the paper: the bias-corrected and randomised LinUCB
  widths (BClinUCB, RandUCB) and the expected-PAR10 selection rule (`_rev`
  variants).

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
