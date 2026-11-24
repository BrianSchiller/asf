# Streams from benchmark data

ASlib scenarios have no time order, so drift is constructed by **ordering
real instances**. Every instance keeps its own runtimes, so the relation
from features to runtimes never changes: these streams have covariate drift
(the feature distribution moves) and no concept drift. Which algorithm wins
*on average* can still change, because different instances arrive.

All builders are in `asf.online.streams` and return an `InstanceStream`
that `run_prequential` accepts directly.

| Builder | Drift | `drift_points` |
|---|---|---|
| `grouped_stream` | sudden or recurring (switch between groups) | every switch |
| `grouped_stream(transition_length=L)` | gradual switch over `L` instances | start of every transition |
| `sorted_stream` | incremental over the whole stream | none (one interval `(0, n)`) |

## Switching between scenarios

`combine_scenarios` stacks scenarios with the same features, algorithms and
cutoff, and labels each instance with its scenario. SAT12-HAND, SAT12-INDU
and SAT12-RAND fit together (115 features, 31 solvers, cutoff 1200, no shared
instances). SAT12-ALL overlaps with them and is refused alongside them.

```python
from asf.online.streams import combine_scenarios, grouped_stream
from asf.online.streams.loaders import load_aslib_raw

root = "path/to/aslib_data"
features, performance, cutoff, labels = combine_scenarios(
    {name: load_aslib_raw(f"{root}/SAT12-{name}") for name in ["HAND", "INDU", "RAND"]}
)

# half of HAND, half of INDU, half of RAND, then the other halves
stream = grouped_stream(
    features, performance, cutoff, labels,
    schedule=["HAND", "INDU", "RAND"] * 2,
    seed=0,
)
```

`schedule` lists one entry per segment. Repeating an entry gives recurring
drift; a tuple such as `("HAND", "INDU")` is a segment that mixes groups.
With `segment_length=None` (default) all instances are used and each group is
split evenly over its segments; an int (or one int per segment) draws that
many instances at random instead. No instance appears twice.
`transition_length` turns each switch into a gradual one: around the switch,
the share of the new group's instances rises linearly from 0 to 1.

## Any grouping

`labels` can be any per-instance grouping:

```python
labels = performance.idxmin(axis=1)               # best algorithm (Rook et al. sudden/recurrent)
labels = pd.qcut(features["vars_clauses_ratio"], 4, labels=False)  # feature bins
labels = cluster_labels(features, n_clusters=4, seed=0)            # feature-space clusters
```

`cluster_labels` (KMeans on median-imputed, standardised features) works on
any scenario, so sudden and recurring drift can be built even where no family
of scenarios exists.

Rook et al.'s sudden stream with a balanced prefix of three groups is, up to
sampling details, `schedule=[("BF", "FF", "WF"), "NF"]`. Differences from
their code: groups in a mixed segment are sampled in proportion to their
size (Rook: equal numbers), and instances never repeat (Rook's recurrent
phases may reuse instances).

## Sorting

```python
from asf.online.streams import sorted_stream

stream = sorted_stream(features, performance, cutoff, key=features["vars_clauses_ratio"])
stream = sorted_stream(features, performance, cutoff, key=features.nclauses / features.nvars)
# Rook et al. smooth stream: by the larger runtime of two solvers, descending
stream = sorted_stream(features, performance, cutoff,
                       key=performance[["a", "b"]].max(axis=1), ascending=False)
```

Instances without a key value are dropped with a warning. `jitter` adds
Gaussian noise to the rank (scaled to [0, 1]), so the stream is only roughly
sorted and differs per seed; `jitter=0` sorts exactly, with ties broken by
the seed.

## Bin packing (Rook et al.)

`load_bin_packing(folder)` reads the `.bpp` files of one instance set of
[Kevin-Sim/BPP](https://github.com/Kevin-Sim/BPP), packs every instance with
best, first, next and worst fit, and returns the Falkenauer fitness (`k = 2`,
higher is better) as performance and `cutoff = inf`. Selectors therefore need
`maximize=True`. The features are the 10 tsfresh features Rook et al. selected,
computed with numpy; on `E_120_N_40_60` (the set their code uses, although the
paper describes `U(20, 100)` items) they and the fitness values match the
authors' `BP_selected_features.csv`.

```python
from asf.online.streams import grouped_stream, sorted_stream
from asf.online.streams.loaders import load_bin_packing

features, performance, cutoff = load_bin_packing("BppInstances/E_120_N_40_60")
winner = performance.idxmax(axis=1)

# sudden3: 500 reference instances won by BF/FF/WF, the rest of them, then NF
stream = grouped_stream(features, performance, cutoff, winner,
                        schedule=[("BF", "FF", "WF"), ("BF", "FF", "WF"), "NF"],
                        segment_length=[500, 2500, 1000], seed=0)
# smoothWFFF: by the better of FF and WF, best first
stream = sorted_stream(features, performance, cutoff,
                       key=performance[["FF", "WF"]].max(axis=1), ascending=False)
```

Ties in best and worst fit go to the earliest opened bin, which reproduces the
authors' fitness values.
