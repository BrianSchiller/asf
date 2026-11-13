# Frugal pairwise selector

`FrugalPairwiseSelector` (`asf.online.selectors`) adapts *frugal algorithm
selection* (Kuş et al., arXiv:2405.11059, 2024; JAIR 86, 2026) to a stream.
Like the Rook et al. selector it trains one classifier per algorithm pair,
but it runs a second algorithm **only when it is unsure** about the choice.

```python
from asf.online.selectors import FrugalPairwiseSelector
from asf.online.gate import UncertaintyGate
from asf.online.evaluation.prequential import run_prequential

selector = FrugalPairwiseSelector(gate=UncertaintyGate(threshold=0.25), random_state=0)
result = run_prequential(selector, stream, n_warmstart=20)
print(result.summary()["extra_run_rate"])
```

## How it decides

1. **Vote.** Each pair classifier (random forest by default, as in the paper)
   predicts which of its two algorithms is faster. The algorithm with the most
   votes runs (`RESULT_MODE = "primary"`); ties are broken at random.
2. **Measure uncertainty.** For every other algorithm `a`, the selector takes
   the pair (chosen, `a`) and computes its *least confidence*
   `1 - max(p, 1 - p)`: 0 = sure, 0.5 = coin flip. For two classes this ranks
   pairs exactly like margin or entropy would.
3. **Nominate.** The other algorithms are the nominees, most uncertain pair
   first. `PredictionRecord.uncertainty` is the uncertainty of the first one.
4. **Gate.** The default `UncertaintyGate(threshold=0.25, n=1)` runs the first
   nominee if the uncertainty is at least 0.25 (that is, if the pair predicts
   at most 75 % for either side). Any other gate can be passed as `gate=`, e.g.
   `AlwaysProbe(1)` to always run two algorithms like Rook et al.
5. **Learn.** After the runs, only the pairs whose algorithms both ran are
   refitted (`update_every` batches this).

Training labels: a finished run beats a timed-out run; pairs where both timed
out are skipped, since they cost the most and teach nothing. A pair whose
labels all agree predicts `(k + 1) / (n + 2)` (Laplace) instead of 100 %, so a
single example does not silence the gate. Pairs without data predict 0.5 and
cast no vote, so on a cold start the gate fires until data arrives.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `model_class` | `RandomForestClassifier` | Pair classifier; needs `predict_proba` |
| `candidate_order` | `"uncertainty"` | Nominee order; `"random"` is the paper's random baseline |
| `timeout_filter` | `False` | Drop a nominee if it and the chosen algorithm are both predicted to time out |
| `timeout_voting` | `False` | Exclude predicted timeouts from voting (unless all are) |
| `timeout_model_class` | `RandomForestClassifier` | Classifier of the timeout predictors |
| `cutoff` | `None` | Runtimes `>= cutoff` count as timeouts; falls back to `budget`, then to the first observation's cutoff. Needed for timeout predictors on a warm start |
| `update_every` | `1` | Refit the changed pairs every `k` instances |
| `window_size` | `None` | Buffer size; `None` keeps everything |

Drift handling: `RetrainResponse` refits all models on the buffer,
`ResetResponse` forgets all data seen so far, and a `retrain_policy` works as
for every online selector.

## Differences from the paper

* **Stream instead of pool.** The paper ranks all unlabelled
  (instance, pair) combinations and labels the top 1 % per round. Here each
  instance is decided on arrival, and only pairs containing the chosen
  algorithm can be labelled (it runs anyway). The batch size becomes the gate
  threshold.
* The paper's initial set (all algorithms on 20 instances) is the warm start
  (`n_warmstart=20`).
* **Not implemented:** dynamic time limits (they need a shorter budget for
  extra runs than for the primary run) and the cost-aware criteria PC/PD
  (they need runtime predictions).
