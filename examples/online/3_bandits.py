"""Compare the bandit selectors on a real ASlib stream that switches scenarios.

Stationary baselines: UCB, Thompson Sampling, LinUCB, LinTS. Non-stationary:
DS-UCB and DS-TS (discounted), SW-UCB and SW-TS (sliding window) and f-dsw TS
(discounted + per-arm window, combined). Clustered: DTS-OC (online k-means,
discounted Beta TS per cluster).

Every bandit runs exactly one algorithm per instance and learns only from that
run. The context-free bandits (UCB, TS and all non-stationary variants) can at
best find the single best algorithm at any time; DTS-OC can find one per
cluster; the linear ones (LinUCB, LinTS) use the instance features.

Stream: as in ``4_aslib_sat12.py``, half of SAT12-HAND, -INDU and -RAND, then
the other halves (covariate drift only). The first 100 instances are the
warm start; features are imputed and standardised with their statistics.

Usage: python examples/online/3_bandits.py <folder with the ASlib scenarios>
"""

import sys

import numpy as np
import pandas as pd
from asf.online.evaluation import run_prequential
from asf.online.selectors import (
    DiscountedTSSelector,
    DiscountedUCBSelector,
    DTSOCSelector,
    FDSWTSSelector,
    LinTSSelector,
    LinUCBSelector,
    SlidingWindowTSSelector,
    SlidingWindowUCBSelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from asf.online.streams import combine_scenarios, grouped_stream
from asf.online.streams.loaders import load_aslib_raw

ROOT = sys.argv[1] if len(sys.argv) > 1 else "aslib_data"
WARM = 100

names = ["HAND", "INDU", "RAND"]
features, performance, cutoff, labels = combine_scenarios(
    {name: load_aslib_raw(f"{ROOT}/SAT12-{name}") for name in names}
)
stream = grouped_stream(
    features, performance, cutoff, labels, schedule=names * 2, seed=0, name="SAT12"
)
X = stream.features.to_numpy()
X = np.where(np.isnan(X), np.nan_to_num(np.nanmedian(X[:WARM], axis=0)), X)
std = X[:WARM].std(axis=0)
X = (X - X[:WARM].mean(axis=0)) / np.where(std > 0, std, 1.0)
stream.features = pd.DataFrame(X, stream.features.index, stream.features.columns)

SELECTORS = {
    "UCB": UCBSelector,
    "TS": ThompsonSamplingSelector,
    "LinUCB": LinUCBSelector,
    "LinTS": LinTSSelector,
    "DS-UCB": DiscountedUCBSelector,
    "SW-UCB": SlidingWindowUCBSelector,
    "DS-TS": DiscountedTSSelector,
    "SW-TS": SlidingWindowTSSelector,
    "f-dsw TS": FDSWTSSelector,
    "DTS-OC": DTSOCSelector,
}

print(f"SAT12 stream (cutoff {cutoff:g}, {WARM} warm-start instances)")
for name, cls in SELECTORS.items():
    # budget = cutoff, so warm-start rewards use the same scale as the stream.
    selector = cls(budget=cutoff, random_state=0)
    s = run_prequential(selector, stream, n_warmstart=WARM).summary()
    print(
        f"{name:>8}: user cost {s['mean_user_cost']:8.1f} "
        f"(SBS {s['sbs_cost']:.1f}, VBS {s['vbs_cost']:.1f}), "
        f"closed gap {s['closed_gap']:5.2f}"
    )
