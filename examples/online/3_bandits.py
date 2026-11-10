"""Compare the bandit selectors.

Stationary baselines: UCB, Thompson Sampling, LinUCB, LinTS. Non-stationary:
DS-UCB and DS-TS (discounted), SW-UCB and SW-TS (sliding window) and f-dsw TS
(discounted + per-arm window, combined). Clustered: DTS-OC (online k-means,
discounted Beta TS per cluster).

Every bandit runs exactly one algorithm per instance and learns only from that
run (no extra runs for information gathering). The context-free bandits (UCB,
TS and all non-stationary variants) can at best find the single best
algorithm at any time; DTS-OC can find one per cluster; the linear ones
(LinUCB, LinTS) use the instance features to pick per instance.

The stream keeps one runtime model throughout; only the feature distribution
shifts (at instances 150 and 300).
"""

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
from asf.online.streams import make_synthetic_scenario

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

for kind in ["gradual_covariate", "sudden_covariate"]:
    stream = make_synthetic_scenario(
        kind,
        n_instances=1000,
        n_features=10,
        n_algorithms=10,
        breakpoints=[200, 400, 500, 700, 900] if kind != "stationary" else None,
        random_state=0,
        shift_strength=3,
    )
    print(f"\n{kind} stream (cutoff {stream.cutoff:g}, 50 warm-start instances)")
    for name, cls in SELECTORS.items():
        # budget = cutoff, so warm-start rewards use the same scale as the stream.
        selector = cls(budget=stream.cutoff, random_state=0)
        s = run_prequential(selector, stream, n_warmstart=50).summary()
        print(
            f"{name:>8}: user cost {s['mean_user_cost']:7.2f} "
            f"(SBS {s['sbs_cost']:.2f}, VBS {s['vbs_cost']:.2f}), "
            f"closed gap {s['closed_gap']:5.2f}"
        )
