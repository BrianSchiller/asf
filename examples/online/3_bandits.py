"""Compare the four bandit baselines: UCB, Thompson Sampling, LinUCB, LinTS.

Every bandit runs exactly one algorithm per instance and learns only from that
run (no extra runs for information gathering). The context-free bandits (UCB,
TS) can at best find the single best algorithm; the linear ones (LinUCB,
LinTS) use the instance features to pick per instance.

Both streams share the same runtime model; the second one additionally shifts
the feature distribution after 300 instances.
"""

from asf.online.evaluation import run_prequential
from asf.online.selectors import (
    LinTSSelector,
    LinUCBSelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from asf.online.streams import make_synthetic_scenario

SELECTORS = {
    "UCB": UCBSelector,
    "TS": ThompsonSamplingSelector,
    "LinUCB": LinUCBSelector,
    "LinTS": LinTSSelector,
}

for kind in ["sudden_covariate"]:
    stream = make_synthetic_scenario(
        kind,
        n_instances=400,
        n_features=5,
        n_algorithms=5,
        breakpoints=[150, 300] if kind != "stationary" else None,
        random_state=0,
    )
    print(f"\n{kind} stream (cutoff {stream.cutoff:g}, 50 warm-start instances)")
    for name, cls in SELECTORS.items():
        # budget = cutoff, so warm-start rewards use the same scale as the stream.
        selector = cls(budget=stream.cutoff, random_state=0)
        s = run_prequential(selector, stream, n_warmstart=50).summary()
        print(
            f"{name:>7}: user cost {s['mean_user_cost']:7.2f} "
            f"(SBS {s['sbs_cost']:.2f}, VBS {s['vbs_cost']:.2f}), "
            f"closed gap {s['closed_gap']:5.2f}"
        )
