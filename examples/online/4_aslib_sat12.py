"""Compare online selectors on a real ASlib stream that switches scenarios.

SAT12-HAND, SAT12-INDU and SAT12-RAND share features, solvers and cutoff.
The stream plays half of each scenario, then the other halves:
HAND, INDU, RAND, HAND, INDU, RAND (five switches, covariate drift only).

The static selector is trained on the first 100 instances (warm start); all
other selectors start cold on the whole stream. "Frugal + KS" adds a drift
detector to the frugal selector and runs every solver once after each alarm.
All are compared on the same instances: everything after the first 100.

Features are median-imputed and standardised with statistics of the first
100 instances only (no runtimes), so the linear bandits get usable contexts
on a cold start.

Usage: python examples/online/4_aslib_sat12.py <folder with the ASlib scenarios>
"""

import sys

import numpy as np
import pandas as pd
from asf.online.drift import KSSubspaceEnsembleDetector
from asf.online.evaluation import run_prequential
from asf.online.gate import ProbeAllAfterDrift, UncertaintyGate
from asf.online.retraining import PeriodicRetrain, RereferenceResponse
from asf.online.selectors import (
    FrugalPairwiseSelector,
    LinTSSelector,
    LinUCBSelector,
    OnlinePairwiseSelector,
    StaticOnlineSelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from asf.online.streams import combine_scenarios, grouped_stream
from asf.online.streams.loaders import load_aslib_raw
from asf.selectors import MultiClassClassifier

ROOT = sys.argv[1] if len(sys.argv) > 1 else "aslib_data"
WARM = 100
PAR = 10.0

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
    "UCB": (UCBSelector(budget=cutoff, random_state=0), 0),
    "TS": (ThompsonSamplingSelector(budget=cutoff, random_state=0), 0),
    "LinUCB": (LinUCBSelector(budget=cutoff, random_state=0), 0),
    "LinTS": (LinTSSelector(budget=cutoff, random_state=0), 0),
    "Online pairwise": (
        OnlinePairwiseSelector(
            reference_period=0, retrain_policy=PeriodicRetrain(100), random_state=0
        ),
        0,
    ),
    "Frugal pairwise": (
        FrugalPairwiseSelector(candidate_order="runner_up", random_state=0),
        0,
    ),
    "Frugal + KS": (
        FrugalPairwiseSelector(
            candidate_order="runner_up",
            gate=ProbeAllAfterDrift(UncertaintyGate(threshold=0.25, n=1)),
            detector=KSSubspaceEnsembleDetector(reference_size=WARM, random_state=0),
            detector_reference_size=WARM,
            drift_response=RereferenceResponse(),
            random_state=0,
        ),
        0,
    ),
    "Static multi-class": (
        StaticOnlineSelector(MultiClassClassifier(random_state=0)),
        WARM,
    ),
}

# Segments between switches, in stream positions; the first starts after the prefix.
edges = [WARM, *stream.drift_points, len(stream)]
seg_names = [
    f"{labels[stream.features.index[a]]} {a}-{b}" for a, b in zip(edges, edges[1:])
]
segment = pd.Series(pd.cut(range(len(stream)), edges, right=False, labels=seg_names))


def hours(costs: pd.Series) -> pd.Series:
    """Sum per segment and in total, in hours."""
    out = costs.groupby(segment, observed=False).sum()
    out.index = out.index.astype(str)
    out["total"] = costs.sum()
    return out / 3600


user, compute, runs, solved, decide = {}, {}, {}, {}, {}
for name, (selector, n_warm) in SELECTORS.items():
    steps = run_prequential(selector, stream, n_warmstart=n_warm, par_factor=PAR).steps
    steps.index = steps["step"] + n_warm  # position in the full stream
    steps = steps.loc[WARM:]
    user[name], compute[name] = hours(steps["user_cost"]), hours(steps["compute_cost"])
    runs[name] = steps["n_runs"].sum()  # solver runs, extra runs included
    solved[name] = (steps["user_cost"] < PAR * cutoff).mean()
    decide[name] = (steps["selector_seconds"] + steps["detector_seconds"]).sum()
    print(f"done: {name}")

par = performance.loc[stream.features.index].reset_index(drop=True)
par = par.where(par <= cutoff, PAR * cutoff).iloc[WARM:]
sbs, vbs = par[par.mean().idxmin()], par.min(axis=1)
user["SBS (hindsight)"], user["VBS"] = hours(sbs), hours(vbs)
solved["SBS (hindsight)"] = (sbs < PAR * cutoff).mean()
solved["VBS"] = (vbs < PAR * cutoff).mean()

pd.set_option("display.width", 200)
print(f"\nPAR{PAR:g} runtime the user gets, in hours (instances {WARM}-{len(stream)}):")
print(pd.DataFrame(user).T.round(1).to_string())
print(
    "\nTotals: user PAR10 and compute (all runs, extra runs included) in hours,"
    " number of solver runs, solved share, selector decision time in seconds:"
)
print(
    pd.DataFrame(
        {
            "user PAR10": {k: v["total"] for k, v in user.items()},
            "compute": {k: v["total"] for k, v in compute.items()},
            "solvers run": runs,
            "solved": solved,
            "decision s": decide,
        }
    )
    .round(2)
    .to_string()
)
