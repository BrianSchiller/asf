"""Feature drift detectors on a real ASlib stream that switches scenarios.

Same stream as ``4_aslib_sat12.py``: half of SAT12-HAND, -INDU and -RAND,
then the other halves (switches at 384, 968, 1649, 2032 and 2615). No
selector is involved; the detectors only see the instance features.

1. Fixed reference (the first 200 instances, all HAND): how often each
   detector signals per segment, and its mean per-instance score.
2. Re-referenced on the last 200 instances after every signal (as a
   selector would after reacting): delay, missed switches and false alarms.

Usage: python examples/online/5_detectors_sat12.py <folder with the ASlib scenarios>
"""

import sys
import time

import numpy as np
import pandas as pd
from asf.online.drift import (
    KSSubspaceEnsembleDetector,
    OneClassDriftDetector,
    PCAReconstructionDetector,
)
from asf.online.evaluation.metrics import detection_metrics
from asf.online.streams import combine_scenarios, grouped_stream
from asf.online.streams.loaders import load_aslib_raw

ROOT = sys.argv[1] if len(sys.argv) > 1 else "aslib_data"
REF = 200  # reference size
TOLERANCE = 150  # a signal within this many instances after a switch counts

names = ["HAND", "INDU", "RAND"]
features, performance, cutoff, labels = combine_scenarios(
    {name: load_aslib_raw(f"{ROOT}/SAT12-{name}") for name in names}
)
stream = grouped_stream(
    features, performance, cutoff, labels, schedule=names * 2, seed=0, name="SAT12"
)
# impute and standardise with the reference instances only
X = stream.features.to_numpy()
X = np.where(np.isnan(X), np.nan_to_num(np.nanmedian(X[:REF], axis=0)), X)
std = X[:REF].std(axis=0)
X = (X - X[:REF].mean(axis=0)) / np.where(std > 0, std, 1.0)
X = pd.DataFrame(X, stream.features.index, stream.features.columns)

DETECTORS = {
    "PCA": lambda: PCAReconstructionDetector(random_state=0),
    "KS ensemble": lambda: KSSubspaceEnsembleDetector(random_state=0),
    "OCDD": lambda: OneClassDriftDetector(),
}

points = list(stream.drift_points)
edges = [REF, *points, len(X)]
segment = pd.cut(
    range(REF, len(X)),
    edges,
    right=False,
    labels=[f"{labels[X.index[a]]} {a}-{b}" for a, b in zip(edges, edges[1:])],
)

signals, scores, switches = {}, {}, {}
for name, make in DETECTORS.items():
    t = time.perf_counter()
    # 1) fixed reference, never re-referenced
    det = make().fit(X.iloc[:REF])
    flags = pd.Series([det.update(X.iloc[[k]]).drift for k in range(REF, len(X))])
    signals[name] = flags.groupby(segment, observed=False).sum()
    score = pd.Series(np.asarray(det.score(X.iloc[REF:]), dtype=float))
    scores[name] = score.groupby(segment, observed=False).mean()

    # 2) re-referenced on the last REF instances after every signal
    det, alarms = make().fit(X.iloc[:REF]), []
    for k in range(REF, len(X)):
        if det.update(X.iloc[[k]]).drift:
            alarms.append(k)
            det.reset(X.iloc[k + 1 - REF : k + 1])
    switches[name] = {
        **detection_metrics(points, alarms, tolerance=TOLERANCE),
        "signals": len(alarms),
        "seconds": round(time.perf_counter() - t, 1),
    }
    print(f"done: {name}")

pd.set_option("display.width", 200)
print(f"\n1) Reference = first {REF} instances (HAND). Signals per segment:")
print(pd.DataFrame(signals).T.to_string())
print("\nMean per-instance score per segment (higher = less familiar):")
print(pd.DataFrame(scores).T.round(3).to_string())
print(
    f"\n2) Re-referenced on the last {REF} instances after each signal "
    f"(switches at {points}, tolerance {TOLERANCE}):"
)
print(pd.DataFrame(switches).T.to_string())
