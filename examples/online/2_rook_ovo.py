"""
Online pairwise (one-vs-one) selector in the style of Rook et al. (FOGA 2025).

Cold start: no prior performance data. The selector runs random algorithm
pairs during a reference period, then trains one binary classifier per pair
from the parallel runs. The stream shifts into a new region of feature space
halfway through; periodic retraining (on the full history) adapts to it. A
drift detector can be passed via ``detector=`` instead (see ``asf.online.drift``).
"""

import numpy as np
import pandas as pd

from asf.online.evaluation import run_prequential
from asf.online.retraining import PeriodicRetrain
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams import InstanceStream

rng = np.random.default_rng(0)
n_before, n_after = 300, 300
f0 = np.concatenate([rng.normal(0, 1, n_before), rng.normal(8, 1, n_after)])
features = np.column_stack([f0, rng.normal(size=f0.size)])
# region 1: "a" is best, region 2: "b" is best, "c" is mediocre everywhere
runtimes = np.where(
    (np.arange(f0.size) < n_before)[:, None], [[5.0, 50.0, 20.0]], [[50.0, 5.0, 20.0]]
)
index = pd.Index([f"inst_{i}" for i in range(f0.size)])
stream = InstanceStream(
    pd.DataFrame(features, index=index, columns=["f0", "f1"]),
    pd.DataFrame(runtimes, index=index, columns=["a", "b", "c"]),
    cutoff=1000.0,
    drift_points=(n_before,),
)

for name, policy in [
    ("never retrain", None),
    ("retrain every 25", PeriodicRetrain(25)),
]:
    selector = OnlinePairwiseSelector(
        reference_period=50, retrain_policy=policy, random_state=0
    )
    summary = run_prequential(selector, stream).summary()
    print(
        f"{name:>18}: user cost {summary['mean_user_cost']:6.2f} "
        f"(SBS {summary['sbs_cost']:.1f}, VBS {summary['vbs_cost']:.1f}), "
        f"closed gap {summary['closed_gap']:.2f}, "
        f"retrains {int(summary['n_retrains'])}, "
        f"retrain time {summary['retrain_seconds']:.2f}s"
    )
