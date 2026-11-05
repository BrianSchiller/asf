"""
Evaluate an offline ASF selector on a drifting stream (no adaptation).

This is the "static" lower bound that online methods have to beat.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from asf.online.evaluation import run_prequential
from asf.online.selectors import StaticOnlineSelector
from asf.online.streams import InstanceStream
from asf.selectors import PairwiseClassifier

rng = np.random.default_rng(0)
n, drift_at, cutoff = 400, 250, 100.0
features = rng.normal(size=(n, 3))

# Before the drift algorithm "a" wins when f0 > 0, afterwards the roles flip.
a_wins = features[:, 0] > 0
a_wins[drift_at:] = ~a_wins[drift_at:]
runtimes = np.where(a_wins[:, None], [[5.0, 60.0]], [[60.0, 5.0]])

index = pd.Index([f"inst_{i}" for i in range(n)])
stream = InstanceStream(
    pd.DataFrame(features, index=index, columns=["f0", "f1", "f2"]),
    pd.DataFrame(runtimes, index=index, columns=["a", "b"]),
    cutoff=cutoff,
    drift_points=(drift_at,),
)

selector = StaticOnlineSelector(
    PairwiseClassifier(model_class=RandomForestClassifier, random_state=0)
)
result = run_prequential(selector, stream, n_warmstart=100)

for key, value in result.summary().items():
    print(f"{key:>26}: {value:.3f}")
