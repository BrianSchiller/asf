"""
Drift-triggered retraining: OnlinePairwiseSelector + PCAReconstructionDetector.

"extra" counts signals outside the tolerance window after the true drift point
(and signals before it). Here they are repeat signals: the detector is refitted
on the FULL history, which is still dominated by the old regime, so it keeps
flagging the new regime for a while and triggers several retrainings.

The stream has 3 latent factors mixed into 10 correlated features. After 300
instances the features shift off the principal subspace (covariate drift) and
a different algorithm becomes best. We compare

* no retraining,
* the detector with NannyML-style in-sample thresholds (as in Rook et al.),
* the detector with out-of-fold thresholds.
"""

import numpy as np
import pandas as pd

from asf.online.drift import PCAReconstructionDetector
from asf.online.evaluation import run_prequential
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams import InstanceStream

D, N_BEFORE, N_AFTER = 10, 300, 600
mix = np.random.default_rng(123).normal(size=(3, D))


def sample(n: int, seed: int, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) @ mix + 0.3 * rng.normal(size=(n, D))
    x[:, -2:] += shift  # moves data away from the retained subspace
    return x


features = np.vstack([sample(N_BEFORE, 1), sample(N_AFTER, 2, shift=3.0)])
runtimes = np.where(
    (np.arange(len(features)) < N_BEFORE)[:, None],
    [[5.0, 50.0, 20.0]],  # before: a is best
    [[50.0, 5.0, 20.0]],  # after:  b is best
)
index = pd.Index([f"inst_{i}" for i in range(len(features))])
stream = InstanceStream(
    pd.DataFrame(features, index=index, columns=[f"f{i}" for i in range(D)]),
    pd.DataFrame(runtimes, index=index, columns=["a", "b", "c"]),
    cutoff=1000.0,
    drift_points=(N_BEFORE,),
)

configs = {
    "no retraining": None,
    "PCA, in-sample": "in_sample",
    "PCA, cross-validated": "cross_validated",
}
print(f"{'method':>22} {'user cost':>10} {'signals':>8} {'delay':>6} {'extra':>6}")
for name, mode in configs.items():
    detector = (
        None
        if mode is None
        else PCAReconstructionDetector(
            chunk_size=25, reference_errors=mode, random_state=0
        )
    )
    selector = OnlinePairwiseSelector(
        reference_period=80, detector=detector, random_state=0
    )
    s = run_prequential(selector, stream).summary()
    print(
        f"{name:>22} {s['mean_user_cost']:10.2f} {int(s['n_drift_signals']):8d} "
        f"{s.get('mean_delay', float('nan')):6.0f} {int(s.get('n_false_alarms', 0)):6d}"
    )
