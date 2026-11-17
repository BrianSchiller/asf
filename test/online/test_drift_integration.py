"""End to end: each covariate drift detector adapts a selector after a feature shift.

Replaces the per-detector integration tests. Data generators come from the
detector test modules so the shifts stay identical to the unit tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from asf.online.drift import (
    KSSubspaceEnsembleDetector,
    OneClassDriftDetector,
    PCAReconstructionDetector,
)
from asf.online.evaluation import run_prequential
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams.stream import InstanceStream

from . import test_ks_subspace_ensemble as ks
from . import test_ocdd as ocdd
from . import test_pca_detector as pca

DRIFT_AT = 200


def make_stream(pre: pd.DataFrame, post: pd.DataFrame) -> InstanceStream:
    """The best algorithm swaps at the covariate shift."""
    x = pd.concat([pre, post], ignore_index=True)
    perf = np.where(
        (np.arange(len(x)) < len(pre))[:, None], [[5, 50, 20]], [[50, 5, 20]]
    ).astype(float)
    idx = pd.Index([f"i{k}" for k in range(len(x))])
    x.index = idx
    return InstanceStream(
        x,
        pd.DataFrame(perf, index=idx, columns=["a", "b", "c"]),
        cutoff=1000.0,
        drift_points=(DRIFT_AT,),
    )


CASES: dict[str, tuple[Callable, Callable]] = {
    "pca": (
        lambda: PCAReconstructionDetector(
            chunk_size=25, reference_errors="cross_validated", random_state=0
        ),
        lambda: (pca.sample(200, 5), pca.sample(200, 6, shift=pca.OFF_SUBSPACE)),
    ),
    "ks": (
        lambda: KSSubspaceEnsembleDetector(
            subspace_size=3, window_size=25, random_state=0
        ),
        lambda: (ks.sample(200, 5), ks.sample(200, 6, shift=ks.BLOCK0)),
    ),
    "ocdd": (
        lambda: OneClassDriftDetector(window_size=50, baseline="cross_validated"),
        lambda: (ocdd.sample(200, 5), ocdd.sample(200, 6, shift=2.0)),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.slow
def test_detector_adapts_the_selector_after_the_shift(name: str) -> None:
    make_detector, make_data = CASES[name]
    stream = make_stream(*make_data())
    kw = dict(reference_period=80, random_state=0)
    adaptive = run_prequential(
        OnlinePairwiseSelector(detector=make_detector(), **kw), stream
    )
    static = run_prequential(OnlinePairwiseSelector(**kw), stream)

    assert adaptive.retrain_steps, "a drift must trigger a retrain"
    post_shift = [s for s in adaptive.drift_steps if s >= DRIFT_AT]
    assert post_shift and post_shift[0] - DRIFT_AT < 50  # within two windows
    after = slice(300, None)  # the stream starts cold; shift at position 200
    assert (
        adaptive.steps["user_cost"].iloc[after].mean()
        < static.steps["user_cost"].iloc[after].mean()
    )
