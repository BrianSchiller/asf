"""What the feature drift detectors share, and that each adapts a selector.

Data comes from the detector test modules, so the shifts match the unit tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from asf.online.drift import (
    AbstractDriftDetector,
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
from .conftest import stream_from

DRIFT_AT = 200


def make_stream(pre: pd.DataFrame, post: pd.DataFrame) -> InstanceStream:
    """The best algorithm swaps at the covariate shift."""
    x = pd.concat([pre, post], ignore_index=True)
    before = (np.arange(len(x)) < len(pre))[:, None]
    perf = np.where(before, [[5, 50, 20]], [[50, 5, 20]])
    return stream_from(x, perf, cutoff=1000.0, drift_points=(DRIFT_AT,))


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


DETECTORS: dict[str, Callable[[], AbstractDriftDetector]] = {
    "pca": lambda: PCAReconstructionDetector(random_state=0),
    "ks": lambda: KSSubspaceEnsembleDetector(subspace_size=3, random_state=0),
    "ocdd": lambda: OneClassDriftDetector(window_size=40),
}


@pytest.mark.parametrize("name", list(DETECTORS))
def test_shared_contract(name: str) -> None:
    make = DETECTORS[name]
    ref, new = ks.sample(100, 0), ks.sample(60, 1, shift=ks.BLOCK0)
    with pytest.raises(RuntimeError):
        make().update(new)
    with pytest.raises(RuntimeError):
        make().score(new)
    for bad in (ref.iloc[:1], pd.DataFrame(index=range(5))):
        with pytest.raises(ValueError):
            make().fit(bad)
    det = make().fit(ref)
    # columns are matched by name; a missing one is an error
    assert (det.score(new[new.columns[::-1]]) == det.score(new)).all()
    with pytest.raises(ValueError):
        det.score(new.drop(columns="f0"))
    # performance is ignored
    perf = pd.DataFrame({"a": np.arange(100.0)})
    assert make().fit(ref, perf).update(new, perf.iloc[:60]) == det.update(new)
