"""Shared fixtures: a tiny synthetic stream with one abrupt drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asf.online.streams.stream import InstanceStream

CUTOFF = 100.0


def make_stream(n: int = 120, drift_at: int = 80, seed: int = 0) -> InstanceStream:
    """Algo ``a`` is best when f0 > 0 before drift; after drift the roles flip."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 2))
    good, bad = 5.0, 50.0
    a_best = f[:, 0] > 0
    a_best[drift_at:] = ~a_best[drift_at:]
    perf = np.where(a_best[:, None], [[good, bad]], [[bad, good]])
    idx = pd.Index([f"i{k}" for k in range(n)])
    return InstanceStream(
        pd.DataFrame(f, index=idx, columns=["f0", "f1"]),
        pd.DataFrame(perf, index=idx, columns=["a", "b"]),
        cutoff=CUTOFF,
        drift_points=(drift_at,),
        name="toy",
    )


@pytest.fixture
def stream() -> InstanceStream:
    return make_stream()
