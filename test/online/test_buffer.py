from typing import Iterable

import numpy as np
import pandas as pd

from asf.online.base import Observation
from asf.online.buffer import ObservationBuffer


def _obs(
    i: int, runtimes: dict[str, float], censored: Iterable[str] = ()
) -> Observation:
    x = pd.DataFrame([[float(i), 0.0]], index=[f"i{i}"], columns=["f0", "f1"])
    return Observation(f"i{i}", x, runtimes, frozenset(censored), cutoff=100.0)


def test_partial_observations_are_nan_and_censoring_tracked() -> None:
    buf = ObservationBuffer(["a", "b"], ["f0", "f1"], window_size=10)
    buf.append(_obs(0, {"a": 3.0}))
    buf.append(_obs(1, {"a": 100.0, "b": 7.0}, censored={"a"}))
    perf, cens = buf.performance(), buf.censored_mask()
    assert np.isnan(perf.loc["i0", "b"])
    assert cens.loc["i1", "a"] and not cens.loc["i1", "b"]
    assert buf.n_observed("b") == 1


def test_window_drops_oldest() -> None:
    buf = ObservationBuffer(["a", "b"], ["f0", "f1"], window_size=3)
    for i in range(5):
        buf.append(_obs(i, {"a": 1.0, "b": 2.0}))
    assert list(buf.features().index) == ["i2", "i3", "i4"]


def test_pair_data_only_rows_with_both_algorithms() -> None:
    buf = ObservationBuffer(["a", "b"], ["f0", "f1"], window_size=10)
    buf.append(_obs(0, {"a": 1.0}))
    buf.append(_obs(1, {"a": 1.0, "b": 2.0}))
    x, ya, yb = buf.pair_data("a", "b")
    assert list(x.index) == ["i1"] and yb.iloc[0] == 2.0
