"""Streams built from static datasets: grouping, sorting, clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from asf.online.evaluation import run_prequential
from asf.online.selectors import StaticOnlineSelector
from asf.online.streams import (
    cluster_labels,
    combine_scenarios,
    grouped_stream,
    sorted_stream,
)
from asf.online.streams.stream import InstanceStream
from asf.selectors import PairwiseClassifier

Data = tuple[pd.DataFrame, pd.DataFrame, float, pd.Series]

CUTOFF = 100.0


def _dataset(
    prefix: str, n: int, shift: float, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"{prefix}{i}" for i in range(n)])
    features = pd.DataFrame(rng.normal(shift, 1.0, (n, 2)), idx, ["f0", "f1"])
    performance = pd.DataFrame(rng.uniform(1, 10, (n, 2)), idx, ["a", "b"])
    return features, performance, CUTOFF


@pytest.fixture
def data() -> Data:
    """Three groups H (10), I (12), R (8) that are far apart in feature space."""
    return combine_scenarios(
        {
            "H": _dataset("h", 10, 0.0, 0),
            "I": _dataset("i", 12, 10.0, 1),
            "R": _dataset("r", 8, -10.0, 2),
        }
    )


def _groups(stream: InstanceStream, labels: pd.Series) -> list[str]:
    return [labels[i] for i in stream.features.index]


def test_combine_scenarios_labels_and_checks(data: Data) -> None:
    features, performance, cutoff, labels = data
    assert len(features) == len(performance) == len(labels) == 30
    assert cutoff == CUTOFF
    assert labels.value_counts().to_dict() == {"I": 12, "H": 10, "R": 8}
    h, (f, p, _) = _dataset("h", 5, 0), _dataset("i", 5, 0)
    with pytest.raises(ValueError, match="algorithms"):
        combine_scenarios({"H": h, "I": (f, p.rename(columns={"b": "c"}), CUTOFF)})
    with pytest.raises(ValueError, match="features"):
        combine_scenarios({"H": h, "I": (f.rename(columns={"f1": "g"}), p, CUTOFF)})
    with pytest.raises(ValueError, match="cutoff"):
        combine_scenarios({"H": h, "I": (f, p, 50.0)})
    with pytest.raises(ValueError, match="several"):
        combine_scenarios({"H": h, "I": h})


def test_sudden_switch_uses_every_instance_once(data: Data) -> None:
    features, performance, cutoff, labels = data
    s = grouped_stream(features, performance, cutoff, labels, ["H", "I", "R"], seed=0)
    assert sorted(s.features.index) == sorted(features.index)
    assert _groups(s, labels) == ["H"] * 10 + ["I"] * 12 + ["R"] * 8
    assert s.drift_points == (10, 22) and s.transition_intervals == ()
    pd.testing.assert_frame_equal(s.performance, performance.loc[s.features.index])


def test_recurring_halves(data: Data) -> None:
    features, performance, cutoff, labels = data
    s = grouped_stream(
        features, performance, cutoff, labels, ["H", "I", "R"] * 2, seed=0
    )
    assert not s.features.index.has_duplicates and len(s) == 30
    assert _groups(s, labels) == (["H"] * 5 + ["I"] * 6 + ["R"] * 4) * 2
    assert s.drift_points == (5, 11, 15, 20, 26)


def test_fixed_lengths_mixed_segment_and_exhaustion(data: Data) -> None:
    features, performance, cutoff, labels = data
    s = grouped_stream(
        features,
        performance,
        cutoff,
        labels,
        [("H", "I"), "R", "H"],
        segment_length=[16, 4, 3],
        seed=0,
    )
    groups = _groups(s, labels)
    assert set(groups[:16]) == {"H", "I"} and groups[16:20] == ["R"] * 4
    assert groups[20:] == ["H"] * 3 and not s.features.index.has_duplicates
    with pytest.raises(ValueError, match="left"):
        grouped_stream(
            features, performance, cutoff, labels, ["R", "R"], segment_length=5
        )
    with pytest.raises(ValueError, match="Unknown"):
        grouped_stream(features, performance, cutoff, labels, ["X"])


def test_gradual_transition(data: Data) -> None:
    features, performance, cutoff, labels = data
    s = grouped_stream(
        features, performance, cutoff, labels, ["H", "I"], transition_length=8, seed=3
    )
    groups = _groups(s, labels)
    assert sorted(s.features.index) == sorted(labels.index[labels != "R"])
    assert s.drift_points == (6,) and s.transition_intervals == ((6, 14),)
    assert groups[:6] == ["H"] * 6 and groups[14:] == ["I"] * 8  # pure outside
    assert groups[6:14].count("H") == 4  # nothing lost inside the window
    with pytest.raises(ValueError, match="shortest"):
        grouped_stream(
            features, performance, cutoff, labels, ["H", "R"], transition_length=10
        )


def test_gradual_share_of_new_group_rises() -> None:
    labels = pd.Series(
        ["A"] * 2000 + ["B"] * 2000, index=[f"x{i}" for i in range(4000)]
    )
    features = pd.DataFrame({"f": np.arange(4000.0)}, index=labels.index)
    performance = pd.DataFrame({"a": 1.0, "b": 2.0}, index=labels.index)
    s = grouped_stream(
        features,
        performance,
        CUTOFF,
        labels,
        ["A", "B"],
        transition_length=2000,
        seed=0,
    )
    window = (labels[s.features.index[1000:3000]] == "B").to_numpy()
    shares = [window[k : k + 500].mean() for k in range(0, 2000, 500)]
    assert shares == sorted(shares) and shares[0] < 0.25 and shares[-1] > 0.75


def test_sorted_stream(data: Data) -> None:
    features, performance, cutoff, _ = data
    s = sorted_stream(features, performance, cutoff, features["f0"], seed=0)
    assert (np.diff(s.features["f0"]) >= 0).all()
    assert s.drift_points == () and s.transition_intervals == ((0, 30),)
    down = sorted_stream(
        features, performance, cutoff, performance.max(axis=1), ascending=False
    )
    assert (np.diff(down.performance.max(axis=1)) <= 0).all()
    key = features["f0"].copy()
    key.iloc[:3] = np.nan
    with pytest.warns(UserWarning, match="Dropping 3"):
        assert len(sorted_stream(features, performance, cutoff, key)) == 27


def test_sorted_jitter_varies_with_seed(data: Data) -> None:
    features, performance, cutoff, _ = data

    def order(seed: int) -> list:
        return list(
            sorted_stream(
                features, performance, cutoff, features["f0"], jitter=0.2, seed=seed
            ).features.index
        )

    assert order(1) == order(1) and order(1) != order(2)


def test_cluster_labels(data: Data) -> None:
    features, *_, labels = data
    features = features.copy()
    features.iloc[0, 0] = np.nan  # imputed, not an error
    clusters = cluster_labels(features, 3, seed=0)
    assert clusters.index.equals(features.index) and clusters.nunique() == 3
    # the three well-separated groups are found exactly
    assert pd.crosstab(labels, clusters).gt(0).sum(axis=1).eq(1).all()
    pd.testing.assert_series_equal(clusters, cluster_labels(features, 3, seed=0))


def test_prequential_on_grouped_stream(data: Data) -> None:
    features, performance, cutoff, labels = data
    s = grouped_stream(features, performance, cutoff, labels, ["H", "I", "R"], seed=0)
    sel = StaticOnlineSelector(PairwiseClassifier(model_class=DecisionTreeClassifier))
    res = run_prequential(sel, s, n_warmstart=5)
    assert len(res.steps) == 25 and res.drift_points == (5, 17)
