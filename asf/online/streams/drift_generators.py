"""
Turn static (ASlib-like) datasets into streams with covariate drift.

The builders only *order* real instances, each keeping its own runtimes, so
the streams have covariate drift and no concept drift:

* :func:`combine_scenarios` stacks datasets sharing features and solvers
  (e.g. SAT12-HAND, -INDU, -RAND), labelling each instance with its source.
* :func:`grouped_stream` plays groups one after another: sudden, gradual
  and recurring drift (Rook et al.'s sudden and recurrent streams).
* :func:`sorted_stream` sorts by a per-instance value: incremental drift
  (with solver runtimes as key, Rook et al.'s smooth streams).
* :func:`cluster_labels` groups instances by features, for
  :func:`grouped_stream` on any scenario.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from asf.online.streams.stream import InstanceStream

Dataset = tuple[pd.DataFrame, pd.DataFrame, float]


def _int_param(value: int, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def combine_scenarios(
    scenarios: Mapping[str, Dataset],
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.Series]:
    """
    Stack ``name -> (features, performance, cutoff)`` datasets that share
    features, algorithms and cutoff. Returns ``(features, performance,
    cutoff, labels)``, ``labels`` holding each instance's scenario name.
    """
    if not scenarios:
        raise ValueError("Need at least one scenario.")
    names = list(scenarios)
    features0, performance0, cutoff = scenarios[names[0]]
    feature_cols, algo_cols = list(features0.columns), list(performance0.columns)
    for name in names[1:]:
        features, performance, other_cutoff = scenarios[name]
        if set(features.columns) != set(feature_cols):
            raise ValueError(f"{name!r} has different features than {names[0]!r}.")
        if set(performance.columns) != set(algo_cols):
            raise ValueError(f"{name!r} has different algorithms than {names[0]!r}.")
        if float(other_cutoff) != float(cutoff):
            raise ValueError(f"{name!r} has a different cutoff than {names[0]!r}.")
    features = pd.concat([scenarios[n][0][feature_cols] for n in names])
    performance = pd.concat([scenarios[n][1][algo_cols] for n in names])
    if features.index.has_duplicates:
        dupes = features.index[features.index.duplicated()].unique().tolist()
        raise ValueError(f"Instance ids occur in several scenarios: {dupes[:5]}")
    labels = pd.Series(
        np.repeat(names, [len(scenarios[n][0]) for n in names]),
        index=features.index,
        name="scenario",
    )
    return features, performance, float(cutoff), labels


def grouped_stream(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    cutoff: float,
    labels: pd.Series,
    schedule: Sequence[object],
    segment_length: int | Sequence[int] | None = None,
    transition_length: int = 0,
    seed: int | None = None,
    name: str = "grouped",
) -> InstanceStream:
    """
    Play groups of instances one after another.

    Each ``schedule`` entry is one segment: a group label, or a tuple of
    labels for a mixed segment. Repeating entries gives recurring drift, e.g.
    ``["HAND", "INDU", "RAND"] * 2`` with ``segment_length=None`` plays one
    half of every group, then the other. No instance appears twice;
    instances are shuffled within a segment; unlabelled or unscheduled
    instances are not used.

    Parameters
    ----------
    segment_length : int, sequence of int or None, default=None
        Instances per segment (one value or one per segment), drawn from what
        is left of its groups. ``None``: each group is split evenly over the
        segments it appears in.
    transition_length : int, default=0
        0 = sudden. Otherwise (``>= 2``, at most the shortest segment) a
        gradual transition centred on each boundary, in which the share of
        new instances rises linearly from 0 to 1.

    ``drift_points`` of the result are the segment boundaries (for gradual
    transitions: the start of each transition, also in
    ``transition_intervals``).
    """
    if not schedule:
        raise ValueError("schedule must not be empty.")
    labels = labels.reindex(features.index)
    segments = [tuple(e) if isinstance(e, (tuple, list)) else (e,) for e in schedule]
    known = set(labels.dropna())
    unknown = {g for seg in segments for g in seg} - known
    if unknown:
        raise ValueError(f"Unknown groups in schedule: {sorted(map(str, unknown))}")
    rng = np.random.default_rng(seed)
    # Every group's instances in random order; segments take from the front.
    pools = {
        g: list(rng.permutation(labels.index[labels == g].to_numpy()))
        for seg in segments
        for g in seg
    }

    parts: list[np.ndarray] = []
    if segment_length is None:
        uses = {g: sum(g in seg for seg in segments) for g in pools}
        chunks = {
            g: np.array_split(np.array(pools[g], dtype=object), uses[g]) for g in pools
        }
        taken = dict.fromkeys(pools, 0)
        for seg in segments:
            part = []
            for g in seg:
                part.extend(chunks[g][taken[g]])
                taken[g] += 1
            parts.append(rng.permutation(np.array(part, dtype=object)))
    else:
        lengths = (
            [segment_length] * len(segments)
            if isinstance(segment_length, (int, np.integer))
            else list(segment_length)
        )
        if len(lengths) != len(segments):
            raise ValueError("Need one segment_length per schedule entry.")
        for seg, length in zip(segments, lengths):
            length = _int_param(length, "segment_length")
            left = [(g, i) for g in seg for i in pools[g]]
            if length > len(left):
                raise ValueError(
                    f"Segment {seg} needs {length} instances, "
                    f"only {len(left)} are left."
                )
            pick = set(rng.choice(len(left), size=length, replace=False).tolist())
            part = [left[k][1] for k in sorted(pick)]
            used = set(part)
            for g in seg:
                pools[g] = [i for i in pools[g] if i not in used]
            parts.append(rng.permutation(np.array(part, dtype=object)))
    if any(len(p) == 0 for p in parts):
        raise ValueError("A segment is empty; its group has too few instances.")

    order = np.concatenate(parts)
    bounds = np.cumsum([len(p) for p in parts])[:-1]
    boundaries = [
        int(b) for b, prev, nxt in zip(bounds, segments, segments[1:]) if prev != nxt
    ]
    intervals: list[tuple[int, int]] = []
    points = boundaries
    if transition_length:
        length = _int_param(transition_length, "transition_length", 2)
        if length > min(len(p) for p in parts):
            raise ValueError("transition_length must not exceed the shortest segment.")
        points = []
        for b in boundaries:
            start, end = b - length // 2, b + length - length // 2
            # Old keys have density 2(1-x), new ones 2x: after sorting, the share
            # of new instances at relative position x is about x.
            keys = np.r_[
                1 - np.sqrt(rng.random(b - start)), np.sqrt(rng.random(end - b))
            ]
            order[start:end] = order[start:end][np.argsort(keys, kind="stable")]
            points.append(start)
            intervals.append((start, end))

    return InstanceStream(
        features.loc[order],
        performance.loc[order],
        float(cutoff),
        drift_points=tuple(points),
        name=name,
        transition_intervals=tuple(intervals),
    )


def sorted_stream(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    cutoff: float,
    key: pd.Series,
    ascending: bool = True,
    jitter: float = 0.0,
    seed: int | None = None,
    name: str = "sorted",
) -> InstanceStream:
    """
    Order instances by ``key``: incremental drift.

    ``key`` can be a feature, a derived feature or solver runtimes
    (``performance[["a", "b"]].max(axis=1)``, ``ascending=False``: Rook et
    al.'s smooth streams). Instances without a key are dropped (with a
    warning). ``jitter`` adds Gaussian noise to the [0, 1]-scaled rank, so the
    order is only roughly sorted; ties are broken at random. The result has
    no drift points; the whole stream is one transition interval.
    """
    if not jitter >= 0:
        raise ValueError("jitter must be >= 0.")
    key = key.reindex(features.index)
    missing = int(key.isna().sum())
    if missing:
        warnings.warn(f"Dropping {missing} instances without a sort value.")
        key = key.dropna()
    if key.empty:
        raise ValueError("No instance has a sort value.")
    rng = np.random.default_rng(seed)
    shuffled = key.iloc[rng.permutation(len(key))]
    rank = shuffled.rank(method="first", ascending=ascending).to_numpy() / len(key)
    rank = rank + jitter * rng.normal(size=len(key))
    order = shuffled.index[np.argsort(rank, kind="stable")]
    return InstanceStream(
        features.loc[order],
        performance.loc[order],
        float(cutoff),
        drift_points=(),
        name=name,
        transition_intervals=((0, len(order)),),
    )


def cluster_labels(
    features: pd.DataFrame, n_clusters: int, seed: int | None = None
) -> pd.Series:
    """
    KMeans labels on median-imputed, standardised features, for
    :func:`grouped_stream`.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_clusters = _int_param(n_clusters, "n_clusters", 2)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KMeans(n_clusters=n_clusters, n_init=10, random_state=seed),
    )
    return pd.Series(model.fit_predict(features), index=features.index, name="cluster")


__all__ = ["combine_scenarios", "grouped_stream", "sorted_stream", "cluster_labels"]
