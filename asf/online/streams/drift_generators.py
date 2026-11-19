"""
Turn static (ASlib-like) datasets into streams with covariate drift.

Every builder only *orders* real instances; each instance keeps its own
runtimes. So the relation from features to runtimes never changes: the
streams have covariate drift (the feature distribution moves) and no concept
drift.

* :func:`combine_scenarios` stacks datasets that share features and solvers
  (e.g. SAT12-HAND, -INDU, -RAND) and labels each instance with its source.
* :func:`grouped_stream` plays groups of instances one after another: sudden,
  gradual and recurring drift (Rook et al.'s sudden and recurrent streams).
* :func:`sorted_stream` sorts instances by any per-instance value (a feature,
  a derived feature, solver runtimes as in Rook et al.'s smooth streams):
  incremental drift over the whole stream.
* :func:`cluster_labels` groups instances by their features, for
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
    Stack datasets that share features, algorithms and cutoff.

    Parameters
    ----------
    scenarios : Mapping[str, tuple]
        Name -> ``(features, performance, cutoff)``, e.g. the output of
        :func:`~asf.online.streams.loaders.load_aslib_raw` per scenario.

    Returns
    -------
    tuple
        ``(features, performance, cutoff, labels)``; ``labels`` holds the
        scenario name of every instance (index = instance ids).
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

    Each schedule entry is one segment of the stream. An entry is a group
    label, or a tuple (or list) of labels for a segment that mixes groups.
    Repeating an entry gives recurring drift, e.g.
    ``["HAND", "INDU", "RAND", "HAND", "INDU", "RAND"]`` with
    ``segment_length=None`` plays one half of every group, then the other.
    No instance appears twice; instances are shuffled within a segment.

    Parameters
    ----------
    features, performance : pd.DataFrame
        The dataset (same index).
    cutoff : float
        Scenario cutoff.
    labels : pd.Series
        Group of every instance (index = instance ids). Instances without a
        label, or whose group is not scheduled, are not used.
    schedule : sequence
        Group label (or tuple of labels) per segment.
    segment_length : int, sequence of int or None, default=None
        Instances per segment (one value for all, or one per segment), drawn
        at random from what is left of the segment's groups. ``None`` uses
        every instance: each group is split evenly over the segments it
        appears in.
    transition_length : int, default=0
        0 = sudden changes. Otherwise the length of a gradual transition
        centred on every boundary between different segments: inside it the
        share of the new segment's instances rises linearly from 0 to 1. Must
        be >= 2 and at most the shortest segment.
    seed : int or None, default=None
        Seed for sampling and shuffling.
    name : str, default="grouped"
        Stream name.

    Returns
    -------
    InstanceStream
        ``drift_points`` are the boundaries between different segments (for
        gradual transitions: the start of the transition, which is also in
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
    Order instances by a per-instance value: incremental drift.

    Examples of ``key``: a feature (``features["vars_clauses_ratio"]``), a
    derived feature (``features.nclauses / features.nvars``), or solver
    runtimes (``performance[["a", "b"]].max(axis=1)`` with ``ascending=False``
    gives Rook et al.'s smooth streams).

    Parameters
    ----------
    features, performance : pd.DataFrame
        The dataset (same index).
    cutoff : float
        Scenario cutoff.
    key : pd.Series
        Sort value per instance (index = instance ids). Instances with a
        missing value are dropped (with a warning).
    ascending : bool, default=True
        Sort direction.
    jitter : float, default=0.0
        Standard deviation of Gaussian noise added to the rank scaled to
        [0, 1], so the order is only roughly sorted and differs per seed.
        ``0`` sorts exactly (ties broken at random).
    seed : int or None, default=None
        Seed for tie-breaking and jitter.
    name : str, default="sorted"
        Stream name.

    Returns
    -------
    InstanceStream
        No drift points (the drift has no single onset); the whole stream is
        one transition interval.
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
    Group instances by their features (KMeans), as labels for
    :func:`grouped_stream`.

    Missing values are imputed with the column median and features are
    standardised first, so no single feature dominates the distances.
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
