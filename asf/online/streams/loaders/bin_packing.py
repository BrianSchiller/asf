"""
Online 1-D bin packing (Alissa et al. 2019; Rook et al. 2025).

A ``.bpp`` file holds the number of items, the bin capacity and the item
sizes, one value per line. Items are packed strictly in order by one of four
heuristics, and a packing is scored with Falkenauer's fitness (higher is
better, 1 = all bins full). Features are the 10 tsfresh features Rook et al.
selected, computed with numpy (they match their ``BP_selected_features.csv``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

HEURISTICS = ("BF", "FF", "NF", "WF")


def read_bpp(path: str | Path) -> tuple[np.ndarray, int]:
    """Item sizes and bin capacity of one ``.bpp`` file."""
    values = [int(v) for v in Path(path).read_text().split()]
    if len(values) < 2 or len(values) - 2 != values[0]:
        raise ValueError(f"{path}: header announces a different number of items.")
    items, capacity = np.array(values[2:]), values[1]
    if len(items) == 0 or items.min() <= 0 or items.max() > capacity:
        raise ValueError(f"{path}: item sizes must lie in (0, capacity].")
    return items, capacity


def pack(items: Sequence[float], capacity: float, rule: str) -> list[float]:
    """
    Bin fills after packing ``items`` in order. ``NF`` only tries the last
    bin; ``FF``/``BF``/``WF`` take the first / fullest / emptiest feasible
    bin (ties: the earliest opened). No feasible bin opens a new one.
    """
    if rule not in HEURISTICS:
        raise ValueError(f"Unknown heuristic {rule!r}; use one of {HEURISTICS}.")
    fills: list[float] = []
    for size in items:
        if rule == "NF":
            fits = fills and fills[-1] + size <= capacity
            feasible = [len(fills) - 1] if fits else []
        else:
            feasible = [i for i, f in enumerate(fills) if f + size <= capacity]
        if not feasible:
            fills.append(size)
            continue
        if rule == "BF":
            i = max(feasible, key=lambda j: fills[j])  # max keeps the first on ties
        elif rule == "WF":
            i = min(feasible, key=lambda j: fills[j])
        else:
            i = feasible[0]
        fills[i] += size
    return fills


def falkenauer(fills: Sequence[float], capacity: float, k: float = 2.0) -> float:
    """Mean of ``(fill / capacity) ** k`` over all bins."""
    return float(np.mean((np.asarray(fills, float) / capacity) ** k))


def rook_features(items: np.ndarray) -> dict[str, float]:
    """Rook et al.'s 10 tsfresh features of the item sequence (tsfresh names)."""
    x = np.asarray(items, float)

    def c3(lag: int) -> float:
        return float(np.mean(x[2 * lag :] * x[lag:-lag] * x[: -2 * lag]))

    def trend_intercept(chunk: int) -> float:
        means = [x[i : i + chunk].mean() for i in range(0, len(x), chunk)]
        if len(means) < 2:
            return float("nan")
        return float(np.polyfit(np.arange(len(means)), means, 1)[1])

    # tsfresh autocorrelation; statsmodels' PACF at lag 1 gives the same value
    autocorr = float(
        np.sum((x[:-1] - x.mean()) * (x[1:] - x.mean())) / ((len(x) - 1) * x.var())
    )
    # mean |change| between consecutive items that both lie in the 0-40% quantile band
    low, high = np.quantile(x, [0.0, 0.4])
    inside = (x >= low) & (x <= high)
    both = inside[1:] & inside[:-1]
    changes = np.abs(np.diff(x))[both]

    trend = 'ts__agg_linear_trend__attr_"intercept"__chunk_len_{}__f_agg_"mean"'
    return {
        "ts__skewness": float(pd.Series(x).skew()),
        "ts__c3__lag_1": c3(1),
        "ts__sum_values": float(x.sum()),
        'ts__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0': (
            float(changes.mean()) if len(changes) else 0.0
        ),
        "ts__c3__lag_2": c3(2),
        trend.format(50): trend_intercept(50),
        trend.format(10): trend_intercept(10),
        trend.format(5): trend_intercept(5),
        "ts__partial_autocorrelation__lag_1": autocorr,
        "ts__autocorrelation__lag_1": autocorr,
    }


def load_bin_packing(
    path: str, k: float = 2.0
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    ``(features, performance, cutoff)`` of all ``*.bpp`` files in the folder
    ``path`` (index: file name without extension). ``performance`` is the
    Falkenauer fitness of each heuristic, so selectors need
    ``maximize=True``; ``cutoff`` is ``inf`` (nothing is censored).
    """
    files = sorted(Path(path).glob("*.bpp"))
    if not files:
        raise ValueError(f"No .bpp files in {path}.")
    features, performance = [], []
    for file in files:
        items, capacity = read_bpp(file)
        features.append(rook_features(items))
        performance.append(
            {h: falkenauer(pack(items, capacity, h), capacity, k) for h in HEURISTICS}
        )
    index = pd.Index([f.stem for f in files], name="instance_id")
    return (
        pd.DataFrame(features, index=index),
        pd.DataFrame(performance, index=index, columns=list(HEURISTICS)),
        float("inf"),
    )
