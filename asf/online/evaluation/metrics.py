"""Metrics for online evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def penalised_runtime(runtime: float, cutoff: float, par_factor: float = 10.0) -> float:
    """PAR-k score of a single run: runtime if solved, else ``cutoff * par_factor``."""
    return float(runtime) if runtime <= cutoff else float(cutoff * par_factor)


def closed_gap(
    selector_cost: float,
    sbs_cost: float,
    vbs_cost: float,
    maximize: bool = False,
) -> float:
    """Fraction of the SBS-VBS gap closed (1 = VBS, 0 = SBS).
    """
    gap = vbs_cost - sbs_cost if maximize else sbs_cost - vbs_cost
    if gap <= 0:
        return float("nan")
    improvement = selector_cost - sbs_cost if maximize else sbs_cost - selector_cost
    return float(improvement / gap)


def detection_metrics(
    true_points: Sequence[int],
    detected: Sequence[int],
    tolerance: int = 100,
) -> dict[str, float]:
    """
    Compare detected drift steps with injected drift points.

    A detection counts for a true drift point ``p`` if it falls in
    ``[p, p + tolerance]``. Detections matching no point are counted as
    ``n_false_alarms``; note that this includes repeat signals after a true
    drift once the window has passed.

    Returns
    -------
    dict
        ``mean_delay`` (NaN if nothing was detected), ``n_missed``,
        ``n_false_alarms``.
    """
    delays: list[int] = []
    missed = 0
    used: set[int] = set()
    for p in true_points:
        hits = [d for d in detected if p <= d <= p + tolerance]
        if hits:
            delays.append(min(hits) - p)
            used.update(hits)
        else:
            missed += 1
    false_alarms = len([d for d in detected if d not in used])
    return {
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "n_missed": float(missed),
        "n_false_alarms": float(false_alarms),
    }
