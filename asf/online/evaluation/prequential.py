"""Prequential ("test-then-train") evaluation of online selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.evaluation.metrics import (
    closed_gap,
    detection_metrics,
    penalised_runtime,
)
from asf.online.streams.stream import InstanceStream

ResultMode = Literal["primary", "best_of_scheduled"]


@dataclass
class OnlineResult:
    """
    Outcome of one prequential run.

    Attributes
    ----------
    steps : pd.DataFrame
        One row per streamed instance. Columns: ``primary`` (selected
        algorithm), ``n_runs``, ``user_cost`` (PAR-k of what the user gets),
        ``compute_cost`` (summed, capped runtimes of ALL executed algorithms),
        ``vbs_cost``, ``sbs_cost``, ``retrained``.
    retrain_steps, drift_steps : list[int]
        Stream positions (0-based) of retrainings / drift signals.
    retrain_seconds : float
        Total wall-clock time the selector spent retraining.
    drift_points : tuple[int, ...]
        Injected drift points of the stream (ground truth).
    """

    steps: pd.DataFrame
    retrain_steps: list[int]
    drift_steps: list[int]
    retrain_seconds: float
    drift_points: tuple[int, ...]

    def summary(self, detection_tolerance: int = 100) -> dict[str, float]:
        """Aggregate numbers for result tables."""
        s = self.steps
        user = float(s["user_cost"].mean())
        sbs, vbs = float(s["sbs_cost"].mean()), float(s["vbs_cost"].mean())
        out = {
            "mean_user_cost": user,
            "mean_compute_cost": float(s["compute_cost"].mean()),
            "mean_runs_per_instance": float(s["n_runs"].mean()),
            "sbs_cost": sbs,
            "vbs_cost": vbs,
            "closed_gap": closed_gap(user, sbs, vbs),
            "n_retrains": float(len(self.retrain_steps)),
            "retrain_seconds": self.retrain_seconds,
            "n_drift_signals": float(len(self.drift_steps)),
        }
        if self.drift_points:
            out.update(
                detection_metrics(
                    self.drift_points, self.drift_steps, detection_tolerance
                )
            )
        return out


def run_prequential(
    selector: AbstractOnlineSelector,
    stream: InstanceStream,
    n_warmstart: int = 0,
    par_factor: float = 10.0,
    result_mode: ResultMode | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> OnlineResult:
    """
    Evaluate an online selector on a stream, test-then-train.

    Protocol per instance:

    1. ``selector.propose(features)`` -> schedule.
    2. The evaluator "runs" the scheduled algorithms by looking up the hidden
       ground truth. A run stops at ``min(budget, cutoff)``; if the true runtime
       exceeds that, the run is *censored*. (For non-runtime objectives such as
       bin-packing quality use ``cutoff=inf``; nothing is ever censored.)
    3. Scoring, then ``selector.observe(...)``.

    Scoring: ``user_cost`` is the PAR-k score of what the user receives:

    * ``"primary"``: the first scheduled algorithm's result.
    * ``"best_of_scheduled"``: the best result among all scheduled algorithms
      (parallel runs, as in Rook et al.).

    ``compute_cost`` is always the summed capped runtime of all executed
    algorithms, i.e. the resources spent. It is reported separately because it
    is not on the same scale as PAR-k.

    The default mode is the selector's ``RESULT_MODE``.

    SBS is the single algorithm with the best mean PAR-k on the warm-start
    data, or on the evaluated stream itself for a cold start
    (``n_warmstart=0``). VBS is the per-instance oracle.

    Parameters
    ----------
    selector : AbstractOnlineSelector
        Unfitted selector; it is fitted on the warm-start prefix here (on empty
        frames carrying only the algorithm and feature names for a cold start).
    stream : InstanceStream
        Full stream; the first ``n_warmstart`` instances are the warm start.
    n_warmstart : int, default=0
        Size of the fully observed warm-start prefix; 0 = cold start.
    par_factor : float, default=10.0
        PAR-k factor for unsolved runs.
    result_mode : {"primary", "best_of_scheduled"} or None
        Scoring mode; ``None`` uses ``selector.RESULT_MODE``.
    fit_kwargs : dict or None
        Extra keyword arguments for ``selector.fit``.
    """
    mode = result_mode or selector.RESULT_MODE
    if mode not in ("primary", "best_of_scheduled"):
        raise ValueError(f"Unknown result_mode: {mode!r}")
    warm, tail = stream.split(n_warmstart)
    cutoff = stream.cutoff

    def pen(v: float) -> float:
        return penalised_runtime(v, cutoff, par_factor)

    selector.fit(warm.features, warm.performance, **(fit_kwargs or {}))
    reference = warm if len(warm) else tail
    sbs_algo = reference.performance.map(pen).mean().idxmin()

    rows: list[dict[str, Any]] = []
    retrains_seen = 0
    for item in tail:
        truth = tail.true_runtimes(item.instance_id)
        schedule = selector.propose(item.features)

        runs: dict[str, float] = {}
        censored: set[str] = set()
        for algo, budget in schedule:
            if algo in runs:
                continue
            limit = cutoff if not budget or budget <= 0 else min(budget, cutoff)
            true_rt = float(truth[algo])
            if true_rt <= limit:
                runs[algo] = true_rt
            else:
                runs[algo] = limit
                censored.add(algo)

        primary_algo = schedule[0][0]
        scored = {
            a: (par_factor * cutoff if a in censored else rt) for a, rt in runs.items()
        }
        user_cost = scored[primary_algo] if mode == "primary" else min(scored.values())

        selector.observe(
            Observation(
                instance_id=item.instance_id,
                features=item.features,
                runtimes=runs,
                censored=frozenset(censored),
                cutoff=cutoff,
            )
        )

        retrained = len(selector.retrain_steps) > retrains_seen
        retrains_seen = len(selector.retrain_steps)
        rows.append(
            {
                "instance_id": item.instance_id,
                "primary": primary_algo,
                "n_runs": len(runs),
                "user_cost": float(user_cost),
                "compute_cost": float(sum(runs.values())),
                "vbs_cost": min(pen(v) for v in truth),
                "sbs_cost": pen(truth[sbs_algo]),
                "retrained": retrained,
            }
        )

    return OnlineResult(
        steps=pd.DataFrame(rows),
        retrain_steps=list(selector.retrain_steps),
        drift_steps=list(selector.drift_steps),
        retrain_seconds=selector.retrain_seconds,
        drift_points=tail.drift_points,
    )
