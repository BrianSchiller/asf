"""Prequential ("test-then-train") evaluation of online selectors."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.evaluation.metrics import (
    closed_gap,
    detection_metrics,
    penalised_runtime,
)
from asf.online.evaluation.provenance import asf_provenance, describe
from asf.online.streams.stream import InstanceStream

ResultMode = Literal["primary", "best_of_scheduled"]

#: Version of the on-disk format written by :meth:`OnlineResult.save`.
SCHEMA_VERSION = 1

#: Per-step columns and their dtypes, in file order (schema v1).
STEP_COLUMNS: dict[str, str] = {
    "step": "int64",
    "instance_id": "string",
    "primary": "string",
    "extras": "string",
    "runs": "string",
    "n_runs": "int64",
    "pred_runtime": "float64",
    "pred_score": "float64",
    "pred_uncertainty": "float64",
    "instance_score": "float64",
    "primary_runtime": "float64",
    "primary_censored": "bool",
    "cost_primary": "float64",
    "cost_best_of_scheduled": "float64",
    "user_cost": "float64",
    "compute_cost": "float64",
    "vbs_cost": "float64",
    "vbs_algorithm": "string",
    "sbs_cost": "float64",
    "sbs_cost_warmstart": "float64",
    "retrained": "bool",
    "drift_signal": "bool",
    "drift_warning": "bool",
    "detector_score": "float64",
    "drift_statistic": "float64",
    "response": "string",
    "selector_seconds": "float64",
    "detector_seconds": "float64",
}

_STEPS_FILE = "steps.parquet"
_META_FILE = "meta.json"
_PARQUET_ENGINE = "pyarrow"


@dataclass
class OnlineResult:
    """
    Outcome of one prequential run; ``save``/``load`` use the on-disk format
    in ``docs/features/online_selection.md`` (:data:`SCHEMA_VERSION`).

    ``steps`` has one row per streamed instance (:data:`STEP_COLUMNS`): the
    primary and extra runs, the cost under both result modes and as
    ``user_cost``, ``compute_cost`` (summed capped runtimes of all runs),
    VBS and SBS cost (SBS in hindsight on the evaluated stream), the
    prediction record, detector output and per-step seconds. ``meta`` is
    everything written to ``meta.json``. Detection metrics measure delay from
    the onset of a gradual change, not its completion.
    """

    steps: pd.DataFrame
    retrain_steps: list[int]
    drift_steps: list[int]
    retrain_seconds: float
    drift_points: tuple[int, ...]
    maximize: bool = False
    transition_intervals: tuple[tuple[int, int], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self, detection_tolerance: int = 100) -> dict[str, float]:
        """
        Aggregate numbers for result tables. ``mean_user_cost`` and
        ``closed_gap`` follow the result mode; both modes are reported too.
        ``closed_gap_warmstart`` uses the warm-start SBS (NaN on cold starts).
        """
        s = self.steps
        mx = self.maximize
        user = float(s["user_cost"].mean())
        sbs, vbs = float(s["sbs_cost"].mean()), float(s["vbs_cost"].mean())
        sbs_warm = float(s["sbs_cost_warmstart"].mean())
        primary = float(s["cost_primary"].mean())
        best = float(s["cost_best_of_scheduled"].mean())
        out = {
            "mean_user_cost": user,
            "mean_compute_cost": float(s["compute_cost"].mean()),
            "mean_runs_per_instance": float(s["n_runs"].mean()),
            "extra_run_rate": float((s["n_runs"] > 1).mean()),
            "sbs_cost": sbs,
            "vbs_cost": vbs,
            "closed_gap": closed_gap(user, sbs, vbs, maximize=mx),
            "mean_cost_primary": primary,
            "mean_cost_best_of_scheduled": best,
            "closed_gap_primary": closed_gap(primary, sbs, vbs, maximize=mx),
            "closed_gap_best_of_scheduled": closed_gap(best, sbs, vbs, maximize=mx),
            "sbs_cost_warmstart": sbs_warm,
            "closed_gap_warmstart": (
                closed_gap(user, sbs_warm, vbs, maximize=mx)
                if math.isfinite(sbs_warm)
                else float("nan")
            ),
            "n_retrains": float(len(self.retrain_steps)),
            "retrain_seconds": self.retrain_seconds,
            "n_drift_signals": float(len(self.drift_steps)),
            "mean_selector_seconds": float(s["selector_seconds"].mean()),
            "mean_detector_seconds": float(s["detector_seconds"].mean()),
        }
        if self.drift_points:
            out.update(
                detection_metrics(
                    self.drift_points, self.drift_steps, detection_tolerance
                )
            )
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """
        Write ``steps.parquet`` and ``meta.json`` to the directory ``path``
        (needs ``pyarrow``).
        """
        _require_parquet_engine()
        out = Path(path)
        steps_file, meta_file = out / _STEPS_FILE, out / _META_FILE
        if not overwrite and (steps_file.exists() or meta_file.exists()):
            raise FileExistsError(f"{out} already holds a result; pass overwrite=True.")
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.update(
            {
                "schema_version": SCHEMA_VERSION,
                "maximize": bool(self.maximize),
                "retrain_steps": [int(v) for v in self.retrain_steps],
                "drift_steps": [int(v) for v in self.drift_steps],
                "retrain_seconds": float(self.retrain_seconds),
                "created_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )
        stream = dict(meta.get("stream", {}))
        stream["drift_points"] = [int(v) for v in self.drift_points]
        stream["transition_intervals"] = [
            [int(a), int(b)] for a, b in self.transition_intervals
        ]
        meta["stream"] = stream
        frame = _typed_steps(self.steps)
        frame.to_parquet(steps_file, engine=_PARQUET_ENGINE, index=False)
        meta_file.write_text(
            json.dumps(describe(meta), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "OnlineResult":
        """Read a result written by :meth:`save`."""
        _require_parquet_engine()
        src = Path(path)
        meta = json.loads((src / _META_FILE).read_text(encoding="utf-8"))
        version = meta.get("schema_version")
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r} (this ASF reads up to "
                f"{SCHEMA_VERSION})."
            )
        steps = _typed_steps(pd.read_parquet(src / _STEPS_FILE, engine=_PARQUET_ENGINE))
        stream = meta.get("stream", {})
        return cls(
            steps=steps,
            retrain_steps=[int(v) for v in meta.get("retrain_steps", [])],
            drift_steps=[int(v) for v in meta.get("drift_steps", [])],
            retrain_seconds=float(meta.get("retrain_seconds", 0.0)),
            drift_points=tuple(int(v) for v in stream.get("drift_points", [])),
            maximize=bool(meta.get("maximize", False)),
            transition_intervals=tuple(
                (int(a), int(b)) for a, b in stream.get("transition_intervals", [])
            ),
            meta=meta,
        )


def run_prequential(
    selector: AbstractOnlineSelector,
    stream: InstanceStream,
    n_warmstart: int = 0,
    par_factor: float = 10.0,
    result_mode: ResultMode | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    stream_seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OnlineResult:
    """
    Evaluate an online selector on a stream, test-then-train.

    Per instance: ``propose`` -> "run" the schedule on the hidden ground truth
    (a run stops at ``min(budget, cutoff)`` and is censored if it does not
    finish before) -> score -> ``observe``. The first ``n_warmstart``
    instances are a fully observed warm start (0 = cold start, rejected for
    selectors with ``SUPPORTS_COLD_START = False``).

    The user gets the primary result (``"primary"``) or the best scheduled
    one (``"best_of_scheduled"``, parallel runs as in Rook et al.);
    ``result_mode=None`` takes the selector's ``RESULT_MODE``. Both are
    logged. ``compute_cost`` is reported separately (not on the PAR-k scale).

    SBS is the best single algorithm on the *evaluated* stream (hindsight);
    the warm-start SBS is logged as ``sbs_cost_warmstart``. VBS ties go to the
    stream's algorithm order.

    With ``selector.maximize=True`` costs are raw quality scores; this needs
    ``cutoff=inf`` and no run budgets (scores reveal no runtimes), and
    ``compute_cost`` is NaN.

    ``stream_seed`` and ``metadata`` are only stored in the result's meta.
    """
    mode = result_mode or selector.RESULT_MODE
    if mode not in ("primary", "best_of_scheduled"):
        raise ValueError(f"Unknown result_mode: {mode!r}")
    protocol = "warm" if n_warmstart > 0 else "cold"
    if protocol == "cold" and not selector.SUPPORTS_COLD_START:
        raise ValueError(
            f"{type(selector).__name__} needs a warm start (SUPPORTS_COLD_START "
            "is False); use n_warmstart >= 1."
        )
    provenance = asf_provenance()
    warm, tail = stream.split(n_warmstart)
    cutoff = stream.cutoff
    maximize = selector.maximize
    if maximize and not (math.isinf(cutoff) and cutoff > 0):
        raise ValueError(
            "Maximizing quality requires cutoff=inf: quality scores do not "
            "provide the runtimes needed to simulate a finite cutoff."
        )
    algorithms = [str(a) for a in stream.performance.columns]

    def pen(v: float) -> float:
        return float(v) if maximize else penalised_runtime(v, cutoff, par_factor)

    def best_algorithm(values: pd.Series) -> str:
        """Best entry; ties go to the first in stream order."""
        return str(values.idxmax() if maximize else values.idxmin())

    selector.fit(warm.features, warm.performance, **(fit_kwargs or {}))
    sbs_stream = best_algorithm(tail.performance.map(pen).mean())
    sbs_warm = best_algorithm(warm.performance.map(pen).mean()) if len(warm) else None

    rows: list[dict[str, Any]] = []
    retrains_seen = 0
    for step, item in enumerate(tail):
        truth = tail.true_runtimes(item.instance_id)
        t0 = time.perf_counter()
        schedule = selector.propose(item.features)
        propose_seconds = time.perf_counter() - t0

        runs: dict[str, float] = {}
        censored: set[str] = set()
        for algo, budget in schedule:
            if maximize and budget is not None and budget > 0:
                raise ValueError(
                    "Maximizing quality requires unlimited run budgets: "
                    "quality scores do not provide runtimes."
                )
            if algo in runs:
                continue
            limit = cutoff if not budget or budget <= 0 else min(budget, cutoff)
            true_rt = float(truth[algo])
            if true_rt < limit:  # runtime == limit counts as a timeout
                runs[algo] = true_rt
            else:
                runs[algo] = limit
                censored.add(algo)

        primary_algo = schedule[0][0]
        scored = {
            a: (par_factor * cutoff if a in censored else rt) for a, rt in runs.items()
        }
        cost_primary = scored[primary_algo]
        cost_best = max(scored.values()) if maximize else min(scored.values())
        truth_scores = pd.Series(
            [pen(v) for v in truth.to_numpy()], index=algorithms, dtype=float
        )
        vbs_algo = best_algorithm(truth_scores)

        t0 = time.perf_counter()
        selector.observe(
            Observation(
                instance_id=item.instance_id,
                features=item.features,
                runtimes=runs,
                censored=frozenset(censored),
                cutoff=cutoff,
            )
        )
        observe_seconds = time.perf_counter() - t0

        retrained = len(selector.retrain_steps) > retrains_seen
        retrains_seen = len(selector.retrain_steps)
        trace = selector.last_trace_
        record = trace.prediction if trace is not None else None
        signal = trace.signal if trace is not None else None
        detector_seconds = trace.detector_seconds if trace is not None else 0.0
        selector_seconds = propose_seconds + observe_seconds - detector_seconds
        rows.append(
            {
                "step": step,
                "instance_id": str(item.instance_id),
                "primary": primary_algo,
                "extras": _json([a for a, _ in schedule[1:]]),
                "runs": _json(
                    {
                        a: {"runtime": rt, "censored": a in censored}
                        for a, rt in runs.items()
                    }
                ),
                "n_runs": len(runs),
                "pred_runtime": _num(record.predicted_runtime if record else None),
                "pred_score": _num(record.score if record else None),
                "pred_uncertainty": _num(record.uncertainty if record else None),
                "instance_score": _num(trace.instance_score if trace else None),
                "primary_runtime": float(runs[primary_algo]),
                "primary_censored": primary_algo in censored,
                "cost_primary": float(cost_primary),
                "cost_best_of_scheduled": float(cost_best),
                "user_cost": float(cost_primary if mode == "primary" else cost_best),
                "compute_cost": float("nan") if maximize else float(sum(runs.values())),
                "vbs_cost": float(truth_scores[vbs_algo]),
                "vbs_algorithm": vbs_algo,
                "sbs_cost": float(truth_scores[sbs_stream]),
                "sbs_cost_warmstart": (
                    float(truth_scores[sbs_warm])
                    if sbs_warm is not None
                    else float("nan")
                ),
                "retrained": retrained,
                "drift_signal": bool(signal is not None and signal.drift),
                "drift_warning": bool(signal is not None and signal.warning),
                "detector_score": _num(signal.score if signal is not None else None),
                "drift_statistic": _num(trace.drift_statistic if trace else None),
                "response": ",".join(trace.responses) if trace else "",
                "selector_seconds": selector_seconds,
                "detector_seconds": detector_seconds,
            }
        )

    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asf": provenance,
        "selector": describe(selector),
        "stream": {
            "name": tail.name,
            "seed": stream_seed,
            "n_instances": len(stream),
            "n_evaluated": len(tail),
            "cutoff": cutoff if math.isfinite(cutoff) else None,
            "algorithms": algorithms,
            "n_features": int(stream.features.shape[1]),
            "drift_points": list(tail.drift_points),
            "transition_intervals": [list(i) for i in tail.transition_intervals],
        },
        "protocol": protocol,
        "n_warmstart": int(n_warmstart),
        "seed": selector.random_state,
        "result_mode": mode,
        "par_factor": float(par_factor),
        "maximize": bool(maximize),
        "sbs_algorithm_stream": sbs_stream,
        "sbs_algorithm_warmstart": sbs_warm,
        "extra": describe(dict(metadata or {})),
    }
    return OnlineResult(
        steps=_typed_steps(pd.DataFrame(rows, columns=list(STEP_COLUMNS))),
        retrain_steps=list(selector.retrain_steps),
        drift_steps=list(selector.drift_steps),
        retrain_seconds=selector.retrain_seconds,
        drift_points=tail.drift_points,
        maximize=maximize,
        transition_intervals=tail.transition_intervals,
        meta=meta,
    )


def _num(value: float | None) -> float:
    """``None`` -> NaN, everything else -> float."""
    return float("nan") if value is None else float(value)


def _json(value: Any) -> str:
    """Compact JSON with non-finite numbers as ``null``."""
    return json.dumps(describe(value), separators=(",", ":"), allow_nan=False)


def _typed_steps(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns in schema order with schema dtypes (strings as ``object``)."""
    out = frame.copy()
    for column, dtype in STEP_COLUMNS.items():
        if column not in out:
            continue
        if dtype == "string":
            out[column] = out[column].astype(object).map(str)
        elif dtype == "bool":
            out[column] = out[column].astype(bool)
        else:
            out[column] = out[column].astype(dtype)
    ordered = [c for c in STEP_COLUMNS if c in out] + [
        c for c in out if c not in STEP_COLUMNS
    ]
    return out[ordered].reset_index(drop=True)


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Saving and loading online results needs pyarrow: "
            "pip install 'asf[online]'."
        ) from exc


__all__ = ["OnlineResult", "run_prequential", "SCHEMA_VERSION", "STEP_COLUMNS"]
