"""Fixed runs whose results must not change when the online interfaces evolve.

The reference numbers in ``data/f1_regression_baseline.json`` were recorded on
commit ``3083aa0`` (branch ``online-selection``), before the F1 interface
refactoring. Regenerate them only on purpose::

    python -m test.online.regression_cases --write
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from asf.online.base import AbstractOnlineSelector
from asf.online.drift import PCAReconstructionDetector
from asf.online.evaluation import run_prequential
from asf.online.retraining import PeriodicRetrain
from asf.online.selectors import (
    DiscountedTSSelector,
    LinUCBSelector,
    OnlinePairwiseSelector,
)
from asf.online.streams import InstanceStream, make_synthetic_scenario

BASELINE = Path(__file__).parent / "data" / "f1_regression_baseline.json"
STEP_COLUMNS = (
    "primary",
    "n_runs",
    "user_cost",
    "compute_cost",
    "vbs_cost",
    "retrained",
)


def runtime_stream() -> InstanceStream:
    """Sudden covariate shift, 600 instances, 3 algorithms, cutoff 100."""
    return make_synthetic_scenario("sudden_covariate", random_state=0)


def quality_stream() -> InstanceStream:
    """The same instances as a maximisation problem (quality = -log runtime)."""
    s = runtime_stream()
    quality = -np.log(s.performance.clip(lower=1e-3))
    return InstanceStream(
        s.features,
        quality,
        cutoff=float("inf"),
        drift_points=s.drift_points,
        name="synthetic_quality",
        transition_intervals=s.transition_intervals,
    )


def _pca() -> PCAReconstructionDetector:
    return PCAReconstructionDetector(chunk_size=50, random_state=0)


#: name -> (selector factory, stream factory, n_warmstart)
CASES: dict[
    str, tuple[Callable[[], AbstractOnlineSelector], Callable[[], Any], int]
] = {
    "rook_paper_detector_cold": (
        lambda: OnlinePairwiseSelector(
            reference_period=100, detector=_pca(), random_state=0
        ),
        runtime_stream,
        0,
    ),
    "rook_paper_detector_warm": (
        lambda: OnlinePairwiseSelector(detector=_pca(), random_state=0),
        runtime_stream,
        100,
    ),
    "rook_paper_periodic_cold": (
        lambda: OnlinePairwiseSelector(
            reference_period=100,
            retrain_policy=PeriodicRetrain(50),
            additional_strategy="uniform",
            random_state=0,
        ),
        runtime_stream,
        0,
    ),
    "rook_code_detector_cold": (
        lambda: OnlinePairwiseSelector(
            variant="code",
            reference_period=100,
            detector=_pca(),
            maximize=True,
            random_state=0,
        ),
        quality_stream,
        0,
    ),
    "rook_code_cost_based_warm": (
        lambda: OnlinePairwiseSelector(
            variant="code",
            additional_strategy="cost_based",
            retrain_policy=PeriodicRetrain(50),
            maximize=True,
            random_state=0,
        ),
        quality_stream,
        100,
    ),
    "ds_ts_cold": (
        lambda: DiscountedTSSelector(gamma=0.98, budget=100.0, random_state=0),
        runtime_stream,
        0,
    ),
    "ds_ts_warm": (
        lambda: DiscountedTSSelector(gamma=0.98, budget=100.0, random_state=0),
        runtime_stream,
        100,
    ),
    "linucb_cold": (
        lambda: LinUCBSelector(budget=100.0, random_state=0),
        runtime_stream,
        0,
    ),
    "linucb_warm": (
        lambda: LinUCBSelector(budget=100.0, random_state=0),
        runtime_stream,
        100,
    ),
}


def run_case(name: str) -> dict[str, Any]:
    """Run one case and collect everything that must stay identical."""
    make_selector, make_stream, n_warm = CASES[name]
    selector = make_selector()
    schedules: list[list[list[Any]]] = []
    propose = selector.propose

    def recording_propose(features: Any) -> list[tuple[str, float]]:
        schedule = propose(features)
        schedules.append([[str(a), float(b)] for a, b in schedule])
        return schedule

    selector.propose = recording_propose  # type: ignore[method-assign]
    result = run_prequential(selector, make_stream(), n_warmstart=n_warm)
    summary = {k: v for k, v in result.summary().items() if k != "retrain_seconds"}
    steps = result.steps
    return {
        "summary": summary,
        "sbs_cost_per_step": [float(v) for v in steps["sbs_cost"]],
        **(
            {
                "sbs_cost_warmstart_per_step": [
                    float(v) for v in steps["sbs_cost_warmstart"]
                ]
            }
            if "sbs_cost_warmstart" in steps
            else {}
        ),
        "steps": {c: steps[c].tolist() for c in STEP_COLUMNS},
        "schedules": schedules,
        "retrain_steps": list(result.retrain_steps),
        "drift_steps": list(result.drift_steps),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) else v
    return obj


def compute_all() -> dict[str, Any]:
    """Results of every case, JSON-ready (NaN as ``None``)."""
    return {name: _jsonable(run_case(name)) for name in CASES}


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("Pass --write to overwrite the stored baseline.")
    BASELINE.write_text(json.dumps(compute_all(), separators=(",", ":")) + "\n")
    print(f"wrote {BASELINE}")
