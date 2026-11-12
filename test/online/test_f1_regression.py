"""Behaviour must not change through the F1 interface refactoring.

Compares 9 fixed runs with the numbers recorded on the pre-F1 code (see
``regression_cases.py``). The only intended difference: ``sbs_cost`` and
``closed_gap`` now use the single best algorithm on the evaluated stream; the
old warm-start definitions live on as ``sbs_cost_warmstart`` and
``closed_gap_warmstart``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from .regression_cases import BASELINE, CASES, run_case, _jsonable

RENAMED_FOR_WARM_START = {
    "sbs_cost": "sbs_cost_warmstart",
    "closed_gap": "closed_gap_warmstart",
}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text())


@pytest.mark.parametrize("name", list(CASES))
def test_results_identical_to_pre_f1_baseline(
    name: str, baseline: dict[str, Any]
) -> None:
    old = baseline[name]
    new = _jsonable(run_case(name))
    warm = CASES[name][2] > 0

    assert new["schedules"] == old["schedules"]
    assert new["steps"] == old["steps"]
    assert new["retrain_steps"] == old["retrain_steps"]
    assert new["drift_steps"] == old["drift_steps"]
    for key, value in old["summary"].items():
        new_key = RENAMED_FOR_WARM_START.get(key, key) if warm else key
        assert _same(new["summary"][new_key], value), (key, new_key)
    if warm:
        assert new["sbs_cost_warmstart_per_step"] == old["sbs_cost_per_step"]
    else:
        assert new["sbs_cost_per_step"] == old["sbs_cost_per_step"]
