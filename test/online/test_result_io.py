"""Result schema v1: OnlineResult.save / OnlineResult.load and per-step columns."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asf.online.drift import PCAReconstructionDetector
from asf.online.evaluation import (
    SCHEMA_VERSION,
    STEP_COLUMNS,
    OnlineResult,
    run_prequential,
)
from asf.online.selectors import OnlinePairwiseSelector, ThompsonSamplingSelector
from asf.online.streams.stream import InstanceStream

from .conftest import make_stream

pq = pytest.importorskip("pyarrow.parquet")

META_KEYS = {
    "schema_version",
    "asf",
    "selector",
    "stream",
    "protocol",
    "n_warmstart",
    "seed",
    "result_mode",
    "par_factor",
    "maximize",
    "sbs_algorithm_stream",
    "sbs_algorithm_warmstart",
    "retrain_steps",
    "drift_steps",
    "retrain_seconds",
    "created_utc",
    "extra",
}
STREAM_KEYS = {
    "name",
    "seed",
    "n_instances",
    "n_evaluated",
    "cutoff",
    "algorithms",
    "n_features",
    "drift_points",
    "transition_intervals",
}


def _rook_result(n_warmstart: int = 0) -> OnlineResult:
    selector = OnlinePairwiseSelector(
        reference_period=30,
        detector=PCAReconstructionDetector(chunk_size=10, random_state=0),
        random_state=3,
    )
    return run_prequential(
        selector,
        make_stream(),
        n_warmstart=n_warmstart,
        stream_seed=0,
        metadata={"run": "io-test", "lr": float("nan")},
    )


def test_roundtrip_is_lossless(tmp_path: Path) -> None:
    result = _rook_result()
    out = result.save(tmp_path / "run")
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "steps.parquet"]

    loaded = OnlineResult.load(out)
    pd.testing.assert_frame_equal(loaded.steps, result.steps)
    assert loaded.retrain_steps == result.retrain_steps
    assert loaded.drift_steps == result.drift_steps
    assert loaded.drift_points == result.drift_points
    assert loaded.transition_intervals == result.transition_intervals
    assert loaded.maximize == result.maximize
    a, b = loaded.summary(), result.summary()
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a)


def test_file_schema(tmp_path: Path) -> None:
    out = _rook_result().save(tmp_path / "run")
    schema = pq.read_schema(out / "steps.parquet")
    assert schema.names == list(STEP_COLUMNS)
    arrow = {"string": "string", "int64": "int64", "float64": "double", "bool": "bool"}
    for name, dtype in STEP_COLUMNS.items():
        assert str(schema.field(name).type) == arrow[dtype], name

    text = (out / "meta.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    meta = json.loads(text)
    assert set(meta) == META_KEYS
    assert set(meta["stream"]) == STREAM_KEYS
    assert meta["schema_version"] == SCHEMA_VERSION == 1
    assert set(meta["asf"]) == {"version", "commit", "dirty", "commit_source"}
    assert meta["selector"]["class"] == (
        "asf.online.selectors.online_pairwise.OnlinePairwiseSelector"
    )
    params = meta["selector"]["params"]
    assert params["reference_period"] == 30 and params["random_state"] == 3
    assert params["detector"]["class"].endswith("PCAReconstructionDetector")
    assert params["detector"]["params"]["chunk_size"] == 10
    assert meta["protocol"] == "cold" and meta["n_warmstart"] == 0
    assert meta["seed"] == 3
    assert meta["stream"]["seed"] == 0
    assert meta["stream"]["n_instances"] == meta["stream"]["n_evaluated"] == 120
    assert meta["stream"]["algorithms"] == ["a", "b"]
    assert meta["sbs_algorithm_warmstart"] is None
    assert meta["sbs_algorithm_stream"] in ("a", "b")
    assert meta["result_mode"] == "best_of_scheduled"
    assert meta["extra"] == {"run": "io-test", "lr": None}
    assert meta["created_utc"].endswith("Z")


def test_warm_start_meta_and_sbs_columns(tmp_path: Path) -> None:
    result = _rook_result(n_warmstart=40)
    meta = json.loads((result.save(tmp_path / "w") / "meta.json").read_text())
    assert meta["protocol"] == "warm" and meta["n_warmstart"] == 40
    assert meta["stream"]["n_evaluated"] == 80
    assert meta["stream"]["drift_points"] == [40]  # 80 in the full stream
    assert meta["sbs_algorithm_warmstart"] in ("a", "b")
    assert result.steps["sbs_cost_warmstart"].notna().all()


def test_runs_extras_and_costs_columns() -> None:
    s = _rook_result().steps
    for _, row in s.iterrows():
        runs = json.loads(row["runs"])
        extras = json.loads(row["extras"])
        assert list(runs) == [row["primary"], *extras]
        assert row["n_runs"] == len(runs)
        assert runs[row["primary"]]["runtime"] == row["primary_runtime"]
        assert runs[row["primary"]]["censored"] == row["primary_censored"]
        assert row["cost_best_of_scheduled"] <= row["cost_primary"]
        assert row["user_cost"] == row["cost_best_of_scheduled"]
    assert (s["step"] == np.arange(len(s))).all()
    assert (s["selector_seconds"] >= 0).all() and (s["detector_seconds"] >= 0).all()
    # the detector runs after the 30-instance reference period
    assert (s["detector_seconds"].iloc[31:] > 0).all()


def test_vbs_algorithm_ties_follow_algorithm_order() -> None:
    features = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    performance = pd.DataFrame(
        {"x": [5.0, 1.0, 3.0], "y": [5.0, 2.0, 1.0], "z": [9.0, 1.0, 3.0]}
    )
    stream = InstanceStream(features, performance, cutoff=100.0)
    sel = ThompsonSamplingSelector(budget=100.0, random_state=0)
    s = run_prequential(sel, stream).steps
    assert s["vbs_algorithm"].tolist() == ["x", "x", "y"]
    assert s["vbs_cost"].tolist() == [5.0, 1.0, 1.0]


def test_infinite_cutoff_is_null_in_meta(tmp_path: Path) -> None:
    stream = make_stream()
    stream.cutoff = float("inf")
    result = run_prequential(
        ThompsonSamplingSelector(random_state=0), stream, n_warmstart=10
    )
    meta = json.loads((result.save(tmp_path / "r") / "meta.json").read_text())
    assert meta["stream"]["cutoff"] is None


def test_save_refuses_to_overwrite_and_load_rejects_newer_schema(
    tmp_path: Path,
) -> None:
    result = _rook_result()
    out = result.save(tmp_path / "r")
    with pytest.raises(FileExistsError):
        result.save(out)
    result.save(out, overwrite=True)

    meta = json.loads((out / "meta.json").read_text())
    meta["schema_version"] = SCHEMA_VERSION + 1
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema_version"):
        OnlineResult.load(out)
