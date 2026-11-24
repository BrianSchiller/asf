import math
from pathlib import Path

import numpy as np
import pytest

from asf.online.streams.loaders import load_bin_packing
from asf.online.streams.loaders.bin_packing import (
    HEURISTICS,
    falkenauer,
    pack,
    rook_features,
)

TREND = 'ts__agg_linear_trend__attr_"intercept"__chunk_len_{}__f_agg_"mean"'


@pytest.mark.parametrize(
    "items, expected",
    [
        # BF takes the fullest bin, FF and WF the first / emptiest one
        ([5, 7, 3], {"FF": [8, 7], "BF": [5, 10], "WF": [8, 7], "NF": [5, 10]}),
        ([7, 5, 2], {"FF": [9, 5], "BF": [9, 5], "WF": [7, 7], "NF": [7, 7]}),
        # NF closes the first bin when 5 does not fit, FF returns to it
        ([6, 5, 4, 3], {"FF": [10, 8], "BF": [10, 8], "WF": [9, 9], "NF": [6, 9, 3]}),
        # ties go to the earliest opened bin
        ([6, 6, 2], {"FF": [8, 6], "BF": [8, 6], "WF": [8, 6], "NF": [6, 8]}),
    ],
)
def test_pack(items: list[int], expected: dict[str, list[int]]) -> None:
    for rule, fills in expected.items():
        assert pack(items, 10, rule) == fills, rule


def test_pack_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError):
        pack([1], 10, "XF")


def test_falkenauer() -> None:
    assert falkenauer([10, 5], 10) == pytest.approx((1 + 0.25) / 2)
    assert falkenauer([10, 10], 10) == 1.0


def test_rook_features_by_hand() -> None:
    f = rook_features(np.arange(1, 11))
    c3 = np.mean([i * (i + 1) * (i + 2) for i in range(1, 9)])
    assert len(f) == 10
    assert f["ts__sum_values"] == 55
    assert f["ts__skewness"] == pytest.approx(0)
    assert f["ts__c3__lag_1"] == pytest.approx(c3)
    assert f["ts__autocorrelation__lag_1"] == f["ts__partial_autocorrelation__lag_1"]
    assert f['ts__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0'] == 1.0
    assert f[TREND.format(5)] == pytest.approx(3.0)  # chunk means 3 and 8
    assert math.isnan(f[TREND.format(50)])  # a single chunk has no trend
    short = rook_features(np.array([1, 2, 3, 4]))
    assert short["ts__autocorrelation__lag_1"] == pytest.approx(1 / 3)


def _write(path: Path, items: list[int], capacity: int = 10) -> None:
    lines = [len(items), capacity, *items]
    path.write_bytes("\r\n".join(map(str, lines)).encode() + b"\r\n")


def test_load_bin_packing(tmp_path: Path) -> None:
    items = {"X_BF0001": [5, 7, 3] * 4, "X_WF0000": [7, 5, 2] * 4}
    for name, sizes in items.items():
        _write(tmp_path / f"{name}.bpp", sizes)
    features, performance, cutoff = load_bin_packing(str(tmp_path))
    assert cutoff == math.inf
    assert list(performance.index) == ["X_BF0001", "X_WF0000"]
    assert list(performance.columns) == list(HEURISTICS)
    assert features.index.equals(performance.index) and features.shape == (2, 10)
    for name, sizes in items.items():
        for h in HEURISTICS:
            expected = falkenauer(pack(sizes, 10, h), 10)
            assert performance.loc[name, h] == pytest.approx(expected)


@pytest.mark.parametrize("values", [[3, 10, 5, 7], [2, 10, 5, 11], []])
def test_load_bin_packing_rejects_bad_files(tmp_path: Path, values: list[int]) -> None:
    (tmp_path / "bad.bpp").write_text("\n".join(map(str, values)))
    with pytest.raises(ValueError):
        load_bin_packing(str(tmp_path))
