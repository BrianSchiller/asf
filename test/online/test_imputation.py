"""Tests for censored-runtime imputation (Tornede et al., 2022).

Covers the imputation rules, their use in the bandits (rewards) and in the
online EPM selector (training targets).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from asf.online.imputation import (
    CutoffImputer,
    DropCensored,
    ExpectedImputer,
    SampledImputer,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ================================================================== rules
def test_cutoff_returns_bound() -> None:
    out = CutoffImputer()([1.0, 2.5], [10.0, -3.0], [1.0, 0.0], _rng())
    np.testing.assert_array_equal(out, [1.0, 2.5])


def test_drop_returns_nan() -> None:
    out = DropCensored()([1.0, 2.0, 3.0], 0.0, 1.0, _rng())
    assert out.shape == (3,) and np.isnan(out).all()


@pytest.mark.parametrize(
    "bound, mean, sd, expected",
    [
        # phi(0) / (1 - Phi(0)) = 0.398942 / 0.5
        (0.0, 0.0, 1.0, 0.7978845608),
        # phi(1) / (1 - Phi(1)) = 0.241971 / 0.158655
        (1.0, 0.0, 1.0, 1.5251352762),
        # 2 + phi(2) / Phi(2) = 2 + 0.053991 / 0.977250
        (0.0, 2.0, 1.0, 2.0552478627),
        # scaling: bound 10, mean 6, sd 4 -> 6 + 4 * 1.525135
        (10.0, 6.0, 4.0, 6.0 + 4.0 * 1.5251352762),
    ],
)
def test_expected_hand_computed(
    bound: float, mean: float, sd: float, expected: float
) -> None:
    out = ExpectedImputer()(bound, mean, sd, _rng())
    assert out[0] == pytest.approx(expected, rel=1e-9)


def test_expected_edge_cases() -> None:
    imp = ExpectedImputer()
    # sd == 0: no uncertainty -> max(mean, bound)
    np.testing.assert_array_equal(imp([0.0, 5.0], [3.0, 3.0], 0.0, _rng()), [3.0, 5.0])
    # sd == inf: no information -> bound
    assert imp(2.0, 0.0, np.inf, _rng())[0] == 2.0
    # far tail stays finite and close to bound + sd^2 / (bound - mean)
    far = imp(40.0, 0.0, 1.0, _rng())[0]
    assert math.isfinite(far) and 40.0 < far < 40.0 + 1.0 / 40.0 + 1e-9


def test_expected_never_below_bound_and_deterministic() -> None:
    rng = _rng(1)
    bound, mean, sd = rng.normal(size=200), rng.normal(size=200) * 5, rng.random(200)
    a = ExpectedImputer()(bound, mean, sd, _rng(0))
    b = ExpectedImputer()(bound, mean, sd, _rng(99))
    assert np.all(a >= bound)
    np.testing.assert_array_equal(a, b)


def test_expected_rejects_negative_sd() -> None:
    with pytest.raises(ValueError):
        ExpectedImputer()(0.0, 0.0, -1.0, _rng())


def test_sampled_above_bound_and_seeded() -> None:
    bound = np.zeros(500)
    a = SampledImputer()(bound, 0.0, 1.0, _rng(3))
    b = SampledImputer()(bound, 0.0, 1.0, _rng(3))
    np.testing.assert_array_equal(a, b)
    assert np.all(a > 0.0)
    assert not np.array_equal(a, SampledImputer()(bound, 0.0, 1.0, _rng(4)))


def test_sampled_falls_back_to_bound() -> None:
    # Mean 100 sd below the bound: no draw is accepted in 20 tries.
    out = SampledImputer()([0.0, 1.0], -100.0, 1.0, _rng())
    np.testing.assert_array_equal(out, [0.0, 1.0])
    # One try with P(accept) = 1/2: some entries fall back, the rest are above.
    out = SampledImputer(max_tries=1)(np.zeros(1000), 0.0, 1.0, _rng())
    fallback = out == 0.0
    assert 400 < fallback.sum() < 600 and np.all(out[~fallback] > 0.0)


def test_sampled_mean_matches_truncated_mean() -> None:
    # With enough tries the accepted draws follow N(0, 1) truncated at 0.5.
    out = SampledImputer(max_tries=200)(np.full(20000, 0.5), 0.0, 1.0, _rng(7))
    truncated_mean = norm.pdf(0.5) / norm.sf(0.5)
    assert out.mean() == pytest.approx(truncated_mean, abs=0.02)


def test_sampled_edge_cases() -> None:
    imp = SampledImputer()
    assert imp(1.0, 0.0, np.inf, _rng())[0] == 1.0  # no information
    np.testing.assert_array_equal(imp([0.0, 5.0], 3.0, 0.0, _rng()), [3.0, 5.0])
    with pytest.raises(ValueError):
        SampledImputer(max_tries=0)


def test_rule_flags() -> None:
    assert not CutoffImputer.USES_MODEL and not DropCensored.USES_MODEL
    assert ExpectedImputer.USES_MODEL and ExpectedImputer.PREDICTIVE
    assert SampledImputer.USES_MODEL and not SampledImputer.PREDICTIVE
