"""Tests for the PCA reconstruction drift detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asf.online.drift import PCAReconstructionDetector
from asf.online.evaluation import run_prequential
from asf.online.selectors import OnlinePairwiseSelector
from asf.online.streams.stream import InstanceStream

D = 10


def sample(
    n: int, seed: int, noise: float = 0.3, shift: np.ndarray | None = None
) -> pd.DataFrame:
    """3 latent factors mixed into D correlated features plus noise."""
    mix = np.random.default_rng(123).normal(size=(3, D))
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) @ mix + noise * rng.normal(size=(n, D))
    if shift is not None:
        x = x + shift
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(D)])


OFF_SUBSPACE = np.r_[np.zeros(D - 2), 3.0, 3.0]


def feed(det: PCAReconstructionDetector, data: pd.DataFrame) -> list[int]:
    return [i for i in range(len(data)) if det.update(data.iloc[[i]]).drift]


def test_fit_sets_thresholds_and_components() -> None:
    det = PCAReconstructionDetector(random_state=0).fit(sample(200, 0))
    assert 1 <= det.n_components_ < D
    assert 0 <= det.lower_threshold_ < det.reference_error_ < det.upper_threshold_
    with pytest.raises(ValueError):
        PCAReconstructionDetector().fit(sample(1, 0))


def test_signals_only_when_a_chunk_completes() -> None:
    det = PCAReconstructionDetector(chunk_size=20, random_state=0).fit(sample(200, 0))
    signals = [det.update(r.to_frame().T) for _, r in sample(19, 1).iterrows()]
    assert all(not s.drift and s.score is None for s in signals)
    last = det.update(sample(1, 2))
    assert last.score is not None  # 20th instance completes the first chunk


def test_stationary_data_rarely_alarms() -> None:
    det = PCAReconstructionDetector(
        reference_errors="cross_validated", random_state=0
    ).fit(sample(400, 0))
    assert len(feed(det, sample(1000, 1))) <= 1  # 20 chunks, ~1% false alarm each


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shift": OFF_SUBSPACE},  # data leaves the principal subspace
        {"noise": 1.2},  # larger spread around the subspace
    ],
)
def test_drift_is_detected_within_two_chunks(kwargs: dict) -> None:
    det = PCAReconstructionDetector(chunk_size=50, random_state=0).fit(sample(300, 0))
    pre = sample(120, 2)
    post = sample(400, 3, **kwargs)
    hits = feed(det, pd.concat([pre, post], ignore_index=True))
    after = [h for h in hits if h >= len(pre)]
    assert after and after[0] - len(pre) <= 2 * 50


def test_lower_error_alerts_only_when_two_sided() -> None:
    ref, calmer = sample(300, 0), sample(300, 1, noise=0.02)
    two = PCAReconstructionDetector(chunk_size=50, random_state=0).fit(ref)
    one = PCAReconstructionDetector(chunk_size=50, two_sided=False, random_state=0).fit(
        ref
    )
    assert feed(two, calmer) and not feed(one, calmer)


def test_instance_score_is_higher_off_the_subspace() -> None:
    det = PCAReconstructionDetector(random_state=0).fit(sample(300, 0))
    on, off = det.score(sample(100, 1)), det.score(sample(100, 1, shift=OFF_SUBSPACE))
    assert on.shape == (100,) and off.mean() > 2 * on.mean()


def test_reset_refits_on_new_reference_and_clears_state() -> None:
    det = PCAReconstructionDetector(chunk_size=50, random_state=0).fit(sample(300, 0))
    drifted = sample(300, 1, shift=OFF_SUBSPACE)
    assert feed(det, drifted)
    det.reset(drifted)  # new reference = the drifted regime
    assert det.last_chunk_error_ is None
    assert len(feed(det, sample(300, 2, shift=OFF_SUBSPACE))) <= 1


def test_stride_and_validation() -> None:
    det = PCAReconstructionDetector(chunk_size=50, stride=10, random_state=0)
    det.fit(sample(300, 0))
    scored = [det.update(sample(1, i)).score for i in range(60)]
    assert sum(s is not None for s in scored) >= 1
    with pytest.raises(ValueError):
        PCAReconstructionDetector(chunk_size=10, stride=11)
    with pytest.raises(ValueError):
        PCAReconstructionDetector(reference_errors="nope")  # type: ignore[arg-type]


def test_multi_row_update_reports_any_drift() -> None:
    det = PCAReconstructionDetector(chunk_size=50, random_state=0).fit(sample(300, 0))
    assert det.update(sample(100, 1, shift=OFF_SUBSPACE)).drift


def test_missing_values_constant_columns_and_column_order() -> None:
    ref = sample(200, 0)
    ref["const"] = 7.0
    ref.iloc[::7, 2] = np.nan
    det = PCAReconstructionDetector(random_state=0).fit(ref)
    new = sample(60, 1)
    new["const"] = 7.0
    new.iloc[::5, 4] = np.nan
    scores = det.score(new[list(reversed(new.columns))])  # different column order
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError):
        det.score(new.drop(columns=["f0"]))
    with pytest.raises(RuntimeError):
        PCAReconstructionDetector().score(new)


def shift_stream(n_before: int = 300, n_after: int = 300) -> InstanceStream:
    """Covariate shift: features move off the subspace; best algorithm changes."""
    x = pd.concat(
        [sample(n_before, 5), sample(n_after, 6, shift=OFF_SUBSPACE)],
        ignore_index=True,
    )
    perf = np.where(
        (np.arange(len(x)) < n_before)[:, None], [[5, 50, 20]], [[50, 5, 20]]
    ).astype(float)
    idx = pd.Index([f"i{k}" for k in range(len(x))])
    x.index = idx
    return InstanceStream(
        x,
        pd.DataFrame(perf, index=idx, columns=["a", "b", "c"]),
        cutoff=1000.0,
        drift_points=(n_before,),
    )


def test_integration_with_online_pairwise_selector() -> None:
    stream = shift_stream()
    kw = dict(reference_period=80, random_state=0)
    detector = PCAReconstructionDetector(
        chunk_size=25, reference_errors="cross_validated", random_state=0
    )
    adaptive = run_prequential(OnlinePairwiseSelector(detector=detector, **kw), stream)
    static = run_prequential(OnlinePairwiseSelector(**kw), stream)

    summary = adaptive.summary()
    assert summary["n_missed"] == 0 and summary["mean_delay"] <= 2 * 25
    assert len(adaptive.retrain_steps) >= 1 and adaptive.drift_steps[0] >= 0
    # after the shift the adapted selector must do better than the frozen one
    after = slice(400, None)  # stream starts cold, drift at position 300
    a, s = (
        adaptive.steps["user_cost"].iloc[after],
        static.steps["user_cost"].iloc[after],
    )
    assert a.mean() < s.mean()
