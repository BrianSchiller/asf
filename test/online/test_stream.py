import pytest

from asf.online.streams.stream import InstanceStream


def test_iteration_yields_single_row_frames(stream: InstanceStream) -> None:
    items = list(stream)
    assert len(items) == len(stream)
    assert items[3].features.shape == (1, 2) and items[3].step == 3


def test_split_shifts_drift_points(stream: InstanceStream) -> None:
    warm, tail = stream.split(30)
    assert len(warm) == 30 and len(tail) == len(stream) - 30
    assert tail.drift_points == (stream.drift_points[0] - 30,)
    cold_warm, cold_tail = stream.split(0)  # cold start: nothing is warm
    assert len(cold_warm) == 0 and len(cold_tail) == len(stream)
    assert cold_tail.drift_points == stream.drift_points


def test_mismatched_index_rejected(stream: InstanceStream) -> None:
    with pytest.raises(ValueError):
        InstanceStream(
            stream.features, stream.performance.iloc[::-1].reset_index(), 1.0
        )
