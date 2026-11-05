import math

from asf.online.evaluation.metrics import (
    closed_gap,
    detection_metrics,
    penalised_runtime,
)


def test_penalised_runtime() -> None:
    assert penalised_runtime(5, 100) == 5
    assert penalised_runtime(101, 100) == 1000


def test_closed_gap() -> None:
    assert closed_gap(10, 20, 10) == 1.0
    assert closed_gap(20, 20, 10) == 0.0
    assert math.isnan(closed_gap(1, 5, 5))


def test_detection_metrics() -> None:
    m = detection_metrics([50, 200], [55, 300], tolerance=20)
    assert m["mean_delay"] == 5 and m["n_missed"] == 1 and m["n_false_alarms"] == 1
