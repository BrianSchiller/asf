from asf.online.drift import DriftSignal
from asf.online.retraining import DriftTriggeredRetrain, PeriodicRetrain


def test_periodic() -> None:
    p = PeriodicRetrain(every=3)
    assert [p.should_retrain(s) for s in range(1, 7)] == [False, False, True] * 2


def test_drift_triggered_needs_signal() -> None:
    p = DriftTriggeredRetrain()
    assert not p.should_retrain(5)
    assert not p.should_retrain(5, DriftSignal(drift=False))
    assert p.should_retrain(5, DriftSignal(drift=True))
