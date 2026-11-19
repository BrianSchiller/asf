from asf.online.retraining import PeriodicRetrain


def test_periodic_policy() -> None:
    p = PeriodicRetrain(every=3)
    assert [p.should_retrain(s) for s in range(1, 7)] == [False, False, True] * 2
