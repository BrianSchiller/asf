import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from asf.online.streams.stream import InstanceStream

from .conftest import make_stream

from typing import Any

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.evaluation import run_prequential
from asf.online.selectors import StaticOnlineSelector
from asf.selectors import PairwiseClassifier


class _Scripted(AbstractOnlineSelector):
    """Always runs ``a`` first and ``b`` as exploration; records feedback."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.seen: list[Observation] = []

    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kw: Any
    ) -> None:
        pass

    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        return [("a", 0.0), ("b", 0.0)]

    def _observe(self, observation: Observation) -> None:
        self.seen.append(observation)
        if self.step_ == 4:
            with self._timed() as t:
                pass
            self._record_retrain(t.seconds)


def test_cost_accounting_and_feedback(stream: InstanceStream) -> None:
    sel = _Scripted(window_size=50)
    res = run_prequential(sel, stream, n_warmstart=40)
    s = res.steps
    assert len(s) == len(stream) - 40
    assert (s["n_runs"] == 2).all()
    # primary is 'a' (5 or 50); exploration is the runtime of 'b' (50 or 5)
    assert (s["compute_cost"] == 55.0).all()  # 5 + 50, whichever is primary
    assert s["user_cost"].isin([5.0, 50.0]).all()
    assert (s["vbs_cost"] == 5.0).all()
    assert res.retrain_steps == [4] and bool(s["retrained"].iloc[4])
    assert len(sel.seen) == len(s)


def test_best_of_scheduled_mode(stream: InstanceStream) -> None:
    res = run_prequential(
        _Scripted(window_size=50),
        stream,
        n_warmstart=40,
        result_mode="best_of_scheduled",
    )
    assert (res.steps["user_cost"] == 5.0).all()  # best of a and b is always 5
    assert (res.steps["compute_cost"] == 55.0).all()


def test_cold_start_and_infinite_cutoff() -> None:
    """n_warmstart=0 must work, and cutoff=inf (non-runtime objectives) too."""
    stream = make_stream()
    stream.cutoff = float("inf")
    res = run_prequential(_Scripted(window_size=50), stream, n_warmstart=0)
    assert len(res.steps) == len(stream)
    assert res.summary()["mean_user_cost"] > 0


def test_censoring_and_budget(stream: InstanceStream) -> None:
    class _Short(_Scripted):
        def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
            return [("a", 10.0)]  # budget below the 50s of the slow case

    sel = _Short(window_size=50)
    res = run_prequential(sel, stream, n_warmstart=40, par_factor=10)
    cens = [o for o in sel.seen if "a" in o.censored]
    assert cens, "slow 'a' runs must be censored at their budget"
    assert all(o.runtimes["a"] == 10.0 for o in cens)
    # censored primary run is charged PAR-10 of the global cutoff
    assert res.steps["user_cost"].max() == 1000.0


def test_static_wrapper_degrades_after_drift(
    stream: InstanceStream,
) -> None:
    sel = StaticOnlineSelector(
        PairwiseClassifier(model_class=RandomForestClassifier, random_state=0)
    )
    res = run_prequential(sel, stream, n_warmstart=40)
    s = res.steps
    pre, post = s.iloc[: 80 - 40], s.iloc[80 - 40 :]
    # good before the drift, bad after it: that is what online methods must fix
    assert pre["user_cost"].mean() < post["user_cost"].mean()
    assert res.summary()["n_retrains"] == 0


def test_propose_requires_fit_and_single_row(stream: InstanceStream) -> None:
    sel = _Scripted()
    with pytest.raises(RuntimeError):
        sel.propose(stream.features.iloc[:1])
    sel.fit(stream.features.iloc[:20], stream.performance.iloc[:20])
    with pytest.raises(ValueError):
        sel.propose(stream.features.iloc[:2])


def test_batch_predict_still_works(stream: InstanceStream) -> None:
    sel = _Scripted()
    sel.fit(stream.features.iloc[:20], stream.performance.iloc[:20])
    out = sel.predict(stream.features.iloc[20:23])
    assert list(out) == ["i20", "i21", "i22"]
    assert all(len(v) == 1 for v in out.values())


def test_maximize_quality_uses_higher_scores_for_sbs_vbs_and_schedules() -> None:
    features = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    performance = pd.DataFrame({"a": [5.0, 3.0, 8.0], "b": [1.0, 9.0, 4.0]})
    stream = InstanceStream(features, performance, cutoff=float("inf"))

    primary = run_prequential(_Scripted(maximize=True), stream, n_warmstart=1)
    assert primary.maximize
    assert primary.steps["user_cost"].tolist() == [3.0, 8.0]
    assert primary.steps["vbs_cost"].tolist() == [9.0, 8.0]
    assert primary.steps["sbs_cost"].tolist() == [3.0, 8.0]
    assert primary.summary()["closed_gap"] == 0.0
    assert primary.steps["compute_cost"].isna().all()

    parallel = run_prequential(
        _Scripted(maximize=True), stream, n_warmstart=1, result_mode="best_of_scheduled"
    )
    assert parallel.steps["user_cost"].tolist() == [9.0, 8.0]
    assert parallel.summary()["closed_gap"] == 1.0

    cold = run_prequential(_Scripted(maximize=True), stream)
    assert cold.steps["sbs_cost"].tolist() == [5.0, 3.0, 8.0]


def test_maximize_quality_rejects_runtime_constraints() -> None:
    features = pd.DataFrame({"f": [0.0, 1.0]})
    performance = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="cutoff=inf"):
        run_prequential(
            _Scripted(maximize=True), InstanceStream(features, performance, cutoff=10)
        )

    class _Budgeted(_Scripted):
        def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
            return [("a", 5.0)]

    with pytest.raises(ValueError, match="unlimited run budgets"):
        run_prequential(
            _Budgeted(maximize=True),
            InstanceStream(features, performance, cutoff=float("inf")),
        )
