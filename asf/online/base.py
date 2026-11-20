"""
Core contract for online algorithm selection.

Per instance, ``propose(features)`` returns an ASlib-style schedule and
``observe(observation)`` reports the runtimes of the algorithms that ran
(only those, so the selector's data is partially observed).

Inside ``propose`` the base class runs::

    record = _select(features)                  # subclass: PredictionRecord
    extras = gate.extras(GateContext(record))    # extra-run gate
    last_prediction_ = record with extras        # logged by the evaluator

and inside ``observe``::

    buffer.append(observation); _observe(observation)   # subclass: learn
    gate.observe(last_prediction_, observation)          # gate sees the outcome
    signal = detector.update(...)                        # feature or scalar detector
    if signal.drift: on_drift(signal)                    # drift responses
    every retrain_every instances: retrain()             # schedule-based retraining
    re-reference the detector if the model was rebuilt
"""

from __future__ import annotations

import functools
import inspect
import time
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from asf.online.buffer import ObservationBuffer
from asf.online.drift.abstract_detector import (
    AbstractDriftDetector,
    AbstractScalarDriftDetector,
    DriftSignal,
)
from asf.online.drift.statistics import DriftStatistic
from asf.online.gate import AbstractExtraRunGate, AllNomineesGate, GateContext
from asf.online.prediction import PredictionRecord
from asf.online.retraining.response import AbstractDriftResponse
from asf.selectors.abstract_selector import AbstractSelector


@dataclass(frozen=True)
class Observation:
    """
    Feedback for one instance: the runtime of every *executed* algorithm
    (for ``censored`` runs the budget they were stopped at) and the
    scenario ``cutoff``.
    """

    instance_id: Any
    features: pd.DataFrame
    runtimes: Mapping[str, float]
    censored: frozenset[str] = field(default_factory=frozenset)
    cutoff: float = float("inf")


@dataclass
class StepTrace:
    """
    Bookkeeping of one instance, read by the evaluator after ``observe``.

    ``signal`` is ``None`` if the detector did not run, ``drift_statistic``
    is the value fed to a scalar detector, ``responses`` the ``NAME`` of each
    drift response executed and ``detector_seconds`` the time spent in
    detector calls during ``propose`` and ``observe``.
    """

    step: int
    prediction: PredictionRecord | None = None
    instance_score: float | None = None
    signal: DriftSignal | None = None
    drift_statistic: float | None = None
    responses: list[str] = field(default_factory=list)
    detector_seconds: float = 0.0


DriftResponses = AbstractDriftResponse | Sequence[AbstractDriftResponse] | None


class AbstractOnlineSelector(AbstractSelector):
    """
    Base class for selectors that keep learning from the instance stream.

    A warm start goes through the usual ``fit(features, performance)``.
    Subclasses implement ``_fit_initial`` (warm-start model), ``_select`` (a
    :class:`~asf.online.prediction.PredictionRecord` for one instance) and
    ``_observe`` (learn from feedback), plus optionally the hooks drift
    responses need: ``_retrain``, ``_reset_model``, ``_discount_history``,
    ``_boost_exploration``.

    The schedule's first entry is the primary run, the others are extra runs
    with the same budget (``None`` or ``<= 0`` = the scenario cutoff).
    The evaluator reads ``last_trace_``, ``retrain_steps``,
    ``retrain_seconds`` and ``drift_steps``; ``init_params_`` holds the
    constructor arguments (captured automatically, for provenance).

    Parameters
    ----------
    window_size : int or None, default=200
        Instances kept in the observation buffer (``None`` = all).
    gate : AbstractExtraRunGate or None, default=None
        Decides which nominees run; ``None`` = :meth:`_default_gate`.
    detector : AbstractDriftDetector, AbstractScalarDriftDetector or None
        Feature detectors are fitted on the buffer after the warm start or
        after ``detector_reference_size`` instances; scalar detectors need a
        ``drift_statistic``.
    drift_statistic : DriftStatistic or None, default=None
        ``(observation, prediction) -> (value, censored) | None`` for a scalar
        detector (required with one, forbidden otherwise).
    drift_response : AbstractDriftResponse, sequence of them, or None
        Run in order when the detector signals drift (``None``: only record).
    retrain_every : int or None, default=None
        Retrain after every ``retrain_every`` stream instances (warm start
        not counted), independently of drift. Needs ``_retrain``.
    detector_reference_size : int or None, default=None
        Instances (warm start included) a feature detector is fitted on;
        ``None`` = the warm start (a cold start then raises).
    **kwargs : Any
        Forwarded to :class:`~asf.selectors.abstract_selector.AbstractSelector`.
    """

    #: Default scoring: the first algorithm's result (``"primary"``) or the
    #: best of all scheduled ones (``"best_of_scheduled"``, parallel runs).
    RESULT_MODE: str = "primary"

    #: Whether the selector can start without warm-start data.
    SUPPORTS_COLD_START: bool = True

    #: ``True`` if the subclass calls :meth:`fit_detector` itself.
    MANAGES_DETECTOR_FIT: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            cls.__init__ = _capture_init_params(cls.__dict__["__init__"])  # type: ignore[method-assign]

    def __init__(
        self,
        window_size: int | None = 200,
        gate: AbstractExtraRunGate | None = None,
        detector: AbstractDriftDetector | AbstractScalarDriftDetector | None = None,
        drift_statistic: DriftStatistic | None = None,
        drift_response: DriftResponses = None,
        retrain_every: int | None = None,
        detector_reference_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.window_size = None if window_size is None else int(window_size)
        if detector is not None and not isinstance(
            detector, (AbstractDriftDetector, AbstractScalarDriftDetector)
        ):
            raise TypeError(
                "detector must be an AbstractDriftDetector or an "
                f"AbstractScalarDriftDetector, got {type(detector).__name__}."
            )
        if isinstance(detector, AbstractScalarDriftDetector):
            if drift_statistic is None:
                raise ValueError("A scalar detector needs a drift_statistic.")
        elif drift_statistic is not None:
            raise ValueError("drift_statistic is only used with a scalar detector.")
        if retrain_every is not None and retrain_every < 1:
            raise ValueError("retrain_every must be >= 1.")
        if detector_reference_size is not None and detector_reference_size < 1:
            raise ValueError("detector_reference_size must be >= 1.")
        self.gate = gate
        self.gate_: AbstractExtraRunGate = (
            gate if gate is not None else self._default_gate()
        )
        self.detector = detector
        self.drift_statistic = drift_statistic
        self.drift_response = drift_response
        self.drift_responses_: tuple[AbstractDriftResponse, ...] = _as_responses(
            drift_response
        )
        self.retrain_every = retrain_every
        self.detector_reference_size = (
            None if detector_reference_size is None else int(detector_reference_size)
        )

        self.buffer: ObservationBuffer | None = None
        self._reset_state(n_warmstart=0)
        self._retrained_this_step = False
        self._rereference_pending = False

    def _reset_state(self, n_warmstart: int) -> None:
        """Clear the stream bookkeeping (in ``__init__`` and at every ``fit``)."""
        self.step_ = 0
        self.n_seen_ = n_warmstart
        self._n_warmstart = n_warmstart
        self.retrain_steps: list[int] = []
        self.retrain_seconds = 0.0
        self.drift_steps: list[int] = []
        self.detector_ready_ = False
        self.last_prediction_: PredictionRecord | None = None
        self.last_trace_: StepTrace | None = None
        self.last_signal_: DriftSignal | None = None
        self.last_drift_step_: int | None = None
        self._trace: StepTrace | None = None

    # ------------------------------------------------------------------ fit
    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """Seed the buffer with the warm-start data, then fit the initial model."""
        self._check_components(n_warmstart=len(features))
        self._reset_state(n_warmstart=len(features))
        self.gate_.reset()
        self.buffer = ObservationBuffer(
            algorithms=self.algorithms,
            feature_names=list(features.columns),
            window_size=self.window_size,
        )
        self.buffer.extend_full(features, performance)
        self._fit_initial(features, performance, **kwargs)
        if not self.MANAGES_DETECTOR_FIT:
            self._maybe_fit_detector()

    def _check_components(self, n_warmstart: int) -> None:
        for response in self.drift_responses_:
            if not self.supports(response.REQUIRES):
                raise ValueError(
                    f"{type(self).__name__} does not implement {response.REQUIRES}, "
                    f"which {type(response).__name__} needs."
                )
        if self.retrain_every is not None and not self.supports("_retrain"):
            raise ValueError(
                f"{type(self).__name__} does not implement _retrain, which "
                "retrain_every needs."
            )
        if (
            isinstance(self.detector, AbstractDriftDetector)
            and not self.MANAGES_DETECTOR_FIT
            and self.detector_reference_size is None
            and n_warmstart == 0
        ):
            raise ValueError(
                "A feature-based detector on a cold start needs "
                "detector_reference_size (instances to fit it on)."
            )

    # -------------------------------------------------------------- online
    def propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        """``(algorithm, budget)`` schedule for one instance (one-row frame):
        the primary choice, then the extras chosen by the gate."""
        self._check_fitted()
        if len(features) != 1:
            raise ValueError("propose() expects exactly one instance (one row).")
        features = self._augment(features)
        trace = StepTrace(step=self.step_)
        self._trace = trace
        record = self._select(features)
        if not isinstance(record, PredictionRecord):
            raise TypeError("_select() must return a PredictionRecord.")
        self._check_known([record.chosen, *record.candidates])

        if self.gate_.needs_instance_score:
            trace.instance_score = self._instance_score(features)
        context = GateContext(
            prediction=record,
            algorithms=tuple(self.algorithms),
            features=features,
            step=self.step_,
            instance_score=trace.instance_score,
            last_signal=self.last_signal_,
            steps_since_drift=(
                None
                if self.last_drift_step_ is None
                else self.step_ - self.last_drift_step_
            ),
        )
        extras = self._clean_extras(self.gate_.extras(context), record.chosen)
        record = replace(record, extras=extras, instance_id=features.index[0])
        self.last_prediction_ = trace.prediction = record
        return record.schedule

    def observe(self, observation: Observation) -> None:
        """Learn from the executed schedule, then detect drift and react."""
        self._check_fitted()
        assert self.buffer is not None
        obs = replace(observation, features=self._augment(observation.features))
        trace = self._trace if self._trace is not None else StepTrace(self.step_)
        self._trace = trace
        self._retrained_this_step = False
        self._rereference_pending = False

        self.buffer.append(obs)
        self.n_seen_ += 1
        self._observe(obs)  # step_ is the 0-based index of this instance
        if trace.prediction is not None:
            self.gate_.observe(trace.prediction, obs)

        signal: DriftSignal | None = None
        if self._adaptation_active():
            signal = self._detect(obs, trace)
            if signal is not None:
                self.last_signal_ = signal
                if signal.drift:
                    self.drift_steps.append(self.step_)
                    self.last_drift_step_ = self.step_
                    self.on_drift(signal)
            # step_ + 1 = stream instances seen, the current one included
            every = self.retrain_every
            if (
                every
                and (self.step_ + 1) % every == 0
                and not self._retrained_this_step
            ):
                self.retrain()
            if self._rereference_pending:
                self.rereference_detector()
        if not self.MANAGES_DETECTOR_FIT:
            self._maybe_fit_detector()

        trace.signal = signal
        self.last_trace_ = trace
        self._trace = None
        self.step_ += 1

    # ------------------------------------------------------------ drift
    def on_drift(self, signal: DriftSignal) -> None:
        """Run the configured drift responses (override for custom reactions)."""
        for response in self.drift_responses_:
            response.respond(self, signal)
            if self._trace is not None:
                self._trace.responses.append(response.NAME)

    def retrain(self) -> None:
        """Retrain now (timed and recorded); the detector is re-referenced."""
        start = time.perf_counter()
        self._retrain()
        self.retrain_steps.append(self.step_)
        self.retrain_seconds += time.perf_counter() - start
        self._retrained_this_step = True
        self._rereference_pending = True

    def reset_model(self) -> None:
        """Forget what was learnt; the detector is re-referenced."""
        self._reset_model()
        self._rereference_pending = True

    def fit_detector(self) -> None:
        """Fit a feature detector on the buffer (reset a scalar one)."""
        if self.detector is None or self.buffer is None:
            return
        if isinstance(self.detector, AbstractDriftDetector) and len(self.buffer) == 0:
            return
        self._reference_detector("fit")
        self.detector_ready_ = True

    def rereference_detector(self) -> None:
        """Re-reference a fitted detector after the model was rebuilt."""
        if self.detector_ready_:  # implies a detector and a buffer
            self._reference_detector("reset")
        self._rereference_pending = False

    def _reference_detector(self, method: str) -> None:
        """Call a feature detector's ``fit``/``reset`` on the buffer; a scalar
        detector is just ``reset()``."""
        assert self.detector is not None and self.buffer is not None
        if isinstance(self.detector, AbstractScalarDriftDetector):
            self._time_detector(self.detector.reset)
        else:
            X, P = self.buffer.features(), self.buffer.performance()
            self._time_detector(getattr(self.detector, method), X, P)

    def supports(self, hook: str) -> bool:
        """Whether this selector overrides the capability hook ``hook``."""
        base = getattr(AbstractOnlineSelector, hook, None)
        return base is not None and getattr(type(self), hook, None) is not base

    def describe_params(self) -> dict[str, Any]:
        """Constructor arguments plus the base settings, for provenance."""
        params: dict[str, Any] = {
            "window_size": self.window_size,
            "budget": self.budget,
            "maximize": self.maximize,
            "random_state": self.random_state,
        }
        params.update(getattr(self, "init_params_", {}))
        return params

    # ------------------------------------------------------ AbstractSelector
    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Batch prediction without learning: the primary choice per row."""
        if features is None:
            raise ValueError("Online selectors require features for prediction.")
        # features were already augmented by AbstractSelector.predict
        out = {}
        for idx in features.index:
            record = self._select(features.loc[[idx]])
            out[str(idx)] = [(record.chosen, record.budget)]
        return out

    # ---------------------------------------------------------- hooks
    @abstractmethod
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """Build the initial model from the (fully observed) warm-start data."""

    @abstractmethod
    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        """Decide for one instance (single-row frame)."""

    @abstractmethod
    def _observe(self, observation: Observation) -> None:
        """Update the selector after feedback. The buffer is already updated."""

    def _default_gate(self) -> AbstractExtraRunGate:
        """Gate used when none is given: run all of the selector's nominees."""
        return AllNomineesGate()

    def _adaptation_active(self) -> bool:
        """``False`` skips detection and retraining for this instance (e.g.
        during a reference period)."""
        return True

    def _retrain(self) -> None:
        """Capability hook: rebuild the model from the buffer."""
        raise NotImplementedError

    def _reset_model(self) -> None:
        """Capability hook: forget everything learnt so far."""
        raise NotImplementedError

    def _discount_history(self, factor: float) -> None:
        """Capability hook: multiply the weight of all past observations."""
        raise NotImplementedError

    def _boost_exploration(self, factor: float, steps: int | None) -> None:
        """Capability hook: explore more for ``steps`` instances."""
        raise NotImplementedError

    # ---------------------------------------------------------- helpers
    def _maybe_fit_detector(self) -> None:
        if not isinstance(self.detector, AbstractDriftDetector):
            if self.detector is not None and not self.detector_ready_:
                self.fit_detector()  # scalar detectors: reset once, then ready
            return
        if self.detector_ready_:
            return
        needed = self.detector_reference_size
        if needed is None:
            needed = self._n_warmstart  # right after the warm start
        if needed > 0 and self.n_seen_ >= needed:
            self.fit_detector()

    def _detect(self, obs: Observation, trace: StepTrace) -> DriftSignal | None:
        if self.detector is None or not self.detector_ready_:
            return None
        if isinstance(self.detector, AbstractScalarDriftDetector):
            assert self.drift_statistic is not None
            sample = self.drift_statistic(obs, trace.prediction)
            if sample is None:
                return None
            value, censored = sample
            trace.drift_statistic = float(value)
            return self._time_detector(self.detector.update, value, censored)
        return self._time_detector(
            self.detector.update, obs.features, self._performance_row(obs)
        )

    def _instance_score(self, features: pd.DataFrame) -> float | None:
        if not isinstance(self.detector, AbstractDriftDetector) or not (
            self.detector_ready_
        ):
            return None
        return float(self._time_detector(self.detector.score, features)[0])

    def _time_detector(self, call: Callable[..., Any], *args: Any) -> Any:
        """Run a detector call and add its wall-clock time to the step trace."""
        start = time.perf_counter()
        out = call(*args)
        if self._trace is not None:
            self._trace.detector_seconds += time.perf_counter() - start
        return out

    def _clean_extras(self, extras: Sequence[str], chosen: str) -> tuple[str, ...]:
        """Validate the gate's answer; drop duplicates and the primary choice."""
        names = [str(a) for a in extras]
        self._check_known(names)
        out: list[str] = []
        for name in names:
            if name != chosen and name not in out:
                out.append(name)
        return tuple(out)

    def _check_known(self, names: Sequence[str]) -> None:
        unknown = [a for a in names if a not in self.algorithms]
        if unknown:
            raise ValueError(f"Schedule contains unknown algorithms: {unknown}")

    def _augment(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.hierarchical_generator is None:
            return features
        return pd.concat(
            [features, self.hierarchical_generator.generate_features(features)],
            axis=1,
        )

    def _performance_row(self, observation: Observation) -> pd.DataFrame:
        """One-row performance frame of an observation (NaN = not executed)."""
        return pd.DataFrame(
            [dict(observation.runtimes)],
            index=pd.Index([observation.instance_id]),
            columns=pd.Index(self.algorithms),
        )

    def _check_fitted(self) -> None:
        if self.buffer is None:
            raise RuntimeError("Call fit() with warm-start data before streaming.")


def random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest value; ties are broken uniformly at random."""
    return int(rng.choice(np.flatnonzero(values == values.max())))


def _as_responses(value: DriftResponses) -> tuple[AbstractDriftResponse, ...]:
    if value is None:
        return ()
    items = (value,) if isinstance(value, AbstractDriftResponse) else tuple(value)
    for item in items:
        if not isinstance(item, AbstractDriftResponse):
            raise TypeError(
                f"drift_response entries must be AbstractDriftResponse, got "
                f"{type(item).__name__}."
            )
    return items


def _capture_init_params(init: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``__init__`` so the outermost call stores its bound arguments."""
    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        outer = not self.__dict__.get("_capturing_init_params", False)
        if not outer:
            init(self, *args, **kwargs)
            return
        self.__dict__["_capturing_init_params"] = True
        try:
            init(self, *args, **kwargs)
        finally:
            del self.__dict__["_capturing_init_params"]
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params: dict[str, Any] = {}
        for name, value in list(bound.arguments.items())[1:]:
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                params.update(value)
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                params[name] = list(value)
            else:
                params[name] = value
        self.init_params_ = params

    return wrapper
