"""
Core contract for online algorithm selection.

An online selector is driven by an evaluation loop (or a real deployment) in two
steps per incoming instance:

1. ``propose(features)`` -> which algorithm(s) to run, as an ASlib-style schedule.
2. ``observe(observation)`` -> the runtimes of the algorithm(s) that were
   actually executed. The selector may update itself (retrain, update bandit
   statistics, ...) in response.

Only the algorithms that were run are reported back, so the performance
information a selector accumulates is *partially observed*.

Inside ``propose`` the base class runs::

    record = _select(features)                  # subclass: PredictionRecord
    extras = gate.extras(GateContext(record))    # extra-run gate
    last_prediction_ = record with extras        # logged by the evaluator

and inside ``observe``::

    buffer.append(observation); _observe(observation)   # subclass: learn
    signal = detector.update(...)                        # feature or scalar detector
    if signal.drift: on_drift(signal)                    # drift responses
    if retrain_policy.should_retrain(...): retrain()     # schedule-based retraining
    re-reference the detector if the model was rebuilt
"""

from __future__ import annotations

import functools
import inspect
import time
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from asf.online.buffer import ObservationBuffer
from asf.online.drift.abstract_detector import (
    AbstractDriftDetector,
    AbstractScalarDriftDetector,
    DriftSignal,
)
from asf.online.drift.statistics import AbstractDriftStatistic
from asf.online.gate import AbstractExtraRunGate, AlwaysProbe, GateContext
from asf.online.prediction import PredictionRecord
from asf.online.retraining.policy import AbstractRetrainPolicy
from asf.online.retraining.response import AbstractDriftResponse
from asf.selectors.abstract_selector import AbstractSelector


@dataclass(frozen=True)
class Observation:
    """
    Feedback for one instance after the proposed algorithms were executed.

    Attributes
    ----------
    instance_id : Any
        Identifier of the instance (index label of its feature row).
    features : pd.DataFrame
        The single-row feature frame the selector was asked about.
    runtimes : Mapping[str, float]
        Observed runtime per *executed* algorithm. For censored runs this is the
        budget the run was stopped at.
    censored : frozenset[str]
        Names of executed algorithms whose run was stopped before solving.
    cutoff : float
        The global cutoff of the scenario (used e.g. for PAR scoring).
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

    Attributes
    ----------
    step : int
        0-based index of the instance in the stream.
    prediction : PredictionRecord or None
        The record of ``propose`` (``None`` if ``observe`` was called without).
    instance_score : float or None
        Detector score passed to the gate, if one was computed.
    signal : DriftSignal or None
        Detector output for this instance (``None`` = detector not run).
    drift_statistic : float or None
        Value fed to a scalar detector, if any.
    responses : list[str]
        ``NAME`` of every drift response executed for this instance.
    detector_seconds : float
        Wall-clock time spent in detector calls (scoring, updating, fitting,
        re-referencing) during ``propose`` and ``observe`` of this instance.
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

    It extends :class:`~asf.selectors.abstract_selector.AbstractSelector`, so a
    warm start works through the usual ``fit(features, performance)`` and
    the selector can be used with existing ASF tooling. Subclasses implement:

    * ``_fit_initial``: build the initial model from the warm-start data.
    * ``_select``: return a :class:`~asf.online.prediction.PredictionRecord`
      for ONE instance (chosen algorithm, budget, predictions, nominees for
      extra runs). Older subclasses may implement ``_propose`` instead and
      return a schedule; its first entry becomes the choice and the rest the
      nominees.
    * ``_observe``: learn from feedback (update statistics, ...).

    and, optionally, the capability hooks used by drift responses:
    ``_retrain``, ``_reset_model``, ``_discount_history``,
    ``_boost_exploration``.

    The base class owns the components that are shared by all selectors:

    * the **extra-run gate** (``gate``) deciding which nominees actually run,
    * the **drift detector** (``detector``), feature-based or scalar-stream
      (the latter with a ``drift_statistic``),
    * the **drift responses** (``drift_response``) run by :meth:`on_drift`,
    * the schedule-based **retrain policy** (``retrain_policy``).

    Bookkeeping that the evaluation loop reads:

    * ``last_prediction_`` / ``last_trace_``: what happened on the last
      instance.
    * ``retrain_steps`` / ``retrain_seconds``: when and how long we retrained.
    * ``drift_steps``: steps at which the detector signalled drift.
    * ``init_params_``: the constructor arguments (captured automatically).

    Notes
    -----
    Schedule convention: the first tuple of the schedule is the *primary*
    choice. Further tuples are *extra* runs for information gathering; they
    get the primary run's budget. A budget of ``None`` or ``<= 0`` means "use
    the scenario cutoff".

    Parameters
    ----------
    window_size : int or None, default=200
        Size of the sliding window held in the observation buffer (``None`` =
        keep the full history).
    gate : AbstractExtraRunGate or None, default=None
        Extra-run gate; ``None`` uses :meth:`_default_gate`, which runs all of
        the selector's nominees (so every selector keeps its built-in
        behaviour).
    detector : AbstractDriftDetector, AbstractScalarDriftDetector or None, default=None
        Drift detector. Feature-based detectors are fitted on the buffer once
        ``detector_reference_size`` instances have been seen (or right after
        a warm start); scalar detectors need ``drift_statistic``.
    drift_statistic : AbstractDriftStatistic or None, default=None
        Turns observation and prediction into the scalar a scalar detector
        consumes. Required for, and only allowed with, scalar detectors.
    drift_response : AbstractDriftResponse, sequence of them, or None, default=None
        Responses executed, in order, when the detector signals drift. ``None``
        means no response (the signal is only recorded), unless a subclass
        sets a default.
    retrain_policy : AbstractRetrainPolicy or None, default=None
        Checked after every instance (independently of drift responses);
        retrains when it fires. Requires a selector implementing ``_retrain``.
    detector_reference_size : int or None, default=None
        Number of instances (warm start included) after which a feature
        detector is fitted on the buffer. ``None``: fit right after the warm
        start; then a cold start with a feature detector is an error.
    **kwargs : Any
        Forwarded to :class:`~asf.selectors.abstract_selector.AbstractSelector`.
    """

    #: How the evaluator scores a schedule by default: ``"primary"`` (the user
    #: gets the first algorithm's result) or ``"best_of_scheduled"`` (the user
    #: gets the best result among all scheduled algorithms, as with parallel runs).
    RESULT_MODE: str = "primary"

    #: Whether the selector can start without warm-start data.
    SUPPORTS_COLD_START: bool = True

    #: Subclasses that decide themselves when the detector is fitted (by
    #: calling :meth:`fit_detector`) set this to ``True``.
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
        drift_statistic: AbstractDriftStatistic | None = None,
        drift_response: DriftResponses = None,
        retrain_policy: AbstractRetrainPolicy | None = None,
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
        self.retrain_policy = retrain_policy
        self.detector_reference_size = (
            None if detector_reference_size is None else int(detector_reference_size)
        )

        self.buffer: ObservationBuffer | None = None
        self.step_: int = 0
        self.n_seen_: int = 0
        self.retrain_steps: list[int] = []
        self.retrain_seconds: float = 0.0
        self.drift_steps: list[int] = []
        self.detector_ready_: bool = False
        self.last_prediction_: PredictionRecord | None = None
        self.last_trace_: StepTrace | None = None
        self.last_signal_: DriftSignal | None = None
        self.last_drift_step_: int | None = None
        self._trace: StepTrace | None = None
        self._n_warmstart = 0
        self._retrained_this_step = False
        self._rereference_pending = False

    # ------------------------------------------------------------------ fit
    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """Seed the buffer with the warm-start data, then fit the initial model."""
        self._check_components(n_warmstart=len(features))
        self.step_ = 0
        self.n_seen_ = len(features)
        self._n_warmstart = len(features)
        self.retrain_steps = []
        self.retrain_seconds = 0.0
        self.drift_steps = []
        self.detector_ready_ = False
        self.last_prediction_ = None
        self.last_trace_ = None
        self.last_signal_ = None
        self.last_drift_step_ = None
        self._trace = None
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
        if self.retrain_policy is not None and not self.supports("_retrain"):
            raise ValueError(
                f"{type(self).__name__} does not implement _retrain, which a "
                "retrain_policy needs."
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
        """
        Propose a schedule for a single incoming instance.

        Parameters
        ----------
        features : pd.DataFrame
            A frame with exactly one row (the instance).

        Returns
        -------
        list[tuple[str, float]]
            ``(algorithm, budget)`` tuples; the first is the primary choice,
            the others are the extra runs chosen by the gate.
        """
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
            candidates=tuple(record.candidates),
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
        """
        Feed back the outcome of the executed schedule.

        Parameters
        ----------
        observation : Observation
            Runtimes (and censoring info) of the algorithms that were run.
        """
        self._check_fitted()
        assert self.buffer is not None
        obs = observation
        if self.hierarchical_generator is not None:
            obs = Observation(
                instance_id=obs.instance_id,
                features=self._augment(obs.features),
                runtimes=obs.runtimes,
                censored=obs.censored,
                cutoff=obs.cutoff,
            )
        trace = self._trace if self._trace is not None else StepTrace(self.step_)
        self._trace = trace
        self._retrained_this_step = False
        self._rereference_pending = False

        self.buffer.append(obs)
        self.n_seen_ += 1
        self._observe(obs)  # step_ is the 0-based index of this instance

        signal: DriftSignal | None = None
        if self._adaptation_active():
            signal = self._detect(obs, trace)
            if signal is not None:
                self.last_signal_ = signal
                if signal.drift:
                    self._record_drift()
                    self.last_drift_step_ = self.step_
                    self.on_drift(signal)
            policy = self.retrain_policy
            # Number of instances seen in the stream including the current one.
            if policy is not None and policy.should_retrain(self.step_ + 1, signal):
                if not self._retrained_this_step:
                    self.retrain()
                policy.on_retrained(self.step_ + 1)
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
        """
        React to a drift signal by running the configured drift responses.

        Called by :meth:`observe` when the detector signals drift; may be
        overridden for selector-specific reactions.
        """
        for response in self.drift_responses_:
            response.respond(self, signal)
            if self._trace is not None:
                self._trace.responses.append(response.NAME)

    def retrain(self) -> None:
        """Retrain now (timed and recorded); the detector is re-referenced."""
        with self._timed() as timer:
            self._retrain()
        self._record_retrain(timer.seconds)
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
        if isinstance(self.detector, AbstractScalarDriftDetector):
            with self._timed() as timer:
                self.detector.reset()
        else:
            if len(self.buffer) == 0:
                return
            X, P = self.buffer.features(), self.buffer.performance()
            with self._timed() as timer:
                self.detector.fit(X, P)
        self._add_detector_seconds(timer.seconds)
        self.detector_ready_ = True

    def rereference_detector(self) -> None:
        """Re-reference a fitted detector after the model was rebuilt."""
        if self.detector is None or self.buffer is None or not self.detector_ready_:
            return
        if isinstance(self.detector, AbstractScalarDriftDetector):
            with self._timed() as timer:
                self.detector.reset()
        else:
            X, P = self.buffer.features(), self.buffer.performance()
            with self._timed() as timer:
                self.detector.reset(X, P)
        self._add_detector_seconds(timer.seconds)
        self._rereference_pending = False

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

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        """
        Decide for one instance (single-row frame).

        The default wraps the older ``_propose`` hook: the first schedule
        entry is the choice, the remaining entries are the nominees.
        """
        schedule = self._propose(features)
        if not schedule:
            raise RuntimeError("_propose() must return a non-empty schedule.")
        self._check_known([str(a) for a, _ in schedule])
        algo, budget = schedule[0]
        return PredictionRecord(
            chosen=str(algo),
            budget=float(budget or 0.0),
            candidates=tuple(str(a) for a, _ in schedule[1:]),
        )

    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        """Older hook: return the schedule for one instance. Prefer ``_select``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _select (or _propose)."
        )

    @abstractmethod
    def _observe(self, observation: Observation) -> None:
        """Update the selector after feedback. The buffer is already updated."""

    def _default_gate(self) -> AbstractExtraRunGate:
        """Gate used when none is given: run all of the selector's nominees."""
        return AlwaysProbe()

    def _adaptation_active(self) -> bool:
        """Whether detection, drift responses and the retrain policy run for
        the current instance (e.g. ``False`` during a reference period)."""
        return True

    def _retrain(self) -> None:
        """Capability hook: rebuild the model from the buffer."""
        raise NotImplementedError

    def _reset_model(self) -> None:
        """Capability hook: forget everything learnt so far."""
        raise NotImplementedError

    def _discount_history(self, factor: float) -> None:
        """Capability hook: multiply the weight of all past observations by
        ``factor``."""
        raise NotImplementedError

    def _boost_exploration(self, factor: float, steps: int | None) -> None:
        """Capability hook: explore more for ``steps`` instances."""
        raise NotImplementedError

    # ---------------------------------------------------------- helpers
    def _record_retrain(self, seconds: float) -> None:
        """Log a retraining at the current (0-based) step that took ``seconds``."""
        self.retrain_steps.append(self.step_)
        self.retrain_seconds += float(seconds)

    def _record_drift(self) -> None:
        """Log that a drift was signalled at the current step."""
        self.drift_steps.append(self.step_)

    def _timed(self) -> "_Timer":
        """Context manager measuring wall-clock seconds, e.g. for retrain timing."""
        return _Timer()

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
        with self._timed() as timer:
            if isinstance(self.detector, AbstractScalarDriftDetector):
                assert self.drift_statistic is not None
                sample = self.drift_statistic(obs, trace.prediction)
                if sample is None:
                    signal = None
                else:
                    trace.drift_statistic = float(sample.value)
                    signal = self.detector.update(sample.value, sample.censored)
            else:
                signal = self.detector.update(obs.features, self._performance_row(obs))
        self._add_detector_seconds(timer.seconds)
        return signal

    def _instance_score(self, features: pd.DataFrame) -> float | None:
        if not isinstance(self.detector, AbstractDriftDetector) or not (
            self.detector_ready_
        ):
            return None
        with self._timed() as timer:
            score = float(self.detector.score(features)[0])
        self._add_detector_seconds(timer.seconds)
        return score

    def _add_detector_seconds(self, seconds: float) -> None:
        if self._trace is not None:
            self._trace.detector_seconds += float(seconds)

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


class _Timer:
    """Tiny wall-clock timer used as a context manager."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
