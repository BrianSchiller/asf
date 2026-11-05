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
"""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from asf.online.buffer import ObservationBuffer
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


class AbstractOnlineSelector(AbstractSelector):
    """
    Base class for selectors that keep learning from the instance stream.

    It extends :class:`~asf.selectors.abstract_selector.AbstractSelector`, so a
    warm start works through the usual ``fit(features, performance)`` and
    the selector can be used with existing ASF tooling. Subclasses implement:

    * ``_fit_initial``: build the initial model from the warm-start data.
    * ``_propose``: return the schedule for ONE instance.
    * ``_observe``: react to feedback (retrain, update statistics, ...).

    Bookkeeping that the evaluation loop reads is maintained here:

    * ``retrain_steps`` / ``retrain_seconds``: when and how long we retrained.
      Subclasses call :meth:`_record_retrain` around a retraining.
    * ``drift_steps``: steps at which a drift detector signalled drift.
      Subclasses call :meth:`_record_drift`.

    Notes
    -----
    Schedule convention: the first tuple of the schedule is the *primary*
    choice, i.e. the algorithm whose result is what the user gets. Any further
    tuples are *additional* runs for information gathering (exploration cost).
    A budget of ``None`` or ``<= 0`` means "use the scenario cutoff".

    Parameters
    ----------
    window_size : int or None, default=200
        Size of the sliding window held in the observation buffer (``None`` =
        keep the full history).
    **kwargs : Any
        Forwarded to :class:`~asf.selectors.abstract_selector.AbstractSelector`.
    """

    #: How the evaluator scores a schedule by default: ``"primary"`` (the user
    #: gets the first algorithm's result) or ``"best_of_scheduled"`` (the user
    #: gets the best result among all scheduled algorithms, as with parallel runs).
    RESULT_MODE: str = "primary"

    def __init__(self, window_size: int | None = 200, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.window_size = None if window_size is None else int(window_size)
        self.buffer: ObservationBuffer | None = None
        self.step_: int = 0
        self.retrain_steps: list[int] = []
        self.retrain_seconds: float = 0.0
        self.drift_steps: list[int] = []

    # ------------------------------------------------------------------ fit
    def _fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """Seed the buffer with the warm-start data, then fit the initial model."""
        self.step_ = 0
        self.retrain_steps = []
        self.retrain_seconds = 0.0
        self.drift_steps = []
        self.buffer = ObservationBuffer(
            algorithms=self.algorithms,
            feature_names=list(features.columns),
            window_size=self.window_size,
        )
        self.buffer.extend_full(features, performance)
        self._fit_initial(features, performance, **kwargs)

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
            ``(algorithm, budget)`` tuples; the first is the primary choice.
        """
        self._check_fitted()
        if len(features) != 1:
            raise ValueError("propose() expects exactly one instance (one row).")
        features = self._augment(features)
        schedule = self._propose(features)
        if not schedule:
            raise RuntimeError("_propose() must return a non-empty schedule.")
        unknown = [a for a, _ in schedule if a not in self.algorithms]
        if unknown:
            raise ValueError(f"Schedule contains unknown algorithms: {unknown}")
        return schedule

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
        self.buffer.append(obs)
        self._observe(obs)  # step_ is the 0-based index of this instance
        self.step_ += 1

    # ------------------------------------------------------ AbstractSelector
    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Batch prediction without learning: propose per row, keep the primary."""
        if features is None:
            raise ValueError("Online selectors require features for prediction.")
        # features were already augmented by AbstractSelector.predict
        return {
            str(idx): self._propose(features.loc[[idx]])[:1] for idx in features.index
        }

    # ---------------------------------------------------------- hooks
    @abstractmethod
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        """Build the initial model from the (fully observed) warm-start data."""

    @abstractmethod
    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        """Return the schedule for one instance (single-row frame)."""

    @abstractmethod
    def _observe(self, observation: Observation) -> None:
        """Update the selector after feedback. The buffer is already updated."""

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


class _Timer:
    """Tiny wall-clock timer used as a context manager."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
