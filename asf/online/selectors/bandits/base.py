"""Composable (non-stationary, contextual) multi-armed bandit selector.

Arms are algorithms. A bandit selector is assembled from independent parts::

    context      features -> z (impute, standardise); x = [z, 1] or [1]
    posterior    per-arm Bayesian reward model     (posteriors.py)
    policy       posterior -> score (UCB / TS)     (policies.py)
    weighting    which past observations count     (weighting.py)
    partitioner  which cell of arm models to use   (partitioning.py)
    confidence   weight fixed at arrival           (confidence.py)
    reward       observation -> reward             (rewards.py)

UCB, TS, LinUCB and LinTS (``ucb.py``, ``thompson.py``) are presets of this
class. Non-stationary variants are obtained by passing a different weighting,
e.g. ``LinTSSelector(weighting=ExponentialDiscount(0.99))``, or several
weightings at once (``traces``), whose scores are combined per arm (f-dsw TS);
new methods
subclass this class and override one of the hooks listed in its docstring.

Computation modes
-----------------
Each trace is computed in one of two modes:

* incremental -- if the weighting is incremental (uniform / exponential
  discount), every (cell, arm) keeps a posterior that is updated with each
  observation and decayed once per step.
* refit -- otherwise (sliding window, nearest neighbours, re-estimated
  confidence weights), an arm's posterior is rebuilt from the weighted history
  at every decision.

Both modes give identical posteriors whenever both are applicable.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.prediction import PredictionRecord
from asf.online.selectors.bandits.confidence import ConfidenceModel
from asf.online.selectors.bandits.context import ContextBuilder, model_input
from asf.online.selectors.bandits.history import BanditHistory
from asf.online.selectors.bandits.partitioning import Partitioner, SinglePartition
from asf.online.selectors.bandits.policies import ExplorationPolicy
from asf.online.selectors.bandits.posteriors import Posterior
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.weighting import Uniform, Weighting

PosteriorFactory = Callable[[int], Posterior]
Aggregate = Callable[[Sequence[float]], float]

#: Named ways of combining the scores of several traces.
AGGREGATES: dict[str, Aggregate] = {
    "min": lambda v: float(min(v)),
    "max": lambda v: float(max(v)),
    "mean": lambda v: float(np.mean(v)),
}


class AbstractContextualBandit(AbstractOnlineSelector):
    """
    Bandit selector over algorithms, assembled from pluggable components.

    Every step it plays exactly one arm (the schedule ``[(arm, 0.0)]``, i.e.
    run with the scenario cutoff) and learns from all arms reported in the
    observation. It nominates no extra runs; a ``gate`` can add some (e.g.
    from ``GateContext.ranked_alternatives()``, ordered by the arm scores).

    Hooks for subclasses
    --------------------
    ``_feature_vector(features)``
        Instance -> ``z`` (default: imputed, standardised features).
    ``_rewards(observation)``
        Observation -> reward per executed arm (e.g. censored imputation).
    ``_observation_weight(z, observation)``
        Weight stored with an observation (default: confidence score or 1).
    ``_posterior_for(arm, z, cell, trace)``
        The posterior used to score ``arm`` in one trace.
    ``_policy_time(z, cell)``
        The time ``t`` the policy sees (e.g. a discounted play count).
    ``reset_arms()``
        Forget all observations. It is the bandits' ``_reset_model``, so
        ``drift_response=ResetResponse()`` calls it when a detector fires.

    Parameters
    ----------
    posterior : Callable[[int], Posterior]
        Factory creating an empty posterior for an input dimension, e.g.
        ``functools.partial(BayesianLinearRegression, prior_precision=1.0)``.
    policy : ExplorationPolicy
        ``UCBPolicy`` or ``ThompsonPolicy``.
    contextual : bool, default=True
        Posterior input ``[z, 1]`` (contextual) or ``[1]`` (context-free).
        Features are still used by weightings / partitioners either way.
    weighting : Weighting or None, default=None
        Weighting of past observations; ``None`` = ``Uniform()`` (stationary).
    traces : Mapping[str, Weighting] or None, default=None
        Several named weightings instead of one, e.g. ``{"historic":
        ExponentialDiscount(0.99), "hot": LastPulls(30)}``. Every trace has its
        own posterior per arm; the policy scores each and ``aggregate``
        combines the scores. Mutually exclusive with ``weighting``.
    aggregate : {"mean", "min", "max"} or callable, default="mean"
        How the scores of several traces are combined (unused for one trace).
    partitioner : Partitioner or None, default=None
        ``None`` = ``SinglePartition()`` (one model per arm).
    confidence : ConfidenceModel or None, default=None
        If given, each observation is weighted by its confidence score at
        arrival (fixed afterwards). The model is fitted on the warm start and
        updated with every streamed instance. Do not pass the same model
        object to a ``ConfidenceWeighting`` as well (it would be updated twice).
    reward : RewardTransform or None, default=None
        ``None`` = ``RewardTransform("auto")`` (``-log10(PAR10 + 1)`` for
        runtimes, the raw score when maximizing).
    standardize : bool, default=True
        Standardise features with warm-start statistics.
    max_history : int or None, default=None
        Observations kept per arm in refit mode (``None`` = all).
    **kwargs : Any
        Forwarded to :class:`~asf.online.base.AbstractOnlineSelector`
        (e.g. ``budget`` -- the cutoff used to score the warm start --,
        ``maximize``, ``random_state``).
    """

    def __init__(
        self,
        posterior: PosteriorFactory,
        policy: ExplorationPolicy,
        *,
        contextual: bool = True,
        weighting: Weighting | None = None,
        traces: Mapping[str, Weighting] | None = None,
        aggregate: str | Aggregate = "mean",
        partitioner: Partitioner | None = None,
        confidence: ConfidenceModel | None = None,
        reward: RewardTransform | None = None,
        standardize: bool = True,
        max_history: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.posterior = posterior
        self.policy = policy
        self.contextual = bool(contextual)
        if weighting is not None and traces is not None:
            raise ValueError("Pass either weighting or traces, not both.")
        if traces is None:
            traces = {"main": weighting if weighting is not None else Uniform()}
        if not traces:
            raise ValueError("traces must not be empty.")
        if isinstance(aggregate, str) and aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {list(AGGREGATES)}.")
        self.traces = dict(traces)
        self.aggregate = aggregate
        self.partitioner = partitioner if partitioner is not None else SinglePartition()
        self.confidence = confidence
        self.reward = reward if reward is not None else RewardTransform()
        self.standardize = bool(standardize)
        self.max_history = max_history

    @property
    def weighting(self) -> Weighting:
        """The weighting of a single-trace bandit."""
        if len(self.traces) != 1:
            raise AttributeError("This bandit has several traces; use .traces.")
        return next(iter(self.traces.values()))

    @property
    def incremental(self) -> bool:
        """Whether all traces are maintained incrementally (vs. refitted)."""
        return all(w.incremental for w in self.traces.values())

    # ------------------------------------------------------------- warm start
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        self.context_ = ContextBuilder(self.standardize).fit(features)
        self.input_dim_ = self.context_.dim + 1 if self.contextual else 1
        self.history_ = BanditHistory(
            self.algorithms, self.context_.dim, self.max_history
        )
        # (cell, trace) -> arm -> posterior, for incremental traces only
        self.posteriors_: dict[tuple[Hashable, str], dict[str, Posterior]] = {}
        self.clock_ = 0
        self.rng_ = np.random.default_rng(self.random_state)
        self.last_scores_: dict[str, float] = {}
        self._warned_cutoff = False

        Z = (
            self.context_.transform(features)
            if len(features)
            else np.zeros((0, self.context_.dim))
        )
        for weighting in self.traces.values():
            weighting.fit(Z)
        self.partitioner.fit(Z)
        if self.confidence is not None:
            self.confidence.fit(Z)

        # The warm start is fully observed: every arm learns from every row.
        self._warm_cutoff = self.budget if self.budget is not None else math.inf
        self._n_warm = len(features)
        perf = performance.reindex(columns=self.algorithms)
        for i in range(len(features)):
            row = perf.iloc[i]
            runtimes = {str(a): float(v) for a, v in row.items() if pd.notna(v)}
            rewards = self.reward(runtimes, (), self._warm_cutoff, self.maximize)
            self._learn(Z[i], rewards, self._observation_weight(Z[i], None))

    # ------------------------------------------------------------- decisions
    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        z = self._feature_vector(features)
        x = model_input(z, self.contextual)[0]
        cell = self.partitioner.assign(z)
        t = self._policy_time(z, cell)
        scores = np.array([self._score(arm, z, x, cell, t) for arm in self.algorithms])
        self.last_scores_ = dict(zip(self.algorithms, scores.tolist()))
        best = np.flatnonzero(scores == scores.max())
        arm = self.algorithms[int(self.rng_.choice(best))]
        return PredictionRecord(
            chosen=arm,
            budget=0.0,
            score=self.last_scores_[arm],
            scores=dict(self.last_scores_),
        )

    def _observe(self, observation: Observation) -> None:
        z = self._feature_vector(observation.features)
        rewards = self._rewards(observation)
        self._learn(z, rewards, self._observation_weight(z, observation))
        # Let the adaptive components see the new instance (after it was
        # scored, so a confidence weight reflects the model *before* it).
        for weighting in self.traces.values():
            weighting.update(z, self.clock_)
        self.partitioner.update(z)
        if self.confidence is not None:
            self.confidence.update(z)

    def reset_arms(self) -> None:
        """Forget all observations of all arms (the clock keeps running)."""
        self.history_.clear()
        self.posteriors_ = {}

    def _reset_model(self) -> None:
        """Drift-response hook: :meth:`reset_arms`."""
        self.reset_arms()

    # ------------------------------------------------------------- hooks
    def _feature_vector(self, features: pd.DataFrame) -> np.ndarray:
        """Single-row feature frame -> feature vector ``z``."""
        return self.context_.transform(features)[0]

    def _rewards(self, observation: Observation) -> dict[str, float]:
        """Observation -> reward per executed arm."""
        self._check_reward_scale(observation.cutoff)
        return self.reward(
            observation.runtimes,
            observation.censored,
            observation.cutoff,
            self.maximize,
        )

    def _observation_weight(
        self, z: np.ndarray, observation: Observation | None
    ) -> float:
        """Weight stored with an observation (``observation`` is ``None`` for
        warm-start rows)."""
        if self.confidence is None:
            return 1.0
        return float(self.confidence.score(z[None, :])[0])

    def _policy_time(self, z: np.ndarray, cell: Hashable) -> float:
        """Time ``t`` passed to the policy (UCB1's ``ln t``); default: the clock."""
        return float(self.clock_)

    def _score(
        self, arm: str, z: np.ndarray, x: np.ndarray, cell: Hashable, t: float
    ) -> float:
        """Score of ``arm``: one policy score per trace, then aggregated."""
        values = [
            self.policy.score(self._posterior_for(arm, z, cell, trace), x, t, self.rng_)
            for trace in self.traces
        ]
        if len(values) == 1:
            return values[0]
        combine = (
            AGGREGATES[self.aggregate]
            if isinstance(self.aggregate, str)
            else self.aggregate
        )
        return combine(values)

    def _posterior_for(
        self, arm: str, z: np.ndarray, cell: Hashable, trace: str | None = None
    ) -> Posterior:
        """Posterior of ``arm`` in ``trace`` (default: the first) for the
        current instance (``z`` in ``cell``)."""
        trace = trace if trace is not None else next(iter(self.traces))
        weighting = self.traces[trace]
        if weighting.incremental:
            return self._cell_posteriors(cell, trace)[arm]
        data = self.history_.get(arm, cell)
        weights = data.weights * weighting.weights(z, self.clock_, data.steps, data.Z)
        posterior = self.posterior(self.input_dim_)
        posterior.fit(model_input(data.Z, self.contextual), data.rewards, weights)
        return posterior

    # ------------------------------------------------------------- internals
    def _cell_posteriors(self, cell: Hashable, trace: str) -> dict[str, Posterior]:
        key = (cell, trace)
        if key not in self.posteriors_:
            self.posteriors_[key] = {
                arm: self.posterior(self.input_dim_) for arm in self.algorithms
            }
        return self.posteriors_[key]

    def _learn(self, z: np.ndarray, rewards: dict[str, float], weight: float) -> None:
        """Store one instance's rewards and advance the clock by one."""
        cell = self.partitioner.assign(z)
        x = model_input(z, self.contextual)[0]
        for trace, weighting in self.traces.items():
            if not weighting.incremental:
                continue  # refitted from the history when needed
            gamma = weighting.decay_factor
            models = self._cell_posteriors(cell, trace)
            if gamma != 1.0 and weighting.decay_on_update:
                for arm in rewards:  # per pull: only what is updated now
                    models[arm].decay(gamma)
            elif gamma != 1.0:
                for (_, t), cell_models in self.posteriors_.items():
                    if t == trace:  # per step: every arm in every cell
                        for posterior in cell_models.values():
                            posterior.decay(gamma)
            for arm, r in rewards.items():
                models[arm].update(x, r, weight)
        for arm, r in rewards.items():
            self.history_.add(arm, self.clock_, z, cell, r, weight)
        self.clock_ += 1

    def _check_reward_scale(self, cutoff: float) -> None:
        """Warn once if warm start and stream rewards are on different scales."""
        if self._warned_cutoff or self._n_warm == 0:
            return
        kind = self.reward.resolve(cutoff, self.maximize)
        # Runtime rewards depend on the cutoff (it decides what is unsolved).
        if kind != "raw" and cutoff != self._warm_cutoff:
            warnings.warn(
                f"Warm start was scored with cutoff={self._warm_cutoff} but the "
                f"stream uses cutoff={cutoff}; pass budget=<cutoff> to the "
                "selector so both use the same reward scale.",
                stacklevel=3,
            )
            self._warned_cutoff = True
