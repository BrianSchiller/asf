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
e.g. ``LinTSSelector(weighting=ExponentialDiscount(0.99))``; new methods
subclass this class and override one of the hooks listed in its docstring.

Computation modes
-----------------
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
from typing import Any, Callable, Hashable

import numpy as np
import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.selectors.bandits.confidence import ConfidenceModel
from asf.online.selectors.bandits.context import ContextBuilder, model_input
from asf.online.selectors.bandits.history import BanditHistory
from asf.online.selectors.bandits.partitioning import Partitioner, SinglePartition
from asf.online.selectors.bandits.policies import ExplorationPolicy
from asf.online.selectors.bandits.posteriors import Posterior
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.weighting import Uniform, Weighting

PosteriorFactory = Callable[[int], Posterior]


class AbstractContextualBandit(AbstractOnlineSelector):
    """
    Bandit selector over algorithms, assembled from pluggable components.

    Every step it plays exactly one arm (the schedule ``[(arm, 0.0)]``, i.e.
    run with the scenario cutoff) and learns from all arms reported in the
    observation.

    Hooks for subclasses
    --------------------
    ``_feature_vector(features)``
        Instance -> ``z`` (default: imputed, standardised features).
    ``_rewards(observation)``
        Observation -> reward per executed arm (e.g. censored imputation).
    ``_observation_weight(z, observation)``
        Weight stored with an observation (default: confidence score or 1).
    ``_posterior_for(arm, z, cell)``
        The posterior used to score ``arm`` for the current instance.
    ``reset_arms()``
        Forget all observations, e.g. when a drift detector fires.

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
        self.weighting = weighting if weighting is not None else Uniform()
        self.partitioner = partitioner if partitioner is not None else SinglePartition()
        self.confidence = confidence
        self.reward = reward if reward is not None else RewardTransform()
        self.standardize = bool(standardize)
        self.max_history = max_history

    @property
    def incremental(self) -> bool:
        """Whether posteriors are maintained incrementally (vs. refitted)."""
        return bool(self.weighting.incremental)

    # ------------------------------------------------------------- warm start
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        self.context_ = ContextBuilder(self.standardize).fit(features)
        self.input_dim_ = self.context_.dim + 1 if self.contextual else 1
        self.history_ = BanditHistory(
            self.algorithms, self.context_.dim, self.max_history
        )
        self.posteriors_: dict[Hashable, dict[str, Posterior]] = {}
        self.clock_ = 0
        self.rng_ = np.random.default_rng(self.random_state)
        self.last_scores_: dict[str, float] = {}
        self._warned_cutoff = False

        Z = (
            self.context_.transform(features)
            if len(features)
            else np.zeros((0, self.context_.dim))
        )
        self.weighting.fit(Z)
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
    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        z = self._feature_vector(features)
        x = model_input(z, self.contextual)[0]
        cell = self.partitioner.assign(z)
        scores = np.array(
            [
                self.policy.score(
                    self._posterior_for(arm, z, cell), x, self.clock_, self.rng_
                )
                for arm in self.algorithms
            ]
        )
        self.last_scores_ = dict(zip(self.algorithms, scores.tolist()))
        best = np.flatnonzero(scores == scores.max())
        arm = self.algorithms[int(self.rng_.choice(best))]
        return [(arm, 0.0)]

    def _observe(self, observation: Observation) -> None:
        z = self._feature_vector(observation.features)
        rewards = self._rewards(observation)
        self._learn(z, rewards, self._observation_weight(z, observation))
        # Let the adaptive components see the new instance (after it was
        # scored, so a confidence weight reflects the model *before* it).
        self.weighting.update(z, self.clock_)
        self.partitioner.update(z)
        if self.confidence is not None:
            self.confidence.update(z)

    def reset_arms(self) -> None:
        """Forget all observations of all arms (the clock keeps running)."""
        self.history_.clear()
        self.posteriors_ = {}

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

    def _posterior_for(self, arm: str, z: np.ndarray, cell: Hashable) -> Posterior:
        """Posterior of ``arm`` for the current instance (``z`` in ``cell``)."""
        if self.incremental:
            return self._cell_posteriors(cell)[arm]
        data = self.history_.get(arm, cell)
        weights = data.weights * self.weighting.weights(
            z, self.clock_, data.steps, data.Z
        )
        posterior = self.posterior(self.input_dim_)
        posterior.fit(model_input(data.Z, self.contextual), data.rewards, weights)
        return posterior

    # ------------------------------------------------------------- internals
    def _cell_posteriors(self, cell: Hashable) -> dict[str, Posterior]:
        if cell not in self.posteriors_:
            self.posteriors_[cell] = {
                arm: self.posterior(self.input_dim_) for arm in self.algorithms
            }
        return self.posteriors_[cell]

    def _learn(self, z: np.ndarray, rewards: dict[str, float], weight: float) -> None:
        """Store one instance's rewards and advance the clock by one."""
        cell = self.partitioner.assign(z)
        if self.incremental:
            gamma = self.weighting.decay_factor
            if gamma != 1.0:
                for models in self.posteriors_.values():
                    for posterior in models.values():
                        posterior.decay(gamma)
            x = model_input(z, self.contextual)[0]
            models = self._cell_posteriors(cell)
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
