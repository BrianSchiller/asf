"""Composable (non-stationary, contextual) multi-armed bandit selector.

Arms are algorithms. A bandit is assembled from::

    context      features -> z (impute, standardise); x = [z, 1] or [1]
    posterior    per-arm Bayesian reward model     (posteriors.py)
    policy       posterior -> score (UCB / TS)     (policies.py)
    weighting    which past observations count     (weighting.py)
    partitioner  which cell of arm models to use   (partitioning.py)
    confidence   weight fixed at arrival           (confidence.py)
    reward       observation -> reward             (rewards.py)
    imputation   censored run -> imputed reward    (asf/online/imputation.py)

Each trace is kept *incrementally* if its weighting allows it (uniform or
exponential discount: update and decay running posteriors) and is *refitted*
from the weighted history at every decision otherwise. Both give the same
posteriors where both apply.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Collection, Hashable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from asf.online.base import AbstractOnlineSelector, Observation, random_argmax
from asf.online.gate import AbstractExtraRunGate, NoExtrasGate
from asf.online.imputation import CensoredImputer
from asf.online.prediction import PredictionRecord
from asf.online.selectors.bandits.confidence import ConfidenceModel
from asf.online.selectors.bandits.context import ContextBuilder, model_input
from asf.online.selectors.bandits.history import BanditHistory
from asf.online.selectors.bandits.partitioning import Partitioner, SinglePartition
from asf.online.selectors.bandits.policies import (
    ExplorationPolicy,
    ThompsonPolicy,
    UCBPolicy,
)
from asf.online.selectors.bandits.posteriors import Posterior
from asf.online.selectors.bandits.rewards import RewardTransform
from asf.online.selectors.bandits.weighting import Uniform, Weighting

PosteriorFactory = Callable[[int], Posterior]
Aggregate = Literal["mean", "min", "max"]


def _mean(posterior: Posterior, x: np.ndarray) -> float:
    """Posterior mean; ``inf`` without data, so greedy tries such arms first."""
    return posterior.mean(x) if np.isfinite(posterior.width(x)) else math.inf


class AbstractContextualBandit(AbstractOnlineSelector):
    """
    Bandit selector over algorithms, assembled from pluggable components.

    It plays one arm per step unless a ``gate`` adds extra runs, and learns
    from every arm reported back. In the record, ``candidates`` are the other
    arms by policy score; ``uncertainty`` (only with a gate) is how likely
    another arm is better: for TS the share of ``uncertainty_samples`` joint
    draws the chosen arm does not win, for UCB the overlap (IoU) of the
    intervals of the chosen arm and the first nominee (1 without data);
    ``predicted_runtime`` is the expected reward turned back into performance.

    Subclass hooks: ``_feature_vector``, ``_rewards``, ``_observation_weight``,
    ``_posterior_for``, ``_policy_time``; ``reset_arms()`` is the
    ``_reset_model`` used by ``ResetResponse``.

    Parameters
    ----------
    posterior : Callable[[int], Posterior]
        Factory of an empty posterior for an input dimension.
    policy : ExplorationPolicy
        ``UCBPolicy`` or ``ThompsonPolicy``.
    contextual : bool, default=True
        Posterior input ``[z, 1]`` or ``[1]``; weightings and partitioners
        see the features either way.
    weighting : Weighting or None, default=None
        Which past observations count (``None`` = ``Uniform()``).
    traces : Mapping[str, Weighting] or None, default=None
        Several named weightings instead of one (each with its own posteriors);
        their scores are combined by ``aggregate`` (f-dsw TS).
    partitioner : Partitioner or None, default=None
        ``None`` = one model per arm.
    confidence : ConfidenceModel or None, default=None
        Weights each observation by its confidence at arrival (do not share
        the model with a ``ConfidenceWeighting``: it would update twice).
    reward : RewardTransform or None, default=None
        ``None`` = ``-log10(PAR10 + 1)`` for runtimes, the raw score when
        maximizing.
    imputation : CensoredImputer or None, default=None
        Imputes censored rewards from the arm's posterior *before* the update
        instead of the PAR-k penalty (``DropCensored`` skips the update).
        Unsolved warm-start runs count as censored. Needs the ``"log"`` or
        ``"par"`` reward; ``PREDICTIVE`` rules need a posterior with
        ``noise_var``. Uses the first trace's posterior in the instance's cell.
    max_history : int or None, default=None
        Observations kept per arm in refit mode.
    primary : {"policy", "greedy"}, default="policy"
        ``"greedy"`` plays the best posterior mean and leaves exploration to
        the extra runs (nominees stay ordered by policy score; ``scores`` hold
        the means). Without a gate it never explores.
    **kwargs : Any
        Forwarded to :class:`~asf.online.base.AbstractOnlineSelector`
        (``budget`` is the cutoff used to score the warm start).
    """

    def __init__(
        self,
        posterior: PosteriorFactory,
        policy: ExplorationPolicy,
        *,
        contextual: bool = True,
        weighting: Weighting | None = None,
        traces: Mapping[str, Weighting] | None = None,
        aggregate: Aggregate = "mean",
        partitioner: Partitioner | None = None,
        confidence: ConfidenceModel | None = None,
        reward: RewardTransform | None = None,
        imputation: CensoredImputer | None = None,
        standardize: bool = True,
        max_history: int | None = None,
        primary: Literal["policy", "greedy"] = "policy",
        uncertainty_samples: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if primary not in ("policy", "greedy"):
            raise ValueError("primary must be 'policy' or 'greedy'.")
        if uncertainty_samples < 0:
            raise ValueError("uncertainty_samples must be >= 0.")
        self.primary = primary
        self.uncertainty_samples = int(uncertainty_samples)
        self.posterior = posterior
        self.policy = policy
        self.contextual = bool(contextual)
        if weighting is not None and traces is not None:
            raise ValueError("Pass either weighting or traces, not both.")
        if traces is None:
            traces = {"main": weighting if weighting is not None else Uniform()}
        if not traces:
            raise ValueError("traces must not be empty.")
        if aggregate not in ("mean", "min", "max"):
            raise ValueError("aggregate must be 'mean', 'min' or 'max'.")
        self.traces = dict(traces)
        self.aggregate = aggregate
        self.partitioner = partitioner if partitioner is not None else SinglePartition()
        self.confidence = confidence
        self.reward = reward if reward is not None else RewardTransform()
        if imputation is not None and not isinstance(imputation, CensoredImputer):
            raise TypeError("imputation must be a CensoredImputer or None.")
        self.imputation = imputation
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
        self.cutoff_ = self._warm_cutoff  # replaced by the stream's cutoff
        self._n_warm = len(features)
        self._check_imputation(self._warm_cutoff)
        perf = performance.reindex(columns=self.algorithms)
        for i in range(len(features)):
            row = perf.iloc[i]
            runtimes = {str(a): float(v) for a, v in row.items() if pd.notna(v)}
            if self.imputation is None:
                rewards = self.reward(runtimes, (), self._warm_cutoff, self.maximize)
            else:
                rewards = self._imputed_rewards(Z[i], runtimes, (), self._warm_cutoff)
            self._learn(Z[i], rewards, self._observation_weight(Z[i], None))

    # ------------------------------------------------------------- decisions
    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        z = self._feature_vector(features)
        x = model_input(z, self.contextual)[0]
        cell = self.partitioner.assign(z)
        t = self._policy_time(z, cell)
        # Built once: decision, uncertainty and prediction share the posteriors.
        posteriors = [
            [self._posterior_for(arm, z, cell, trace) for trace in self.traces]
            for arm in self.algorithms
        ]
        scores = np.array([self._score(p, x, t) for p in posteriors])
        ranking = np.argsort(-scores, kind="stable")  # by policy score
        if self.primary == "greedy":
            scores = np.array(
                [self._combine([_mean(q, x) for q in p]) for p in posteriors]
            )
        self.last_scores_ = dict(zip(self.algorithms, scores.tolist()))
        chosen = random_argmax(scores, self.rng_)
        others = [int(i) for i in ranking if i != chosen]
        uncertainty = None
        if others and not isinstance(self.gate_, NoExtrasGate):
            uncertainty = self._uncertainty(posteriors, chosen, others[0], x, t)
        model = posteriors[chosen][0]
        predicted = None
        if np.isfinite(model.width(x)):
            predicted = self.reward.to_performance(
                model.mean(x), self.cutoff_, self.maximize
            )
        arm = self.algorithms[chosen]
        return PredictionRecord(
            chosen=arm,
            budget=0.0,
            predicted_runtime=predicted,
            score=self.last_scores_[arm],
            uncertainty=uncertainty,
            scores=dict(self.last_scores_),
            candidates=tuple(self.algorithms[i] for i in others),
        )

    def _default_gate(self) -> AbstractExtraRunGate:
        """Bandits nominate arms but run one per step unless a gate is given."""
        return NoExtrasGate()

    def _uncertainty(
        self,
        posteriors: Sequence[Sequence[Posterior]],
        chosen: int,
        rival: int,
        x: np.ndarray,
        t: float,
    ) -> float | None:
        """How likely another arm is better than ``chosen`` (see class docs)."""
        if isinstance(self.policy, ThompsonPolicy):
            n = self.uncertainty_samples
            if n == 0:
                return None
            draws = np.column_stack(
                [
                    self._combine_draws(
                        [self.policy.score(q, x, t, self.rng_, n) for q in p]
                    )
                    for p in posteriors
                ]
            )
            top = draws == draws.max(axis=1, keepdims=True)
            return float(1.0 - (top[:, chosen] / top.sum(axis=1)).mean())
        if isinstance(self.policy, UCBPolicy):
            lo_c, hi_c = self._interval(posteriors[chosen], x, t)
            lo_r, hi_r = self._interval(posteriors[rival], x, t)
            if not np.all(np.isfinite([lo_c, hi_c, lo_r, hi_r])):
                return 1.0  # an arm without data: anything is possible
            union = max(hi_c, hi_r) - min(lo_c, lo_r)
            overlap = min(hi_c, hi_r) - max(lo_c, lo_r)
            return 1.0 if union <= 0 else max(overlap, 0.0) / union
        return None

    def _interval(
        self, posteriors: Sequence[Posterior], x: np.ndarray, t: float
    ) -> tuple[float, float]:
        """UCB confidence interval ``mean -+ bonus`` of one arm (traces combined)."""
        upper = [self.policy.score(q, x, t, self.rng_) for q in posteriors]
        lower = [2.0 * q.mean(x) - u for q, u in zip(posteriors, upper)]
        return self._combine(lower), self._combine(upper)

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
        """Observation -> reward per executed arm (censored runs imputed if
        ``imputation`` is set)."""
        self._check_reward_scale(observation.cutoff)
        self.cutoff_ = observation.cutoff
        if self.imputation is None:
            return self.reward(
                observation.runtimes,
                observation.censored,
                observation.cutoff,
                self.maximize,
            )
        self._check_imputation(observation.cutoff)
        return self._imputed_rewards(
            self._feature_vector(observation.features),
            observation.runtimes,
            observation.censored,
            observation.cutoff,
        )

    def _imputed_rewards(
        self,
        z: np.ndarray,
        runtimes: Mapping[str, float],
        censored: Collection[str],
        cutoff: float,
    ) -> dict[str, float]:
        """
        Rewards with censored runs imputed (Tornede et al., Alg. 1 l. 17-20).

        A run is censored if marked so or its runtime reached the cutoff; its
        true reward is at most ``solved_reward(min(runtime, cutoff))``. The
        rule works on the negated reward with the belief ``N(-mean, sd^2)``
        of the arm's posterior at ``z``; ``NaN`` drops the arm.
        """
        assert self.imputation is not None
        unsolved = [
            a
            for a in self.algorithms
            if a in runtimes and (a in censored or float(runtimes[a]) >= cutoff)
        ]
        solved = {a: v for a, v in runtimes.items() if a not in unsolved}
        rewards = self.reward(solved, (), cutoff, self.maximize)
        if not unsolved:
            return rewards
        bound = np.array(
            [
                -self.reward.solved_reward(min(float(runtimes[a]), cutoff), cutoff)
                for a in unsolved
            ]
        )
        mean = np.zeros(len(unsolved))
        sd = np.zeros(len(unsolved))
        if self.imputation.USES_MODEL:
            x = model_input(z, self.contextual)[0]
            cell = self.partitioner.assign(z)
            for k, arm in enumerate(unsolved):
                posterior = self._posterior_for(arm, z, cell)
                mean[k] = -posterior.mean(x)
                sd[k] = posterior.width(x)
                if self.imputation.PREDICTIVE:
                    sd[k] = math.sqrt(sd[k] ** 2 + float(posterior.noise_var))
        values = self.imputation(bound, mean, sd, self.rng_)
        for arm, value in zip(unsolved, values):
            if not np.isnan(value):
                rewards[arm] = -float(value)
        return rewards

    def _check_imputation(self, cutoff: float) -> None:
        """Raise if ``imputation`` cannot work with the reward / posterior."""
        if self.imputation is None:
            return
        if self.maximize:
            raise ValueError("Censored imputation is for runtime minimization.")
        kind = self.reward.resolve(cutoff, self.maximize)
        if kind not in ("log", "par"):
            raise ValueError(
                f"Censored imputation needs the 'log' or 'par' reward, not {kind!r}."
            )
        if self.imputation.PREDICTIVE and not hasattr(
            self.posterior(self.input_dim_), "noise_var"
        ):
            raise ValueError(
                f"{self.imputation!r} needs a posterior with noise_var "
                "(GaussianPosterior or BayesianLinearRegression)."
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

    def _score(self, posteriors: Sequence[Posterior], x: np.ndarray, t: float) -> float:
        """Score of one arm: one policy score per trace, then aggregated."""
        return self._combine(
            [self.policy.score(q, x, t, self.rng_) for q in posteriors]
        )

    def _combine(self, values: Sequence[float]) -> float:
        """Combine one value per trace with ``aggregate``."""
        if len(values) == 1:
            return values[0]
        return float(getattr(np, self.aggregate)(values))

    def _combine_draws(self, draws: Sequence[np.ndarray]) -> np.ndarray:
        """:meth:`_combine` per draw (``draws``: one array per trace)."""
        if len(draws) == 1:
            return draws[0]
        return getattr(np, self.aggregate)(np.stack(draws), axis=0)

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
