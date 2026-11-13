"""
Online selector with one runtime model per algorithm (Degroote et al., 2018).

Every algorithm gets its own regression model (an empirical performance
model, EPM) that predicts its performance from the instance features. Each
model learns only from the instances on which its algorithm was run, so the
partially observed online data needs no imputation. The selector picks the
algorithm with the best prediction (greedy), occasionally a random one
(epsilon-greedy), or the best optimistic prediction (UCB).

Reference
---------
Degroote, H., Bischl, B., Kotthoff, L., De Causmaecker, P. (2018).
"A Regression-Based Methodology for Online Algorithm Selection". SoCS 2018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import BaseForest
from sklearn.impute import SimpleImputer

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.gate import AbstractExtraRunGate, NeverProbe
from asf.online.prediction import PredictionRecord
from asf.utils.random_state import clone_with_random_state

Strategy = Literal["greedy", "epsilon_greedy", "ucb"]
Uncertainty = Literal["jackknife", "tree_std"]
TargetTransform = Literal["none", "log10"]

#: Runtimes are clipped to at least this value before taking logs.
LOG_FLOOR = 1e-6


def default_estimator() -> RandomForestRegressor:
    """
    The random forest of Degroote et al.: R ``randomForest`` defaults.

    500 trees, ``mtry = p/3``, bootstrap sampling with replacement and
    ``nodesize = 5``. R only splits nodes with more than ``nodesize``
    instances, which is sklearn's ``min_samples_split=6``.
    """
    return RandomForestRegressor(
        n_estimators=500, max_features=1 / 3, min_samples_split=6, bootstrap=True
    )


def jackknife_after_bootstrap_sd(
    tree_preds: np.ndarray, oob: np.ndarray, bias_correction: bool = True
) -> np.ndarray:
    """
    Jackknife-after-bootstrap standard error of a bagged prediction.

    This is the estimator mlr's ``regr.randomForest`` uses for
    ``se.method="jackknife"`` (Sexton and Laake, 2009; with the Monte-Carlo
    bias correction of Wager, Hastie and Efron, 2014)::

        V = (n-1)/n * sum_i (mean_{b: i out-of-bag} t_b(x) - t(x))^2
            - (e-1) * n / B^2 * sum_b (t_b(x) - t(x))^2

    where ``t(x)`` is the mean over all ``B`` trees and ``n`` the number of
    training instances. Instances that are in-bag for every tree are left out
    of the sum (as in mlr). Negative variances are set to 0.

    Parameters
    ----------
    tree_preds : np.ndarray
        Per-tree predictions, shape ``(n_queries, B)``.
    oob : np.ndarray
        Boolean out-of-bag mask, shape ``(n, B)``: ``True`` if training
        instance ``i`` was not drawn for tree ``b``.
    bias_correction : bool, default=True
        Subtract the Monte-Carlo bias term.

    Returns
    -------
    np.ndarray
        Standard error per query, shape ``(n_queries,)``.
    """
    tree_preds = np.atleast_2d(np.asarray(tree_preds, dtype=float))
    oob = np.asarray(oob, dtype=bool)
    n, n_trees = oob.shape
    if n < 2:
        return np.zeros(len(tree_preds))
    aggregate = tree_preds.mean(axis=1, keepdims=True)
    counts = oob.sum(axis=1)
    used = counts > 0
    oob_means = tree_preds @ oob[used].T.astype(float) / counts[used]
    variance = (n - 1) / n * ((oob_means - aggregate) ** 2).sum(axis=1)
    if bias_correction:
        spread = ((tree_preds - aggregate) ** 2).sum(axis=1)
        variance -= (math.e - 1) * n / n_trees**2 * spread
    return np.sqrt(np.maximum(variance, 0.0))


@dataclass
class _AlgorithmModel:
    """A fitted model of one algorithm and what it was fitted on."""

    imputer: SimpleImputer
    estimator: Any
    oob: np.ndarray | None  # (n, B) out-of-bag mask, only for the jackknife
    signature: tuple[int, int, float]  # first/last training row, cutoff
    n_rows: int


class OnlineEPMSelector(AbstractOnlineSelector):
    """
    Regression-based online selector of Degroote et al. (2018).

    One regression model per algorithm predicts its performance (PAR-k
    runtime, or the quality score for ``maximize=True``). After every
    instance the observed performance of the executed algorithms is added to
    the training data (Algorithm 1 of the paper) and the models are updated
    every ``retrain_every`` new data points (the paper's experiments: 10).

    Selection strategies (Section "Online Algorithm Selection as a
    Multi-armed Bandit"):

    * ``"greedy"`` (Algorithm 2): the algorithm with the best prediction.
    * ``"epsilon_greedy"``: with probability ``epsilon`` a uniformly random
      algorithm (the predicted best included, so each algorithm has
      probability ``epsilon / |A|``), otherwise greedy.
    * ``"ucb"``: the best ``prediction + gamma * sd`` with performance as
      "higher is better"; for runtimes this is the lowest
      ``prediction - gamma * sd``.

    Like the paper, the selector runs one algorithm per instance (default
    gate :class:`~asf.online.gate.NeverProbe`). The paper's ``allData``
    upper bound is ``gate=AlwaysProbe()`` (all other algorithms are nominated)
    evaluated with ``result_mode="primary"``.

    Prediction record: ``predicted_runtime`` is the predicted PAR-k value of
    the chosen algorithm on the raw scale, ``uncertainty`` its standard error
    on the model's target scale, ``scores`` the selection index of every
    modelled algorithm (higher = preferred) and ``candidates`` all other
    algorithms, unmodelled ones first, then by descending score.

    Assumptions and deviations
    --------------------------
    * Assumption (uncertainty): the paper estimates standard deviations with
      the Jackknife (Sexton and Laake, 2009) via mlr; implemented as mlr does
      (:func:`jackknife_after_bootstrap_sd`). With one training instance the
      standard error is 0.
    * Assumption (timeouts): a run is unsolved if it was censored or its
      runtime exceeds the cutoff (as in the evaluator's PAR-k); its target is
      ``par_factor * cutoff``. The cutoff is that of the last observation,
      before the first one ``budget``. Subclasses (e.g. censored imputation)
      override :meth:`_training_targets`.
    * Deviation (missing features): the paper imputes the mean over the whole
      scenario, which would use future instances; here each model imputes the
      mean of its training instances (0 for features that are always missing).
    * Assumption (update schedule): the counter of new data points includes
      extra runs; once it reaches ``retrain_every``, every model whose training
      data changed is refitted and the counter restarts at 0. ``retrain_every=1``
      is Algorithm 1 literally. These updates are part of the normal learning
      and are not logged as retrainings (their time is in ``selector_seconds``).
    * Assumption (cold start): the paper does not say how algorithms without
      data are handled. They are tried first, uniformly at random among them;
      an algorithm gets a model as soon as it has its first observation. If
      the chosen algorithm has no model, the record's predictions are ``None``.
    * Assumption (ties): exact ties are broken uniformly at random.
    * Assumption (window): the paper keeps all data (``window_size=None``).
      With a window, an algorithm whose data left it loses its model at the
      next update and counts as unmodelled again.

    Parameters
    ----------
    strategy : {"greedy", "epsilon_greedy", "ucb"}, default="greedy"
        Selection strategy; greedy performed best in the paper.
    epsilon : float, default=0.05
        Exploration probability of ``"epsilon_greedy"`` (paper: 0.05).
    gamma : float, default=1.0
        Weight of the standard error in ``"ucb"`` (paper: 1).
    estimator : sklearn regressor or None, default=None
        Template cloned for every algorithm; ``None`` = :func:`default_estimator`.
        With ``uncertainty`` set it must be a forest (``RandomForestRegressor``,
        ``ExtraTreesRegressor``), for ``"jackknife"`` with ``bootstrap=True``.
        Set ``n_jobs`` on it to fit in parallel.
    uncertainty : {"jackknife", "tree_std"} or None, default="jackknife"
        Standard error of a prediction: jackknife-after-bootstrap (paper) or
        the standard deviation across trees. ``None`` skips it (faster; not
        allowed with ``"ucb"``).
    jackknife_bias_correction : bool, default=True
        Monte-Carlo bias correction of the jackknife (as in mlr).
    target_transform : {"none", "log10"}, default="none"
        Scale the models are trained on: the raw performance (paper) or its
        log10 (runtimes clipped to ``LOG_FLOOR``). Not for ``maximize=True``.
    par_factor : float, default=10.0
        Penalty factor for unsolved runs (PAR10 in the paper).
    retrain_every : int, default=10
        Number of new data points after which the models are updated.
    window_size : int or None, default=None
        Instances kept as training data; ``None`` = all (paper).
    **kwargs : Any
        Forwarded to :class:`~asf.online.base.AbstractOnlineSelector`
        (``gate``, ``detector``, ``drift_response``, ``retrain_policy``,
        ``budget`` -- the cutoff of the warm start --, ``maximize``,
        ``random_state``, ...).
    """

    def __init__(
        self,
        strategy: Strategy = "greedy",
        epsilon: float = 0.05,
        gamma: float = 1.0,
        estimator: Any | None = None,
        uncertainty: Uncertainty | None = "jackknife",
        jackknife_bias_correction: bool = True,
        target_transform: TargetTransform = "none",
        par_factor: float = 10.0,
        retrain_every: int = 10,
        window_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(window_size=window_size, **kwargs)
        if strategy not in ("greedy", "epsilon_greedy", "ucb"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be >= 0.")
        if uncertainty not in ("jackknife", "tree_std", None):
            raise ValueError(f"Unknown uncertainty: {uncertainty!r}")
        if strategy == "ucb" and uncertainty is None:
            raise ValueError("strategy='ucb' needs an uncertainty estimate.")
        if target_transform not in ("none", "log10"):
            raise ValueError(f"Unknown target_transform: {target_transform!r}")
        if target_transform == "log10" and self.maximize:
            raise ValueError("target_transform='log10' is for runtime minimization.")
        if not par_factor > 0:
            raise ValueError("par_factor must be > 0.")
        if retrain_every < 1:
            raise ValueError("retrain_every must be >= 1.")
        template = estimator if estimator is not None else default_estimator()
        if uncertainty is not None and not isinstance(template, BaseForest):
            raise ValueError("uncertainty needs a forest estimator (BaseForest).")
        if uncertainty == "jackknife" and not template.bootstrap:
            raise ValueError("uncertainty='jackknife' needs bootstrap=True.")

        self.strategy = strategy
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.estimator = estimator
        self.uncertainty = uncertainty
        self.jackknife_bias_correction = bool(jackknife_bias_correction)
        self.target_transform = target_transform
        self.par_factor = float(par_factor)
        self.retrain_every = int(retrain_every)
        self.estimator_ = template

    # ------------------------------------------------------------ fitting
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        self.rng_ = np.random.default_rng(self.random_state)
        self.cutoff_ = self.budget if self.budget else math.inf
        self.models_: dict[str, _AlgorithmModel] = {}
        self.n_rows_seen_ = len(features)  # rows ever added to the buffer
        self.first_row_ = 0  # rows before this one are forgotten (reset)
        self.pending_ = 0  # new data points since the last update
        self._update_models(only_missing=False)

    def _observe(self, observation: Observation) -> None:
        if math.isfinite(observation.cutoff):
            self.cutoff_ = float(observation.cutoff)
        self.n_rows_seen_ += 1
        self.pending_ += len(observation.runtimes)
        if self.pending_ >= self.retrain_every:
            self.pending_ = 0
            self._update_models(only_missing=False)
        elif any(str(a) not in self.models_ for a in observation.runtimes):
            self._update_models(only_missing=True)

    def _retrain(self) -> None:
        """Update every model whose training data changed, now."""
        self.pending_ = 0
        self._update_models(only_missing=False)

    def _reset_model(self) -> None:
        """Forget all models and all data seen so far (back to a cold start)."""
        self.models_ = {}
        self.first_row_ = self.n_rows_seen_
        self.pending_ = 0

    def _default_gate(self) -> AbstractExtraRunGate:
        """One run per instance, as in the paper."""
        return NeverProbe()

    def _training_targets(
        self, performance: pd.DataFrame, censored: pd.DataFrame
    ) -> np.ndarray:
        """
        Training targets per instance and algorithm (``NaN`` = not run).

        Unsolved runs (censored, or runtime above the cutoff) get
        ``par_factor * cutoff``; then ``target_transform`` is applied.
        """
        values = performance.to_numpy(dtype=float, copy=True)
        if math.isfinite(self.cutoff_):
            with np.errstate(invalid="ignore"):
                unsolved = censored.to_numpy(dtype=bool) | (values > self.cutoff_)
            values[unsolved & ~np.isnan(values)] = self.par_factor * self.cutoff_
        if self.target_transform == "log10":
            values = np.log10(np.maximum(values, LOG_FLOOR))
        return values

    def _training_data(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Features, targets and the row number of the first training row."""
        assert self.buffer is not None
        first = self.n_rows_seen_ - len(self.buffer)
        skip = max(0, self.first_row_ - first)
        X = self.buffer.features().to_numpy(dtype=float)[skip:]
        Y = self._training_targets(
            self.buffer.performance().iloc[skip:],
            self.buffer.censored_mask().iloc[skip:],
        )
        return X, Y, first + skip

    def _update_models(self, only_missing: bool) -> None:
        """
        Fit models of algorithms without one and, unless ``only_missing``,
        refit models whose training data changed; drop models without data.
        """
        X, Y, first = self._training_data()
        rows = np.arange(len(X)) + first
        for j, algo in enumerate(self.algorithms):
            observed = ~np.isnan(Y[:, j])
            current = self.models_.get(algo)
            if only_missing and current is not None:
                continue
            if not observed.any():
                self.models_.pop(algo, None)
                continue
            seen = rows[observed]
            signature = (int(seen[0]), int(seen[-1]), self.cutoff_)
            if current is not None and current.signature == signature:
                continue
            self.models_[algo] = self._fit_model(
                j, X[observed], Y[observed, j], signature
            )

    def _fit_model(
        self,
        index: int,
        X: np.ndarray,
        y: np.ndarray,
        signature: tuple[int, int, float],
    ) -> _AlgorithmModel:
        """Fit the model of the ``index``-th algorithm."""
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X = imputer.fit_transform(X)
        seed = None
        if self.random_state is not None:
            entropy = [self.random_state, index]
            seed = int(np.random.SeedSequence(entropy).generate_state(1)[0])
        estimator = clone_with_random_state(self.estimator_, seed)
        estimator.fit(X, y)
        oob = None
        if self.uncertainty == "jackknife":
            oob = np.ones((len(X), len(estimator.estimators_)), dtype=bool)
            for b, sample in enumerate(estimator.estimators_samples_):
                oob[sample, b] = False
        return _AlgorithmModel(imputer, estimator, oob, signature, len(X))

    # ---------------------------------------------------------- selection
    def _predict_model(
        self, model: _AlgorithmModel, x: np.ndarray
    ) -> tuple[float, float | None]:
        """Prediction and standard error (target scale) for one instance."""
        X = model.imputer.transform(x)
        if self.uncertainty is None:
            return float(model.estimator.predict(X)[0]), None
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        forest = model.estimator.estimators_
        trees = np.array([[t.predict(X32, check_input=False)[0] for t in forest]])
        if self.uncertainty == "jackknife":
            assert model.oob is not None
            sd = jackknife_after_bootstrap_sd(
                trees, model.oob, self.jackknife_bias_correction
            )[0]
        else:
            sd = trees[0].std(ddof=1) if trees.shape[1] > 1 else 0.0
        return float(trees.mean()), float(sd)

    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        assert self.buffer is not None
        x = features.reindex(columns=self.buffer.feature_names).to_numpy(dtype=float)
        sign = 1.0 if self.maximize else -1.0
        bonus = self.gamma if self.strategy == "ucb" else 0.0
        means: dict[str, float] = {}
        sds: dict[str, float | None] = {}
        scores: dict[str, float] = {}
        for algo in self.algorithms:
            if algo in self.models_:
                means[algo], sds[algo] = self._predict_model(self.models_[algo], x)
                scores[algo] = sign * means[algo] + bonus * (sds[algo] or 0.0)
        unmodelled = [a for a in self.algorithms if a not in scores]

        if unmodelled:
            chosen = unmodelled[int(self.rng_.integers(len(unmodelled)))]
        elif self.strategy == "epsilon_greedy" and self.rng_.random() < self.epsilon:
            chosen = self.algorithms[int(self.rng_.integers(len(self.algorithms)))]
        else:
            values = np.array([scores[a] for a in self.algorithms])
            best = np.flatnonzero(values == values.max())
            chosen = self.algorithms[int(self.rng_.choice(best))]

        order = {a: i for i, a in enumerate(self.algorithms)}
        ranked = sorted(scores, key=lambda a: (-scores[a], order[a]))
        candidates = [a for a in unmodelled + ranked if a != chosen]
        mean = means.get(chosen)
        return PredictionRecord(
            chosen=chosen,
            budget=0.0,
            predicted_runtime=None if mean is None else self._to_raw(mean),
            score=scores.get(chosen),
            uncertainty=sds.get(chosen),
            scores=scores or None,
            candidates=tuple(candidates),
        )

    def _to_raw(self, value: float) -> float:
        """Target scale -> raw performance scale."""
        return 10.0**value if self.target_transform == "log10" else value
