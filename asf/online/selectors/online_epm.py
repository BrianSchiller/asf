"""
Online selector with one runtime model per algorithm (Degroote et al., 2018).

One regression model (EPM) per algorithm, trained only on the instances it
ran on, so partially observed data needs no imputation. Selection is greedy,
epsilon-greedy or UCB on the predictions.

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

from asf.online.base import AbstractOnlineSelector, Observation, random_argmax
from asf.online.gate import AbstractExtraRunGate, NoExtrasGate
from asf.online.imputation import CensoredImputer, ExpectedImputer
from asf.online.prediction import PredictionRecord
from asf.preprocessing.schmee_hahn_imputer import schmee_hahn_impute
from asf.utils.random_state import clone_with_random_state

Strategy = Literal["greedy", "epsilon_greedy", "ucb"]
Uncertainty = Literal["jackknife", "tree_std"]
TargetTransform = Literal["none", "log10"]

#: Runtimes are clipped to at least this value before taking logs.
LOG_FLOOR = 1e-6


def default_estimator() -> RandomForestRegressor:
    """
    The random forest of Degroote et al. (R ``randomForest`` defaults): 500
    trees, ``mtry = p/3``, bootstrap, ``nodesize = 5`` (= sklearn's
    ``min_samples_split=6``).
    """
    return RandomForestRegressor(
        n_estimators=500, max_features=1 / 3, min_samples_split=6, bootstrap=True
    )


def jackknife_after_bootstrap_sd(
    tree_preds: np.ndarray, oob: np.ndarray, bias_correction: bool = True
) -> np.ndarray:
    """
    Jackknife-after-bootstrap standard error of a bagged prediction, as mlr's
    ``regr.randomForest`` with ``se.method="jackknife"`` (Sexton and Laake,
    2009; bias correction of Wager, Hastie and Efron, 2014)::

        V = (n-1)/n * sum_i (mean_{b: i out-of-bag} t_b(x) - t(x))^2
            - (e-1) * n / B^2 * sum_b (t_b(x) - t(x))^2

    ``tree_preds`` has shape ``(n_queries, B)``, the out-of-bag mask ``oob``
    ``(n, B)``. Instances in-bag for every tree are skipped (as in mlr);
    negative variances become 0.
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

    After every instance the executed runs join the training data
    (Algorithm 1) and the models are updated every ``update_every`` new data
    points (paper: 10). Strategies: ``"greedy"`` (Algorithm 2, best in the
    paper), ``"epsilon_greedy"`` (a uniformly random algorithm with
    probability ``epsilon``) and ``"ucb"`` (best ``prediction + gamma * sd``
    with performance as "higher is better").

    One run per instance, as in the paper; its ``allData`` upper bound is
    ``gate=AllNomineesGate()`` with ``result_mode="primary"``. The record's
    ``uncertainty`` is the standard error on the target scale; ``candidates``
    are unmodelled algorithms first, then by descending score.

    Assumptions and deviations
    --------------------------
    * Uncertainty: the jackknife via mlr, as in the paper
      (:func:`jackknife_after_bootstrap_sd`); 0 with one training instance.
    * Timeouts: censored or runtime ``>= cutoff`` (as the evaluator); target
      ``par_factor * cutoff`` unless ``imputation`` is set. The cutoff is the
      last observation's, before the first one ``budget``.
    * Censored imputation (Tornede et al., 2022; not in the paper):
      ``CutoffImputer`` trains on the budget, ``DropCensored`` drops the run,
      ``ExpectedImputer`` runs Schmee and Hahn (1979) via
      :func:`~asf.preprocessing.schmee_hahn_imputer.schmee_hahn_impute`
      (up to ``imputation_max_iter + 1`` fits per algorithm, stopping at
      ``imputation_tol``; best with ``target_transform="log10"``). Sampling
      rules have no batch version and are rejected.
    * Missing features: each model imputes the mean of its own training data
      (the paper uses the whole scenario, i.e. future instances).
    * Update schedule: extra runs count as data points; ``update_every=1`` is
      Algorithm 1 literally. Updates are not logged as retrainings.
    * Cold start: algorithms without data are tried first, at random among
      them; their record predictions are ``None``.
    * Ties are broken at random. With a ``window_size`` an algorithm whose
      data left the window becomes unmodelled again (paper: all data).

    Parameters
    ----------
    estimator : sklearn regressor or None, default=None
        Template cloned per algorithm (``None`` = :func:`default_estimator`).
        Needs to be a forest for ``uncertainty``, with ``bootstrap=True`` for
        the jackknife. Set ``n_jobs`` on it to fit in parallel.
    uncertainty : {"jackknife", "tree_std"} or None, default="jackknife"
        Jackknife (paper) or the spread across trees; ``None`` is faster but
        not allowed with ``"ucb"``.
    target_transform : {"none", "log10"}, default="none"
        Train on raw performance (paper) or its log10 (runtimes only).
    imputation : CensoredImputer or None, default=None
        See above; ``None`` = the PAR-k target (paper). Runtimes only.
    **kwargs : Any
        Forwarded to :class:`~asf.online.base.AbstractOnlineSelector`
        (``budget`` is the warm start's cutoff).
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
        update_every: int = 10,
        imputation: CensoredImputer | None = None,
        imputation_max_iter: int = 20,
        imputation_tol: float = 1e-3,
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
        if update_every < 1:
            raise ValueError("update_every must be >= 1.")
        template = estimator if estimator is not None else default_estimator()
        if uncertainty is not None and not isinstance(template, BaseForest):
            raise ValueError("uncertainty needs a forest estimator (BaseForest).")
        if uncertainty == "jackknife" and not template.bootstrap:
            raise ValueError("uncertainty='jackknife' needs bootstrap=True.")
        if imputation is not None:
            if not isinstance(imputation, CensoredImputer):
                raise TypeError("imputation must be a CensoredImputer or None.")
            if self.maximize:
                raise ValueError("Censored imputation is for runtime minimization.")
            if imputation.USES_MODEL and not isinstance(imputation, ExpectedImputer):
                raise ValueError(
                    f"{imputation!r} has no batch version; use CutoffImputer, "
                    "DropCensored or ExpectedImputer."
                )
        if imputation_max_iter < 1:
            raise ValueError("imputation_max_iter must be >= 1.")
        if not imputation_tol > 0:
            raise ValueError("imputation_tol must be > 0.")

        self.strategy = strategy
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.estimator = estimator
        self.uncertainty = uncertainty
        self.jackknife_bias_correction = bool(jackknife_bias_correction)
        self.target_transform = target_transform
        self.par_factor = float(par_factor)
        self.update_every = int(update_every)
        self.imputation = imputation
        self.imputation_max_iter = int(imputation_max_iter)
        self.imputation_tol = float(imputation_tol)
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
        if self.pending_ >= self.update_every:
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
        return NoExtrasGate()

    def _training_targets(
        self, performance: pd.DataFrame, censored: pd.DataFrame
    ) -> np.ndarray:
        """
        Training targets (``NaN`` = not run): unsolved runs get
        ``par_factor * cutoff``, or with ``imputation`` the rule's value at
        their bound (``ExpectedImputer``: the bound, as the start of its
        iteration); then ``target_transform`` is applied.
        """
        values = performance.to_numpy(dtype=float, copy=True)
        unsolved = self._unsolved(values, censored)
        if self.imputation is None:
            values[unsolved] = self.par_factor * self.cutoff_
        elif unsolved.any():
            bound = np.minimum(values[unsolved], self.cutoff_)
            if self.imputation.USES_MODEL:
                values[unsolved] = bound  # start of the iteration
            else:
                zeros = np.zeros_like(bound)
                values[unsolved] = self.imputation(bound, zeros, zeros, self.rng_)
        if self.target_transform == "log10":
            with np.errstate(invalid="ignore"):
                values = np.log10(np.maximum(values, LOG_FLOOR))
        return values

    def _unsolved(self, values: np.ndarray, censored: pd.DataFrame) -> np.ndarray:
        """Observed runs that were censored or reached the cutoff."""
        if not math.isfinite(self.cutoff_):
            return np.zeros(values.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            unsolved = censored.to_numpy(dtype=bool) | (values >= self.cutoff_)
        return unsolved & ~np.isnan(values)

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
            y = Y[observed, j]
            if isinstance(self.imputation, ExpectedImputer):
                y = self._schmee_hahn(j, X[observed], y)
            self.models_[algo] = self._fit_model(j, X[observed], y, signature)

    def _schmee_hahn(self, index: int, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Schmee-Hahn imputation of one algorithm's unsolved targets, which
        sit at the cutoff on the target scale.
        """
        if not math.isfinite(self.cutoff_):
            return y
        bound = self.cutoff_
        if self.target_transform == "log10":
            bound = math.log10(max(bound, LOG_FLOOR))
        if not np.any(y >= bound):
            return y
        X = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X)
        seed = self._seed(index)
        imputed, _ = schmee_hahn_impute(
            lambda: clone_with_random_state(self.estimator_, seed),
            y,
            X,
            em_max_iter=self.imputation_max_iter,
            em_tol=self.imputation_tol,
            budget=bound,
        )
        return imputed

    def _seed(self, index: int) -> int | None:
        """Seed of the ``index``-th algorithm's model (``None`` = random)."""
        if self.random_state is None:
            return None
        entropy = [self.random_state, index]
        return int(np.random.SeedSequence(entropy).generate_state(1)[0])

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
        estimator = clone_with_random_state(self.estimator_, self._seed(index))
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
            chosen = self.algorithms[random_argmax(values, self.rng_)]

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
