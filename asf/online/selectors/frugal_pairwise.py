"""
Frugal pairwise selector: Kuş et al.'s frugal algorithm selection on a stream.

References
----------
E. Kuş et al., "Frugal Algorithm Selection", arXiv:2405.11059 (2024).
E. Kuş et al., "Frugal Algorithm Selection for Combinatorial Search", JAIR 86
(2026), Sections 4.1-4.2 and Appendix A.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import ForestClassifier

from asf.online.base import (
    AbstractOnlineSelector,
    DriftResponses,
    Observation,
    random_argmax,
)
from asf.online.drift.abstract_detector import (
    AbstractDriftDetector,
    AbstractScalarDriftDetector,
)
from asf.online.drift.statistics import DriftStatistic
from asf.online.gate import AbstractExtraRunGate, UncertaintyGate
from asf.online.prediction import PredictionRecord
from asf.predictors import SklearnWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.random_state import random_state_kwargs

#: A trained model or, if all its labels agreed, a smoothed P(label 1) as a float.
_Model = Any


def least_confidence(p: np.ndarray | float) -> np.ndarray:
    """
    Least confidence ``1 - max(p, 1 - p)`` of binary predictions, in
    ``[0, 0.5]``. For two classes margin and entropy rank the same (Kuş et
    al., JAIR, Appendix A).
    """
    p = np.asarray(p, dtype=float)
    return 1.0 - np.maximum(p, 1.0 - p)


class FrugalPairwiseSelector(AbstractOnlineSelector, AbstractModelBasedSelector):
    """
    One-vs-one selector that runs a second algorithm only when it is unsure.

    Frugal algorithm selection (Kuş et al.) moved from pool-based to
    stream-based active learning:

    * **Model.** One classifier per pair, trained where both ran; most votes
      wins, ties at random; untrained pairs do not vote. A pair whose labels
      all agree predicts ``(k + 1) / (n + 2)`` (Laplace).
    * **Labels.** A finished run beats a timeout; two timeouts are not used
      (Algorithm 1, line 7); equal performance counts for ``i``.
    * **Uncertainty.** Nominees are the other algorithms, most uncertain pair
      (:func:`least_confidence`) first; ``uncertainty`` is the first
      nominee's. The default gate runs it if that reaches 0.25.
      ``candidate_order="runner_up"`` nominates only the vote runner-up, so
      extra runs go to a rival that might be better; ``"random"`` is the
      paper's random baseline.
    * **Timeout predictors.** ``timeout_filter`` drops a nominee if it and
      the chosen one are both predicted to time out (JAIR, Algorithm 2);
      ``timeout_voting`` excludes predicted timeouts from voting unless all
      are (arXiv, Section 3.2). Runtimes ``>= cutoff`` count as timeouts
      (``None``: ``budget``, else the first observation's cutoff).
    * **Updates.** After every ``update_every`` instances only the pairs and
      timeout predictors of algorithms that ran are refitted.

    The paper's initial set (all algorithms on 20 instances) is a warm start
    of 20. Dynamic time limits and the cost-aware criteria are not
    implemented. The default random forest matches the paper's;
    ``model_class`` needs ``predict_proba``. ``window_size=None`` keeps all
    data, like the paper's growing labelled set. Drift responses:
    ``RetrainResponse`` and ``ResetResponse``.
    """

    PREFIX = "frugal_pairwise"
    RESULT_MODE = "primary"

    def __init__(
        self,
        model_class: Any = RandomForestClassifier,
        candidate_order: Literal["uncertainty", "random", "runner_up"] = "uncertainty",
        timeout_filter: bool = False,
        timeout_voting: bool = False,
        timeout_model_class: Any = RandomForestClassifier,
        cutoff: float | None = None,
        update_every: int = 1,
        window_size: int | None = None,
        gate: AbstractExtraRunGate | None = None,
        detector: AbstractDriftDetector | AbstractScalarDriftDetector | None = None,
        drift_statistic: DriftStatistic | None = None,
        drift_response: DriftResponses = None,
        retrain_every: int | None = None,
        detector_reference_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_class=model_class,
            window_size=window_size,
            gate=gate,
            detector=detector,
            drift_statistic=drift_statistic,
            drift_response=drift_response,
            retrain_every=retrain_every,
            detector_reference_size=detector_reference_size,
            **kwargs,
        )
        if candidate_order not in ("uncertainty", "random", "runner_up"):
            raise ValueError(f"Unknown candidate_order: {candidate_order!r}")
        if update_every < 1:
            raise ValueError("update_every must be >= 1.")
        if (timeout_filter or timeout_voting) and self.maximize:
            raise ValueError("Timeout predictors need a runtime (minimised) metric.")
        self.candidate_order = candidate_order
        self.timeout_filter = bool(timeout_filter)
        self.timeout_voting = bool(timeout_voting)
        self.timeout_model_class = timeout_model_class
        self.cutoff = None if cutoff is None else float(cutoff)
        self.update_every = int(update_every)

        self._rng = np.random.default_rng(self.random_state)
        self._models: dict[tuple[int, int], _Model] = {}
        self._timeout_models: dict[int, _Model] = {}
        self._dirty_pairs: set[tuple[int, int]] = set()
        self._dirty_algos: set[int] = set()
        self._since_update = 0
        self._n_usable = 0  # most recent buffer rows usable for training
        self._cutoff: float | None = None
        #: Training-set size per pair, ``(m, m)``, symmetric.
        self.pair_n_train_: np.ndarray = np.zeros((0, 0), dtype=int)

    @property
    def uses_timeouts(self) -> bool:
        """Whether timeout predictors are trained."""
        return self.timeout_filter or self.timeout_voting

    # ------------------------------------------------------------------ fit
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        m = len(self.algorithms)
        if m < 2:
            raise ValueError("Pairwise selection needs at least two algorithms.")
        if _proba_estimator(self._make_model()) is None:
            raise TypeError("model_class must provide predict_proba.")
        self._cutoff = self.cutoff if self.cutoff is not None else self.budget
        if self.uses_timeouts and len(features) > 0 and self._cutoff is None:
            raise ValueError(
                "Timeout predictors on a warm start need cutoff= (or budget=) "
                "to label timeouts in the warm-start data."
            )
        self._rng = np.random.default_rng(self.random_state)
        self._n_usable = len(features)
        self._since_update = 0
        self._fit_all()

    # -------------------------------------------------------------- online
    def _select(self, features: pd.DataFrame) -> PredictionRecord:
        x = features.to_numpy(dtype=float)
        probs = self._pair_probabilities(x)[0]
        timeouts = self._predicted_timeouts(x)[0]
        votes = self._votes(probs, timeouts)
        chosen = random_argmax(votes, self._rng)

        others = np.array([a for a in range(len(self.algorithms)) if a != chosen])
        if self.timeout_filter and timeouts[chosen]:
            others = others[~timeouts[others]]
        uncertainty = least_confidence(probs[chosen, others])
        if self.candidate_order == "runner_up" and len(others):
            k = random_argmax(votes[others], self._rng)
            others, uncertainty = others[[k]], uncertainty[[k]]
        if self.candidate_order == "random":
            order = self._rng.permutation(len(others))
        else:
            order = np.argsort(-uncertainty, kind="stable")
        return PredictionRecord(
            chosen=self.algorithms[chosen],
            budget=0.0,
            score=float(votes[chosen]),
            uncertainty=float(uncertainty.max()) if len(others) else 0.0,
            scores={a: float(v) for a, v in zip(self.algorithms, votes)},
            candidates=tuple(self.algorithms[int(a)] for a in others[order]),
        )

    def _observe(self, observation: Observation) -> None:
        if self._cutoff is None and np.isfinite(observation.cutoff):
            self._cutoff = float(observation.cutoff)
        self._n_usable += 1
        ran = sorted(self.algorithms.index(str(a)) for a in observation.runtimes)
        self._dirty_algos.update(ran)
        self._dirty_pairs.update(
            (i, j) for k, i in enumerate(ran) for j in ran[k + 1 :]
        )
        self._since_update += 1
        if self._since_update >= self.update_every:
            self._refit(self._dirty_pairs, self._dirty_algos)

    def _retrain(self) -> None:
        """Refit every pair classifier and timeout predictor on the buffer."""
        self._fit_all()

    def _reset_model(self) -> None:
        """Forget all data seen so far (it stays in the buffer but is unused)."""
        self._n_usable = 0
        self._fit_all()

    def _default_gate(self) -> AbstractExtraRunGate:
        return UncertaintyGate(threshold=0.25, n=1)

    # ------------------------------------------------------------- prediction
    def _pair_probabilities(self, x: np.ndarray) -> np.ndarray:
        """``P[n, i, j]`` = probability that ``i`` beats ``j`` (0.5 if untrained)."""
        m = len(self.algorithms)
        probs = np.full((len(x), m, m), 0.5)
        for (i, j), model in self._models.items():
            p = _positive_proba(model, x)
            probs[:, i, j], probs[:, j, i] = p, 1.0 - p
        return probs

    def _predicted_timeouts(self, x: np.ndarray) -> np.ndarray:
        """Boolean ``(n, m)``: algorithm predicted to time out."""
        out = np.zeros((len(x), len(self.algorithms)), dtype=bool)
        for a, model in self._timeout_models.items():
            out[:, a] = _positive_proba(model, x) > 0.5
        return out

    def _votes(self, probs: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
        """
        Votes for one instance: ``i`` wins if ``P(i beats j) > 0.5``. With
        ``timeout_voting``, predicted timeouts neither vote nor get votes
        (unless all are predicted to time out).
        """
        m = len(self.algorithms)
        allowed = np.ones(m, dtype=bool)
        if self.timeout_voting and not timeouts.all():
            allowed = ~timeouts
        votes = np.zeros(m)
        for i, j in self._models:
            if allowed[i] and allowed[j]:
                votes[i if probs[i, j] > 0.5 else j] += 1
        votes[~allowed] = -np.inf
        return votes

    # ------------------------------------------------------------- training
    def _fit_all(self) -> None:
        m = len(self.algorithms)
        self._models, self._timeout_models = {}, {}
        self.pair_n_train_ = np.zeros((m, m), dtype=int)
        pairs = {(i, j) for i in range(m) for j in range(i + 1, m)}
        self._refit(pairs, set(range(m)))

    def _refit(self, pairs: set[tuple[int, int]], algos: set[int]) -> None:
        """Refit the given pairs and timeout predictors on the usable buffer rows."""
        assert self.buffer is not None
        n = min(self._n_usable, len(self.buffer))
        x = self.buffer.features().to_numpy(dtype=float)[len(self.buffer) - n :]
        perf = self.buffer.performance().to_numpy(dtype=float)[len(self.buffer) - n :]
        timeout = self.buffer.censored_mask().to_numpy(dtype=bool)[
            len(self.buffer) - n :
        ]
        if self._cutoff is not None:
            timeout = timeout | (perf >= self._cutoff)  # NaN compares False
        ran = ~np.isnan(perf)

        for i, j in pairs:
            both = ran[:, i] & ran[:, j] & ~(timeout[:, i] & timeout[:, j])
            self.pair_n_train_[i, j] = self.pair_n_train_[j, i] = int(both.sum())
            if not both.any():
                self._models.pop((i, j), None)
                continue
            y = self._labels(
                perf[both, i], perf[both, j], timeout[both, i], timeout[both, j]
            )
            self._models[(i, j)] = self._train(self._make_model, x[both], y)
        if self.uses_timeouts:
            for a in algos:
                if not ran[:, a].any():
                    self._timeout_models.pop(a, None)
                    continue
                y = timeout[ran[:, a], a].astype(int)
                self._timeout_models[a] = self._train(
                    self._make_timeout_model, x[ran[:, a]], y
                )
        self._dirty_pairs, self._dirty_algos = set(), set()
        self._since_update = 0

    def _labels(
        self,
        perf_i: np.ndarray,
        perf_j: np.ndarray,
        timeout_i: np.ndarray,
        timeout_j: np.ndarray,
    ) -> np.ndarray:
        """``1`` where ``i`` beats ``j``; a finished run beats a timeout."""
        better = perf_i >= perf_j if self.maximize else perf_i <= perf_j
        better = np.where(timeout_i & ~timeout_j, False, better)
        better = np.where(timeout_j & ~timeout_i, True, better)
        return better.astype(int)

    @staticmethod
    def _train(make: Any, x: np.ndarray, y: np.ndarray) -> _Model:
        """
        Fit a classifier, or if all labels agree return the Laplace estimate
        ``(k + 1) / (n + 2)``, so a few examples do not make a pair certain.
        """
        if np.all(y == y[0]):
            return (float(y.sum()) + 1.0) / (len(y) + 2.0)
        model = make()
        model.fit(x, y)
        return model

    def _make_timeout_model(self) -> Any:
        factory = self.timeout_model_class
        if isinstance(factory, type) and issubclass(factory, ClassifierMixin):
            factory = partial(SklearnWrapper, factory)
        return factory(**random_state_kwargs(factory, self.random_state))


def _proba_estimator(model: Any) -> Any:
    """The object providing ``predict_proba`` (the model or a wrapped one)."""
    for candidate in (model, getattr(model, "model_class", None)):
        if hasattr(candidate, "predict_proba"):
            return candidate
    return None


def _positive_proba(model: _Model, x: np.ndarray) -> np.ndarray:
    """Probability of label ``1`` per row."""
    if isinstance(model, float):
        return np.full(len(x), model)
    estimator = _proba_estimator(model)
    if _is_plain_forest(estimator) and not np.isnan(x).any():
        proba = _forest_proba(estimator, x)
    else:
        proba = np.asarray(estimator.predict_proba(x))
    classes = list(getattr(estimator, "classes_", [0, 1]))
    return proba[:, classes.index(1)]


def _is_plain_forest(estimator: Any) -> bool:
    """A fitted single-output sklearn forest classifier (random or extra trees)."""
    return (
        isinstance(estimator, ForestClassifier)
        and hasattr(estimator, "estimators_")
        and getattr(estimator, "n_outputs_", 1) == 1
    )


def _forest_proba(forest: ForestClassifier, x: np.ndarray) -> np.ndarray:
    """
    ``forest.predict_proba(x)`` without sklearn's per-call input checks and
    thread pool, which dominate for single instances. Same trees, same mean.
    """
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    total = forest.estimators_[0].predict_proba(x32, check_input=False)
    for tree in forest.estimators_[1:]:
        total += tree.predict_proba(x32, check_input=False)
    return total / len(forest.estimators_)
