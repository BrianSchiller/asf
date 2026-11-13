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

from asf.online.base import AbstractOnlineSelector, DriftResponses, Observation
from asf.online.drift.abstract_detector import (
    AbstractDriftDetector,
    AbstractScalarDriftDetector,
)
from asf.online.drift.statistics import AbstractDriftStatistic
from asf.online.gate import AbstractExtraRunGate, UncertaintyGate
from asf.online.prediction import PredictionRecord
from asf.online.retraining.policy import AbstractRetrainPolicy
from asf.predictors import SklearnWrapper
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector
from asf.utils.random_state import random_state_kwargs

#: A trained model or, if all its labels agreed, a smoothed P(label 1) as a float.
_Model = Any


def least_confidence(p: np.ndarray | float) -> np.ndarray:
    """
    Least-confidence uncertainty of binary predictions, ``1 - max(p, 1 - p)``.

    ``0`` for a certain prediction, ``0.5`` for a coin flip. For two classes,
    margin and entropy rank predictions in the same order (Kuş et al., JAIR,
    Appendix A).

    Parameters
    ----------
    p : np.ndarray or float
        Probability of one of the two classes.

    Returns
    -------
    np.ndarray
        Uncertainty per entry, in ``[0, 0.5]``.
    """
    p = np.asarray(p, dtype=float)
    return 1.0 - np.maximum(p, 1.0 - p)


class FrugalPairwiseSelector(AbstractOnlineSelector, AbstractModelBasedSelector):
    """
    One-vs-one selector that runs a second algorithm only when it is unsure.

    Adapts frugal algorithm selection (Kuş et al.) from pool-based to
    stream-based active learning:

    * **Model.** One binary classifier per algorithm pair ``(i, j)``
      predicting whether ``i`` beats ``j``, trained on the instances where both
      ran. The algorithm with the most votes is chosen; vote ties are broken
      uniformly at random. Pairs without data cast no vote. A pair whose
      labels all agree predicts ``(k + 1) / (n + 2)`` (Laplace) instead of
      being fully certain after a single example.
    * **Labels.** A finished run beats a timed-out run. Pairs where both timed
      out are not used for training (Kuş et al., Algorithm 1, line 7). Equal
      performance counts as a win for ``i``.
    * **Uncertainty.** For every other algorithm ``a``, the least confidence
      (:func:`least_confidence`) of the pair (chosen, ``a``). The nominees for
      an extra run are the other algorithms, most uncertain pair first;
      ``PredictionRecord.uncertainty`` is the uncertainty of the first
      nominee. The default gate, :class:`~asf.online.gate.UncertaintyGate`
      with ``threshold=0.25``, runs that nominee if the uncertainty reaches
      the threshold.
    * **Timeout predictors (optional).** One classifier per algorithm predicts
      whether it times out. ``timeout_filter`` drops a nominee if it and the
      chosen algorithm are both predicted to time out (a double timeout is
      costly and teaches nothing; JAIR, Algorithm 2). ``timeout_voting``
      removes algorithms predicted to time out before voting, unless all are
      (arXiv, Section 3.2).
    * **Updates.** After an instance, only the pairs (and timeout predictors)
      of the algorithms that ran are refitted, on the buffer.

    The paper's initial set (all algorithms on 20 instances) corresponds to
    a warm start (``n_warmstart=20``). On a cold start, all pairs are
    untrained and maximally uncertain, so the default gate always fires until
    data arrives. Dynamic time limits and the cost-aware PC/PD criteria of the
    paper are not implemented.

    Parameters
    ----------
    model_class : type or Callable, default=RandomForestClassifier
        Pair classifier; must provide ``predict_proba`` (directly or as the
        wrapped ``model_class`` of an ASF ``SklearnWrapper``). sklearn's
        defaults match the paper's random forest (100 trees, Gini, sqrt
        features, bootstrap).
    candidate_order : {"uncertainty", "random"}, default="uncertainty"
        Order of the nominees: most uncertain pair first, or uniformly random
        (the paper's random-selection baseline).
    timeout_filter : bool, default=False
        Use timeout predictors to drop likely double-timeout nominees.
    timeout_voting : bool, default=False
        Use timeout predictors to exclude likely timeouts from voting.
    timeout_model_class : type or Callable, default=RandomForestClassifier
        Classifier for the timeout predictors.
    cutoff : float or None, default=None
        Runtimes ``>= cutoff`` count as timeouts (censored runs always do).
        ``None`` uses ``budget`` and otherwise the cutoff of the first
        observation. Timeout predictors with a warm start need it.
    update_every : int, default=1
        Refit the changed models after every ``update_every`` instances.
    window_size : int or None, default=None
        Instances kept in the buffer; ``None`` keeps all (as the paper's
        labelled set, which only grows).
    gate : AbstractExtraRunGate or None, default=None
        Extra-run gate; ``None`` = ``UncertaintyGate(threshold=0.25, n=1)``.
    detector, drift_statistic, drift_response, retrain_policy, detector_reference_size
        See :class:`~asf.online.base.AbstractOnlineSelector`. Supported
        responses: ``RetrainResponse`` (refit all models) and ``ResetResponse``
        (forget all data seen so far).
    **kwargs : Any
        Forwarded to the base classes (``random_state``, ``maximize``, ...).
    """

    PREFIX = "frugal_pairwise"
    RESULT_MODE = "primary"

    def __init__(
        self,
        model_class: Any = RandomForestClassifier,
        candidate_order: Literal["uncertainty", "random"] = "uncertainty",
        timeout_filter: bool = False,
        timeout_voting: bool = False,
        timeout_model_class: Any = RandomForestClassifier,
        cutoff: float | None = None,
        update_every: int = 1,
        window_size: int | None = None,
        gate: AbstractExtraRunGate | None = None,
        detector: AbstractDriftDetector | AbstractScalarDriftDetector | None = None,
        drift_statistic: AbstractDriftStatistic | None = None,
        drift_response: DriftResponses = None,
        retrain_policy: AbstractRetrainPolicy | None = None,
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
            retrain_policy=retrain_policy,
            detector_reference_size=detector_reference_size,
            **kwargs,
        )
        if candidate_order not in ("uncertainty", "random"):
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
        chosen = int(self._rng.choice(np.flatnonzero(votes == votes.max())))

        others = np.array([a for a in range(len(self.algorithms)) if a != chosen])
        if self.timeout_filter and timeouts[chosen]:
            others = others[~timeouts[others]]
        uncertainty = least_confidence(probs[chosen, others])
        if self.candidate_order == "uncertainty":
            order = np.argsort(-uncertainty, kind="stable")
        else:
            order = self._rng.permutation(len(others))
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

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Batch prediction: the vote winner per row, no extra runs."""
        if features is None:
            raise ValueError("FrugalPairwiseSelector requires features.")
        x = features.to_numpy(dtype=float)
        probs, timeouts = self._pair_probabilities(x), self._predicted_timeouts(x)
        budget = float(self.budget or 0)
        out = {}
        for idx, p, t in zip(features.index, probs, timeouts):
            votes = self._votes(p, t)
            winner = int(self._rng.choice(np.flatnonzero(votes == votes.max())))
            out[str(idx)] = [(self.algorithms[winner], budget)]
        return out

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
        Votes of the trained pairs for one instance. ``i`` wins its pair if
        ``P(i beats j) > 0.5`` (like ``predict``, ties go to ``j``). With
        ``timeout_voting``, predicted timeouts get no votes and are not
        counted, unless all algorithms are predicted to time out.
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
        Fit a classifier. If all labels agree, return the Laplace estimate
        ``(k + 1) / (n + 2)`` of ``P(label 1)`` instead, so that one or two
        examples do not make the pair fully certain.
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
    proba = np.asarray(estimator.predict_proba(x))
    classes = list(getattr(estimator, "classes_", [0, 1]))
    return proba[:, classes.index(1)]
