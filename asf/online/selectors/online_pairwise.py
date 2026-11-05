"""
One-vs-one online selector, following the selector of Rook et al. (FOGA 2025).

Reference
---------
J. Rook et al., "Efficient online automated algorithm selection in the face of
data-drift in optimisation problem instances", FOGA 2025, Section 4.4.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from asf.online.base import AbstractOnlineSelector, Observation
from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal
from asf.online.retraining.policy import AbstractRetrainPolicy, DriftTriggeredRetrain
from asf.selectors.abstract_model_based_selector import AbstractModelBasedSelector


class _RandomBinaryPredictor:
    """Bernoulli(0.5) predictor for pairs without any pairwise comparison."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._rng.integers(0, 2, size=len(X))


class OnlinePairwiseSelector(AbstractOnlineSelector, AbstractModelBasedSelector):
    """
    Partial one-vs-one selector trained from parallel runs on a stream.

    This mirrors the selector of Rook et al.:

    * **Parallel runs.** Per instance the selected algorithm is complemented with
      ``n_additional`` further algorithms (default 1). All of them run and the
      best result is returned to the user (``RESULT_MODE``).
    * **Partial one-vs-one.** One binary classifier per algorithm pair ``(i, j)``,
      trained only on instances where BOTH algorithms were run; the label says
      whether ``i`` is at least as good as ``j`` (ties go to ``i``). Prediction
      is by majority vote. Pairs without any comparison use a uniformly random
      predictor.
    * **Choosing additional algorithms.** Either uniformly at random
      (``"uniform"``) or by weighted sampling with weights inversely
      proportional to the estimated accuracy of the classifier between the
      selected and the candidate algorithm (``"error_guided"``).
    * **Initialisation.** No prior knowledge is assumed: during a reference
      period algorithms are sampled uniformly at random without replacement;
      then the classifiers (and the detector, if any) are fitted. If warm-start
      data is passed to ``fit`` instead, classifiers are trained on it directly
      and the reference period is skipped.
    * **Retraining.** When the retrain policy fires (by default: whenever the
      detector signals drift), all classifiers are retrained on the FULL history
      and the detector is refitted on it as well. Hence ``window_size=None``.

    Parameters
    ----------
    model_class : type, default=DecisionTreeClassifier
        Binary classifier class (any sklearn-compatible class or ASF predictor).
    n_additional : int, default=1
        Number of additional algorithms run next to the selected one.
    additional_strategy : {"error_guided", "uniform"}, default="error_guided"
        How the additional algorithms are chosen.
    reference_period : int, default=100
        Number of streamed instances in the random reference period (cold
        start only). TODO: use the value from Section 4.3 of the paper.
    detector : AbstractDriftDetector or None, default=None
        Drift detector, fitted at the end of the reference period and refitted
        after every retraining.
    retrain_policy : AbstractRetrainPolicy or None, default=None
        When to retrain. Defaults to ``DriftTriggeredRetrain`` if a detector is
        given and to "never retrain" otherwise.
    error_estimation_folds : int, default=5
        Folds used to estimate the accuracy of each pair classifier whenever the
        classifiers are (re)trained ("Replace Errors" in the paper's flowchart).
        Values below 2 disable the estimate: all pairs are then weighted equally.
        The excerpt does not say how accuracy is estimated; cross-validation on
        the pair data is our assumption.
    accuracy_floor : float, default=1e-3
        Lower bound on accuracies to keep the inverse weights finite.
    window_size : int or None, default=None
        History kept for retraining; ``None`` = all seen data (as in the paper).
    **kwargs : Any
        Forwarded to the base classes (``random_state``, ``maximize``, ...).
    """

    PREFIX = "online_pairwise"
    RESULT_MODE = "best_of_scheduled"

    def __init__(
        self,
        model_class: Any = DecisionTreeClassifier,
        n_additional: int = 1,
        additional_strategy: Literal["error_guided", "uniform"] = "error_guided",
        reference_period: int = 100,
        detector: AbstractDriftDetector | None = None,
        retrain_policy: AbstractRetrainPolicy | None = None,
        error_estimation_folds: int = 5,
        accuracy_floor: float = 1e-3,
        window_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_class=model_class, window_size=window_size, **kwargs)
        if additional_strategy not in ("error_guided", "uniform"):
            raise ValueError(f"Unknown additional_strategy: {additional_strategy!r}")
        if n_additional < 1:
            raise ValueError("n_additional must be >= 1 (at least two parallel runs).")
        self.n_additional = int(n_additional)
        self.additional_strategy = additional_strategy
        self.reference_period = int(reference_period)
        self.detector = detector
        self.error_estimation_folds = int(error_estimation_folds)
        self.accuracy_floor = float(accuracy_floor)
        if retrain_policy is None and detector is not None:
            retrain_policy = DriftTriggeredRetrain(detector)
        self.retrain_policy = retrain_policy

        self._rng = np.random.default_rng(self.random_state)
        self._pairs: list[tuple[int, int]] = []
        self._models: dict[tuple[int, int], Any] = {}
        self.pair_accuracy_: np.ndarray = np.empty((0, 0))
        self._reference_remaining = 0

    # ------------------------------------------------------------------ fit
    def _fit_initial(
        self, features: pd.DataFrame, performance: pd.DataFrame, **kwargs: Any
    ) -> None:
        m = len(self.algorithms)
        if m < 2:
            raise ValueError("Pairwise selection needs at least two algorithms.")
        if self.n_additional > m - 1:
            raise ValueError(
                f"n_additional={self.n_additional} needs >= {self.n_additional + 1} "
                f"algorithms, got {m}."
            )
        self._rng = np.random.default_rng(self.random_state)
        self._pairs = list(itertools.combinations(range(m), 2))
        self._reset_models()
        self._reference_remaining = 0 if len(features) > 0 else self.reference_period
        if self._reference_remaining == 0:
            self._train()
            self._fit_detector()

    # -------------------------------------------------------------- online
    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        m = len(self.algorithms)
        if self._reference_remaining > 0:
            chosen = self._rng.choice(m, size=1 + self.n_additional, replace=False)
        else:
            votes = self._votes(features)[0]
            selected = int(self._rng.choice(np.flatnonzero(votes == votes.max())))
            candidates = np.array([j for j in range(m) if j != selected])
            extra = self._rng.choice(
                candidates,
                size=self.n_additional,
                replace=False,
                p=self._extra_probabilities(selected, candidates),
            )
            chosen = np.concatenate([[selected], extra])
        return [(self.algorithms[int(j)], 0.0) for j in chosen]

    def _observe(self, observation: Observation) -> None:
        if self._reference_remaining > 0:
            self._reference_remaining -= 1
            if self._reference_remaining == 0:  # end of the reference period
                self._train()
                self._fit_detector()
            return

        signal: DriftSignal | None = None
        if self.detector is not None:
            signal = self.detector.update(
                observation.features, self._performance_row(observation)
            )
            if signal.drift:
                self._record_drift()

        policy = self.retrain_policy
        if policy is not None and policy.should_retrain(self.step_ + 1, signal):
            with self._timed() as timer:
                self._train()
                self._fit_detector(refit=True)
            self._record_retrain(timer.seconds)
            policy.on_retrained(self.step_ + 1)

    def _predict(
        self,
        features: pd.DataFrame | None,
        performance: pd.DataFrame | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Batch prediction: the majority-vote winner only, no exploration."""
        if features is None:
            raise ValueError("OnlinePairwiseSelector requires features.")
        votes = self._votes(features)
        budget = float(self.budget or 0)
        return {
            str(idx): [(self.algorithms[int(np.argmax(v))], budget)]
            for idx, v in zip(features.index, votes)
        }

    # ------------------------------------------------------------- internals
    def _reset_models(self) -> None:
        m = len(self.algorithms)
        self._models = {p: _RandomBinaryPredictor(self._rng) for p in self._pairs}
        self.pair_accuracy_ = np.full((m, m), 0.5)

    def _labels(self, perf_i: np.ndarray, perf_j: np.ndarray) -> np.ndarray:
        """1 if ``i`` is at least as good as ``j`` (ties go to ``i``)."""
        better = perf_i >= perf_j if self.maximize else perf_i <= perf_j
        return better.astype(int)

    def _train(self) -> None:
        """(Re)train every pair classifier on all pairwise comparisons in the buffer."""
        assert self.buffer is not None
        X, P = self.buffer.features(), self.buffer.performance()
        self._reset_models()
        for i, j in self._pairs:
            a, b = self.algorithms[i], self.algorithms[j]
            mask = (P[a].notna() & P[b].notna()).to_numpy()
            if not mask.any():
                continue  # no comparison yet -> keep the random predictor
            Xp = X.loc[mask]
            y = self._labels(P[a].to_numpy()[mask], P[b].to_numpy()[mask])
            model = self._make_model()
            model.fit(Xp, y)
            self._models[(i, j)] = model
            self.pair_accuracy_[i, j] = self.pair_accuracy_[j, i] = self._cv_accuracy(
                Xp, y
            )

    def _cv_accuracy(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Out-of-fold accuracy of one pair classifier (0.5 if not estimable)."""
        folds = min(self.error_estimation_folds, len(y))
        if folds < 2:
            return 0.5
        splitter = KFold(folds, shuffle=True, random_state=self.random_state)
        correct = 0
        for train, test in splitter.split(X):
            if np.unique(y[train]).size == 1:
                pred = np.full(len(test), y[train][0])
            else:
                model = self._make_model()
                model.fit(X.iloc[train], y[train])
                pred = np.asarray(model.predict(X.iloc[test]))
            correct += int((pred == y[test]).sum())
        return correct / len(y)

    def _fit_detector(self, refit: bool = False) -> None:
        """Fit (or, after a retraining, reset) the detector on the full history."""
        if self.detector is None or self.buffer is None or len(self.buffer) == 0:
            return
        X, P = self.buffer.features(), self.buffer.performance()
        if refit:
            self.detector.reset(X, P)
        else:
            self.detector.fit(X, P)

    def _votes(self, features: pd.DataFrame) -> np.ndarray:
        """Vote counts, shape ``(n_instances, n_algorithms)``."""
        votes = np.zeros((len(features), len(self.algorithms)))
        for (i, j), model in self._models.items():
            pred = np.asarray(model.predict(features))
            votes[pred == 1, i] += 1
            votes[pred != 1, j] += 1
        return votes

    def _extra_probabilities(
        self, selected: int, candidates: np.ndarray
    ) -> np.ndarray | None:
        """Sampling probabilities for additional algorithms (None = uniform)."""
        if self.additional_strategy == "uniform":
            return None
        acc = np.maximum(self.pair_accuracy_[selected, candidates], self.accuracy_floor)
        weights = 1.0 / acc
        return weights / weights.sum()
