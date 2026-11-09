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
from sklearn.metrics import f1_score
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
      trained only on instances where BOTH algorithms were run. Prediction is by
      majority vote with ties broken uniformly at random. Label ties differ per
      variant: ``"paper"`` labels ``i >= j`` as ``i`` winning (Section 4.4),
      ``"code"`` uses a strict ``i > j``, so ties go to ``j``. Pairs without
      comparisons use a Bernoulli(0.5) predictor (``"paper"``) or are skipped
      (``"code"``).
    * **Implementation variant.** ``variant="paper"`` follows the paper's
      uniform or inverse-accuracy sampling. ``variant="code"`` follows the
      authors' source code: least-observed sampling or weighted sampling by
      ``1 +`` in-sample F1 error.
    * **Initialisation.** No prior knowledge is assumed. The paper variant
      samples a random subset during its reference period and trains after
      exactly ``reference_period`` instances. The code variant chooses a random
      primary and least-observed additional algorithms and, as in the original
      ``StreamAS``, trains only after ``reference_period + 1`` instances (the
      original checks ``len(stream) >= train_period`` before appending the
      current instance). The drift detector then also processes that instance.
      If warm-start data is passed to ``fit`` (an ASF extension that is in
      neither source), training starts from that data and the reference period
      is skipped.
    * **Retraining.** When the retrain policy fires (by default: whenever the
      detector signals drift), all classifiers are retrained on the FULL history
      and the detector is refitted on it as well. Hence ``window_size=None``.

    Parameters
    ----------
    model_class : type, default=DecisionTreeClassifier
        Binary classifier class (any sklearn-compatible class or ASF predictor).
    n_additional : int, default=1
        Number of additional algorithms run next to the selected one.
    variant : {"paper", "code"}, default="paper"
        Which sampling and pairwise-error implementation to use.
    additional_strategy : str or None, default=None
        Defaults to ``"error_guided"`` for the paper variant and
        ``"least_observed"`` for the code variant. Paper options are
        ``"error_guided"`` and ``"uniform"``; code options are
        ``"least_observed"`` and ``"cost_based"``.
    reference_period : int or None, default=None
        Cold-start period. Defaults to 500 for both variants (paper,
        Section 4.3: "always 500"; code: ``train_period=500``). For
        ``"code"`` one extra instance is consumed, see above. The code variant
        assumes a maximizing quality measure, as does the original
        implementation.
    detector : AbstractDriftDetector or None, default=None
        Drift detector, fitted at the end of the reference period and refitted
        after every retraining. The original detector only uses features.
    retrain_policy : AbstractRetrainPolicy or None, default=None
        When to retrain. Defaults to ``DriftTriggeredRetrain`` if a detector is
        given and to "never retrain" otherwise. The policy receives the number
        of instances seen INCLUDING the current one.
    error_estimation_folds : int, default=5
        Folds used to estimate pairwise accuracy for paper-variant weights.
        Cross-validation is ASF's estimate because the paper does not specify
        this calculation. The code variant instead uses training-set
        F1 error.
    accuracy_floor : float, default=1e-3
        Lower bound on estimated accuracy to keep inverse weights finite.
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
        variant: Literal["paper", "code"] = "paper",
        additional_strategy: Literal[
            "error_guided", "uniform", "least_observed", "cost_based"
        ]
        | None = None,
        reference_period: int | None = None,
        detector: AbstractDriftDetector | None = None,
        retrain_policy: AbstractRetrainPolicy | None = None,
        error_estimation_folds: int = 5,
        accuracy_floor: float = 1e-3,
        window_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_class=model_class, window_size=window_size, **kwargs)
        if variant not in ("paper", "code"):
            raise ValueError(f"Unknown variant: {variant!r}")
        if additional_strategy is None:
            additional_strategy = (
                "error_guided" if variant == "paper" else "least_observed"
            )
        allowed_strategies = (
            ("error_guided", "uniform")
            if variant == "paper"
            else ("least_observed", "cost_based")
        )
        if additional_strategy not in allowed_strategies:
            raise ValueError(f"Unknown additional_strategy: {additional_strategy!r}")
        if n_additional < 1:
            raise ValueError("n_additional must be >= 1 (at least two parallel runs).")
        self.n_additional = int(n_additional)
        self.variant = variant
        self.additional_strategy = additional_strategy
        self.reference_period = int(
            500 if reference_period is None else reference_period
        )
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
        self.pair_f1_error_: np.ndarray = np.empty((0, 0))
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
        if self.variant == "code" and not self.maximize:
            raise ValueError(
                "variant='code' follows an implementation that assumes "
                "higher quality is better; pass maximize=True."
            )
        self._rng = np.random.default_rng(self.random_state)
        self._pairs = list(itertools.combinations(range(m), 2))
        self._reset_models()
        if len(features) > 0:
            self._reference_remaining = 0  # warm start (ASF extension)
        elif self.variant == "code":
            # Original StreamAS trains while processing instance
            # train_period + 1 (stream length is checked before appending).
            self._reference_remaining = self.reference_period + 1
        else:
            self._reference_remaining = self.reference_period
        if self._reference_remaining == 0:
            self._train()
            self._fit_detector()

    # -------------------------------------------------------------- online
    def _propose(self, features: pd.DataFrame) -> list[tuple[str, float]]:
        m = len(self.algorithms)
        if self._reference_remaining > 0:
            if self.variant == "paper":
                chosen = self._rng.choice(m, size=1 + self.n_additional, replace=False)
            else:
                primary = int(self._rng.integers(m))
                candidates = np.array([j for j in range(m) if j != primary])
                extras = self._least_observed(candidates, self.n_additional)
                chosen = np.concatenate([[primary], extras])
        else:
            votes = self._votes(features)[0]
            selected = int(self._rng.choice(np.flatnonzero(votes == votes.max())))
            candidates = np.array([j for j in range(m) if j != selected])
            if self.variant == "paper":
                extra = self._rng.choice(
                    candidates,
                    size=self.n_additional,
                    replace=False,
                    p=self._extra_probabilities(selected, candidates),
                )
            else:
                extra = self.alternative_extras(selected, candidates)
            chosen = np.concatenate([[selected], extra])
        return [(self.algorithms[int(j)], 0.0) for j in chosen]

    def _observe(self, observation: Observation) -> None:
        if self._reference_remaining > 0:
            self._reference_remaining -= 1
            if self._reference_remaining > 0:
                return
            # End of the reference period: train and fit the detector.
            self._train()
            self._fit_detector()
            if self.variant == "paper":
                return
            # Code variant: the original continues with ``detect()`` (and the
            # retrain check) on this very instance, so fall through.

        signal: DriftSignal | None = None
        if self.detector is not None:
            signal = self.detector.update(
                observation.features, self._performance_row(observation)
            )
            if signal.drift:
                self._record_drift()

        policy = self.retrain_policy
        # Number of instances seen including the current one (buffer is already
        # updated; step_ is the 0-based index of the current instance).
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
        predictions = {}
        for idx, row_votes in zip(features.index, votes):
            # Ties are broken at random in both variants, consistent with
            # ``_propose`` and the original ``PartialOVOclassifier.predict``.
            winners = np.flatnonzero(row_votes == row_votes.max())
            winner = int(self._rng.choice(winners))
            predictions[str(idx)] = [(self.algorithms[winner], budget)]
        return predictions

    # ------------------------------------------------------------- internals
    def _reset_models(self) -> None:
        m = len(self.algorithms)
        self._models = (
            {p: _RandomBinaryPredictor(self._rng) for p in self._pairs}
            if self.variant == "paper"
            else {}
        )
        self.pair_accuracy_ = np.full((m, m), 0.5)
        self.pair_f1_error_ = np.full((m, m), np.nan)

    def _labels(self, perf_i: np.ndarray, perf_j: np.ndarray) -> np.ndarray:
        """Return the pairwise winner labels for the selected variant."""
        if self.variant == "code":
            # The original assumes a maximizing quality measure and uses a
            # strict comparison, so ties are assigned to the second algorithm.
            better = perf_i > perf_j
        else:
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
                continue  # paper variant retains random predictor; code skips
            Xp = X.loc[mask]
            y = self._labels(P[a].to_numpy()[mask], P[b].to_numpy()[mask])
            model = self._make_model()
            model.fit(Xp, y)
            self._models[(i, j)] = model
            if self.variant == "paper":
                self.pair_accuracy_[i, j] = self.pair_accuracy_[j, i] = (
                    self._cv_accuracy(Xp, y)
                )
            else:
                error = 1.0 - f1_score(y, np.asarray(model.predict(Xp)))
                self.pair_f1_error_[i, j] = self.pair_f1_error_[j, i] = error

    def _cv_accuracy(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Out-of-fold accuracy estimate for one pair classifier."""
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
                pred = np.asarray(model.predict(X.iloc[test])).reshape(-1)
            correct += int((pred == y[test]).sum())
        return correct / len(y)

    def _fit_detector(self, refit: bool = False) -> None:
        """Fit (or, after a retraining, reset) the detector on the full history.

        The original ``DriftManager`` only uses instance features. ``reset``
        must behave like ``fit`` and leave an empty chunk buffer.
        """
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
            pred = np.asarray(model.predict(features)).reshape(-1)
            votes[pred == 1, i] += 1
            votes[pred != 1, j] += 1
        return votes

    def _extra_probabilities(
        self, selected: int, candidates: np.ndarray
    ) -> np.ndarray | None:
        """Sampling probabilities (None means uniform random sampling)."""
        if self.additional_strategy == "uniform":
            return None
        acc = np.maximum(self.pair_accuracy_[selected, candidates], self.accuracy_floor)
        weights = 1.0 / acc
        return weights / weights.sum()

    def _least_observed(self, candidates: np.ndarray, n: int) -> np.ndarray:
        """Code policy: prefer algorithms run on fewer instances."""
        assert self.buffer is not None
        perf = self.buffer.performance()
        counts = np.array([perf[self.algorithms[j]].notna().sum() for j in candidates])
        return candidates[np.argsort(counts, kind="stable")[:n]]

    def alternative_extras(self, selected: int, candidates: np.ndarray) -> np.ndarray:
        """Apply the original's least-observed or ``1 + F1 error`` sampling rule."""
        if self.additional_strategy == "least_observed":
            return self._least_observed(candidates, self.n_additional)
        errors = self.pair_f1_error_[selected, candidates]
        available = np.isfinite(errors)
        if not available.any():
            # The original has no fallback for an unseen pair; use
            # least-observed exploration so ASF can continue cold-starting.
            return self._least_observed(candidates, self.n_additional)
        available_candidates = candidates[available]
        weights = 1.0 + errors[available]
        weights /= weights.sum()
        # The original draws each extra independently, so repeats are possible.
        return self._rng.choice(
            available_candidates,
            size=self.n_additional,
            replace=True,
            p=weights,
        )
