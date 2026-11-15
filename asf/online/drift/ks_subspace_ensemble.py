"""
Feature drift detection with an ensemble of KS tests over feature subspaces.

This adapts EDFS (Korycki and Krawczyk, "Unsupervised Drift Detector Ensembles
for Data Stream Mining", DSAA 2019) to *covariate* drift, i.e. to noticing that
the instance features leave the distribution seen so far:

1. the features are grouped into ``n_subspaces`` subspaces (Algorithm 1 of the
   paper, greedy and round-robin),
2. every ``stride`` instances, each feature of a subspace is compared between
   the reference data and the most recent ``window_size`` instances with a
   two-sample Kolmogorov-Smirnov test,
3. a subspace has drifted if a strict majority of its tests reject, and the
   detector signals drift if at least one subspace has drifted.

Differences from the paper:

* The paper rates a feature by its mutual information with the class label,
  which needs labels and targets concept drift. Here the quality of a
  candidate is its mean absolute Spearman correlation with the features
  already in the subspace, so that a subspace holds features that tend to
  move together. No performance data is used anywhere.
* The diversity term of Eq. 12 is evaluated for the subspace *including* the
  candidate (as printed it does not depend on the candidate), and the feature
  with the highest score is taken, as in Algorithm 1.
* The KS statistic is computed exactly on the stored windows at every check
  instead of with the incremental KS data structure; the result is the same.
* Labels are never requested. After a handled drift the selector calls
  :meth:`~KSSubspaceEnsembleDetector.reset`, which takes a new reference and
  rebuilds the subspaces.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from asf.online.drift.abstract_detector import AbstractDriftDetector, DriftSignal


def _clean_sorted(values: np.ndarray) -> np.ndarray:
    """Flatten, drop NaNs and sort."""
    values = np.asarray(values, dtype=float).ravel()
    return np.sort(values[~np.isnan(values)])


def _ks_sorted(a: np.ndarray, b: np.ndarray) -> float:
    """KS statistic of two non-empty, sorted, NaN-free samples."""
    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / len(a)
    cdf_b = np.searchsorted(b, pooled, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic (Eq. 4 of the paper).

    ``D = max |F_a(x) - F_b(x)|`` over all observed values ``x``, with ``F``
    the empirical distribution function. NaNs are dropped; ties are handled
    exactly.

    Parameters
    ----------
    a, b : np.ndarray
        The two samples.

    Returns
    -------
    float
        The statistic, in [0, 1].
    """
    a, b = _clean_sorted(a), _clean_sorted(b)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both samples need at least one non-missing value.")
    return _ks_sorted(a, b)


def ks_critical_value(alpha: float, n: int, m: int) -> float:
    """
    Rejection threshold of the two-sample KS test (Eq. 1 of the paper).

    The null hypothesis is rejected at level ``alpha`` if
    ``D > c(alpha) * sqrt((n + m) / (n * m))`` with
    ``c(alpha) = sqrt(-ln(alpha / 2) / 2)``.

    Parameters
    ----------
    alpha : float
        Significance level, in (0, 1).
    n, m : int
        Sizes of the two samples.

    Returns
    -------
    float
        The critical value for ``D``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if n < 1 or m < 1:
        raise ValueError("Sample sizes must be >= 1.")
    return math.sqrt(-math.log(alpha / 2) / 2) * math.sqrt((n + m) / (n * m))


def build_subspaces(
    corr: np.ndarray, n_subspaces: int, size: int, rng: np.random.Generator
) -> list[list[int]]:
    """
    Group features into subspaces (Algorithm 1 of the paper, without labels).

    Subspaces are filled round-robin, one feature at a time. A subspace
    ``S_i`` receives the feature ``c`` (not yet in it) that maximises
    ``quality + diversity``:

    * quality: mean of ``|corr[c, s]|`` over the features ``s`` already in
      ``S_i`` (0 for an empty subspace),
    * diversity: the mean of ``1 - (share of subspaces containing c)``
      (Eq. 11) and ``1 - max_l |S_i' & S_l| / |S_i'|`` over the other
      subspaces ``S_l``, with ``S_i' = S_i + {c}`` (Eq. 12).

    Parameters
    ----------
    corr : np.ndarray
        Square matrix of feature associations; absolute values are used and
        NaNs count as 0.
    n_subspaces : int
        Number of subspaces ``k``.
    size : int
        Features per subspace ``n``, at most the number of features.
    rng : np.random.Generator
        Breaks ties between equally good features.

    Returns
    -------
    list[list[int]]
        Feature indices per subspace, in the order they were added.
    """
    corr = np.abs(np.nan_to_num(np.asarray(corr, dtype=float)))
    d = corr.shape[0]
    if corr.shape != (d, d):
        raise ValueError("corr must be a square matrix.")
    if n_subspaces < 1:
        raise ValueError("n_subspaces must be >= 1.")
    if not 1 <= size <= d:
        raise ValueError("size must be between 1 and the number of features.")

    member = np.zeros((n_subspaces, d), dtype=bool)
    subspaces: list[list[int]] = [[] for _ in range(n_subspaces)]
    for _ in range(size):
        for i, current in enumerate(subspaces):
            quality = corr[:, current].mean(axis=1) if current else np.zeros(d)
            spread = 1.0 - member.sum(axis=0) / n_subspaces
            others = np.delete(member, i, axis=0)
            if len(others):
                shared = (others & member[i]).sum(axis=1, keepdims=True)
                overlap = (shared + others).max(axis=0) / (len(current) + 1)
            else:
                overlap = np.zeros(d)
            score = quality + (spread + 1.0 - overlap) / 2
            score[current] = -np.inf
            best = np.flatnonzero(score >= score.max() - 1e-12)
            choice = int(best[0] if len(best) == 1 else rng.choice(best))
            current.append(choice)
            member[i, choice] = True
    return subspaces


class KSSubspaceEnsembleDetector(AbstractDriftDetector):
    """
    Covariate drift detector: KS tests on feature subspaces (EDFS-style).

    The detector consumes one instance at a time and keeps the most recent
    ``window_size`` instances. Every ``stride`` instances (default
    ``window_size``: non-overlapping windows) it tests each feature of each
    subspace against the reference data. A signal is therefore delayed by up
    to ``stride`` instances. Only features are used; ``performance`` is
    ignored.

    The reference is the data given to :meth:`fit` (the selector passes its
    buffer). The KS test compares distributions, so while the buffer still
    mixes data from before and after a drift, later checks can signal again.

    Features that are in no subspace are not monitored. Missing values are
    left out of the tests; a feature with fewer than two values in the
    reference or in the window does not vote for drift.

    Parameters
    ----------
    n_subspaces : int, default=10
        Number of subspaces ``k`` (paper: 10).
    subspace_size : float or int, default=0.1
        Features per subspace ``n``. A float in (0, 1] is a share of all
        features, rounded up (paper: 0.1); an int is a count (capped at the
        number of features).
    subspaces : {"guided", "random"} or sequence of sequences of str, \
default="guided"
        ``"guided"``: build subspaces with :func:`build_subspaces` from the
        Spearman correlations of the reference data. ``"random"``: draw each
        subspace uniformly without replacement (the paper's RFS baseline).
        A sequence of feature-name groups fixes the subspaces; then
        ``n_subspaces`` and ``subspace_size`` are ignored.
    window_size : int, default=50
        Number of recent instances compared with the reference.
    stride : int or None, default=None
        Test every ``stride`` instances (``None`` = ``window_size``).
    alpha : float, default=0.01
        Significance level of each univariate KS test.
    reference_size : int or None, default=None
        Use only the last ``reference_size`` rows of the fit data as
        reference. ``None`` uses all of them.
    random_state : int or None, default=None
        Seed for tie-breaking in guided subspaces and for random subspaces.

    Attributes
    ----------
    feature_names_ : list[str]
        Feature names seen in ``fit``.
    subspaces_ : list[list[str]]
        Feature names per subspace.
    critical_value_ : float
        KS rejection threshold for a full reference and a full window
        without missing values.
    last_votes_ : dict[str, bool] or None
        Per monitored feature, whether its test rejected at the last check.
    last_score_ : float or None
        Largest share of rejecting tests in any subspace at the last check
        (drift if above 0.5). This is the ``score`` of the returned signals.
    """

    def __init__(
        self,
        n_subspaces: int = 10,
        subspace_size: float | int = 0.1,
        subspaces: Literal["guided", "random"] | Sequence[Sequence[str]] = "guided",
        window_size: int = 50,
        stride: int | None = None,
        alpha: float = 0.01,
        reference_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if n_subspaces < 1:
            raise ValueError("n_subspaces must be >= 1.")
        if isinstance(subspace_size, bool) or not (
            (isinstance(subspace_size, int) and subspace_size >= 1)
            or (isinstance(subspace_size, float) and 0 < subspace_size <= 1)
        ):
            raise ValueError("subspace_size must be a float in (0, 1] or an int >= 1.")
        if isinstance(subspaces, str):
            if subspaces not in ("guided", "random"):
                raise ValueError(f"Unknown subspaces: {subspaces!r}")
        else:
            subspaces = [[str(f) for f in group] for group in subspaces]
            if not subspaces or any(len(group) == 0 for group in subspaces):
                raise ValueError("Explicit subspaces must be non-empty.")
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if stride is not None and not 1 <= stride <= window_size:
            raise ValueError("stride must be in [1, window_size].")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if reference_size is not None and reference_size < 2:
            raise ValueError("reference_size must be >= 2.")
        self.n_subspaces = int(n_subspaces)
        self.subspace_size = subspace_size
        self.subspaces = subspaces
        self.window_size = int(window_size)
        self.stride = int(stride) if stride is not None else self.window_size
        self.alpha = float(alpha)
        self.reference_size = None if reference_size is None else int(reference_size)
        self.random_state = random_state

        self._fitted = False
        self._reference: list[np.ndarray] = []
        self._low = np.empty(0)
        self._high = np.empty(0)
        self._groups: list[np.ndarray] = []
        self._monitored: list[int] = []
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._since_check = 0
        self.feature_names_: list[str] = []
        self.subspaces_: list[list[str]] = []
        self.critical_value_ = float("nan")
        self.last_votes_: dict[str, bool] | None = None
        self.last_score_: float | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> "KSSubspaceEnsembleDetector":
        """Take ``features`` as reference, build the subspaces, clear the window."""
        if len(features) < 2:
            raise ValueError("At least two reference instances are required.")
        if features.shape[1] == 0:
            raise ValueError("At least one feature is required.")
        self.feature_names_ = [str(c) for c in features.columns]
        X = features.to_numpy(float)
        if self.reference_size is not None:
            X = X[-self.reference_size :]

        self._reference = [_clean_sorted(X[:, j]) for j in range(X.shape[1])]
        self._low = np.array([r[0] if len(r) else np.nan for r in self._reference])
        self._high = np.array([r[-1] if len(r) else np.nan for r in self._reference])
        groups = self._make_subspaces(X)
        self._groups = [np.asarray(g, dtype=int) for g in groups]
        self._monitored = sorted({j for g in groups for j in g})
        self.subspaces_ = [[self.feature_names_[j] for j in g] for g in groups]
        self.critical_value_ = ks_critical_value(self.alpha, len(X), self.window_size)

        self._window.clear()
        self._since_check = 0
        self.last_votes_ = None
        self.last_score_ = None
        self._fitted = True
        return self

    def _make_subspaces(self, X: np.ndarray) -> list[list[int]]:
        d = X.shape[1]
        if not isinstance(self.subspaces, str):
            position = {name: j for j, name in enumerate(self.feature_names_)}
            unknown = [f for g in self.subspaces for f in g if f not in position]
            if unknown:
                raise ValueError(f"Unknown features in subspaces: {unknown[:5]}")
            return [[position[f] for f in group] for group in self.subspaces]
        if isinstance(self.subspace_size, float):
            size = max(1, math.ceil(self.subspace_size * d))
        else:
            size = min(self.subspace_size, d)
        rng = np.random.default_rng(self.random_state)
        if self.subspaces == "random":
            return [
                rng.choice(d, size=size, replace=False).tolist()
                for _ in range(self.n_subspaces)
            ]
        corr = pd.DataFrame(X).corr(method="spearman").to_numpy()
        return build_subspaces(corr, self.n_subspaces, size, rng)

    # --------------------------------------------------------------- update
    def update(
        self, features: pd.DataFrame, performance: pd.DataFrame | None = None
    ) -> DriftSignal:
        """
        Consume new instance(s) and report whether the latest check found drift.

        Accepts one or several rows; if several rows complete several checks,
        the signal reports drift if any of them did.
        """
        drift = False
        for row in self._matrix(features):
            self._window.append(row)
            self._since_check += 1
            if (
                len(self._window) == self.window_size
                and self._since_check >= self.stride
            ):
                self._since_check = 0
                drift |= self._check()
        return DriftSignal(drift=drift, score=self.last_score_)

    def _check(self) -> bool:
        """Run all KS tests on the current window and combine the votes."""
        window = np.vstack(self._window)
        rejects = np.zeros(len(self.feature_names_), dtype=bool)
        for j in self._monitored:
            reference, recent = self._reference[j], _clean_sorted(window[:, j])
            if len(reference) < 2 or len(recent) < 2:
                continue
            threshold = ks_critical_value(self.alpha, len(reference), len(recent))
            rejects[j] = _ks_sorted(reference, recent) > threshold
        shares = [float(rejects[g].mean()) for g in self._groups]
        self.last_votes_ = {
            self.feature_names_[j]: bool(rejects[j]) for j in self._monitored
        }
        self.last_score_ = max(shares)
        return self.last_score_ > 0.5

    # ---------------------------------------------------------------- score
    def score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Per-instance share of features outside the reference range.

        For each subspace, the share of its features whose value lies below
        the minimum or above the maximum of the reference data; an instance
        gets the largest share over all subspaces (0 = inside the seen range
        everywhere, 1 = some subspace is left in every feature). Missing
        values count as inside.
        """
        X = self._matrix(features)
        outside = (X < self._low) | (X > self._high)
        shares = np.column_stack([outside[:, g].mean(axis=1) for g in self._groups])
        return shares.max(axis=1)

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before update()/score().")
        if list(features.columns) == self.feature_names_:
            return features.to_numpy(float)
        missing = [
            c for c in self.feature_names_ if c not in map(str, features.columns)
        ]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]}")
        return features.rename(columns=str)[self.feature_names_].to_numpy(float)
