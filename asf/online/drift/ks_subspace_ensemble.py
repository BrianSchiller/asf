"""
Feature drift detection with an ensemble of KS tests over feature subspaces.

EDFS (Korycki and Krawczyk, DSAA 2019) adapted to covariate drift: features
are grouped into subspaces (Algorithm 1, greedy round-robin); every
``stride`` instances each feature is KS-tested between the reference and the
last ``window_size`` instances; a subspace drifts if a strict majority of
its tests reject, the detector if any subspace does.

Differences from the paper:

* A candidate's quality is its mean absolute Spearman correlation with the
  subspace's features, not mutual information with the label (no labels,
  no performance data).
* The diversity term of Eq. 12 includes the candidate (as printed it does
  not depend on it).
* The KS statistic is computed exactly at every check, not incrementally
  (same result).
* ``reset`` takes a new reference and rebuilds the subspaces.
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
    """Two-sample KS statistic ``max |F_a - F_b|`` (Eq. 4); NaNs dropped."""
    a, b = _clean_sorted(a), _clean_sorted(b)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both samples need at least one non-missing value.")
    return _ks_sorted(a, b)


def ks_critical_value(alpha: float, n: int, m: int) -> float:
    """KS rejection threshold at level ``alpha`` for sample sizes ``n, m``
    (Eq. 1): ``sqrt(-ln(alpha / 2) / 2) * sqrt((n + m) / (n m))``."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if n < 1 or m < 1:
        raise ValueError("Sample sizes must be >= 1.")
    return math.sqrt(-math.log(alpha / 2) / 2) * math.sqrt((n + m) / (n * m))


def build_subspaces(
    corr: np.ndarray, n_subspaces: int, size: int, rng: np.random.Generator
) -> list[list[int]]:
    """
    Group features into ``n_subspaces`` of ``size`` (Algorithm 1, no labels).

    Round-robin, each subspace ``S_i`` takes the feature ``c`` maximising
    ``quality + diversity``: quality = mean ``|corr[c, s]|`` over ``s`` in
    ``S_i`` (NaN = 0); diversity = mean of ``1 - share of subspaces holding
    c`` (Eq. 11) and ``1 - max_l |S_i' & S_l| / |S_i'|`` with ``S_i' = S_i +
    {c}`` (Eq. 12). ``rng`` breaks ties. Returns feature indices per subspace.
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

    Every ``stride`` instances (default ``window_size``) each subspace
    feature of the last ``window_size`` instances is tested against the
    reference (the last ``reference_size`` fit rows, ``None`` = all), so a
    signal comes up to ``stride`` instances late. While the buffer mixes
    data from before and after a drift, later checks can signal again.
    Unmonitored features, missing values and features with fewer than two
    values do not vote.

    Parameters
    ----------
    subspace_size : float or int, default=0.1
        Features per subspace: a share (rounded up, paper: 0.1) or a count.
    subspaces : {"guided", "random"} or sequence of sequences of str
        ``"guided"``: :func:`build_subspaces` on Spearman correlations;
        ``"random"``: the paper's RFS baseline; explicit feature-name groups
        override ``n_subspaces`` and ``subspace_size``.
    alpha : float, default=0.01
        Significance level of each KS test.

    Attributes: ``subspaces_``, ``critical_value_`` (for full, NaN-free
    samples), ``last_votes_`` (per feature) and ``last_score_`` (largest
    rejecting share of any subspace; drift if above 0.5).
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
        X = self._reference_matrix(features)
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
        """Consume one or more rows; drift if any completed check drifted."""
        drift = self._strided_checks(self._matrix(features), self._window, self._check)
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
        """Per instance, the largest share of a subspace's features outside
        the reference range (missing values count as inside)."""
        X = self._matrix(features)
        outside = (X < self._low) | (X > self._high)
        shares = np.column_stack([outside[:, g].mean(axis=1) for g in self._groups])
        return shares.max(axis=1)
