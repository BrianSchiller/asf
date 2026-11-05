"""Sliding-window store of partially observed instances."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from asf.online.base import Observation


class ObservationBuffer:
    """
    Sliding window over the most recent instances seen by an online selector.

    For every instance it stores the features and a *partially observed*
    performance row: algorithms that were not run are ``NaN``. A parallel
    boolean mask records which observed entries were censored (run stopped at
    its budget), which is what censored-imputation methods need.

    Parameters
    ----------
    algorithms : Sequence[str]
        Algorithm names (column order of the performance matrix).
    feature_names : Sequence[str]
        Feature names (column order of the feature matrix).
    window_size : int or None, default=200
        Maximum number of instances kept; older instances are dropped.
        ``None`` keeps the full history.
    """

    def __init__(
        self,
        algorithms: Sequence[str],
        feature_names: Sequence[str],
        window_size: int | None = 200,
    ) -> None:
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be >= 1.")
        self.algorithms = [str(a) for a in algorithms]
        self.feature_names = [str(f) for f in feature_names]
        self.window_size = None if window_size is None else int(window_size)
        self._ids: deque[Any] = deque(maxlen=window_size)
        self._x: deque[np.ndarray] = deque(maxlen=window_size)
        self._y: deque[np.ndarray] = deque(maxlen=window_size)
        self._censored: deque[np.ndarray] = deque(maxlen=window_size)

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------- writing
    def append(self, observation: "Observation") -> None:
        """Add one instance with its partially observed runtimes."""
        y = np.full(len(self.algorithms), np.nan)
        c = np.zeros(len(self.algorithms), dtype=bool)
        for algo, runtime in observation.runtimes.items():
            j = self.algorithms.index(str(algo))
            y[j] = float(runtime)
            c[j] = str(algo) in observation.censored
        x = observation.features.iloc[0].reindex(self.feature_names).to_numpy(float)
        self._ids.append(observation.instance_id)
        self._x.append(x)
        self._y.append(y)
        self._censored.append(c)

    def extend_full(self, features: pd.DataFrame, performance: pd.DataFrame) -> None:
        """Add fully observed instances (used for the warm start)."""
        perf = performance.reindex(columns=self.algorithms)
        feats = features.reindex(columns=self.feature_names)
        for idx in features.index:
            self._ids.append(idx)
            self._x.append(feats.loc[idx].to_numpy(float))
            self._y.append(perf.loc[idx].to_numpy(float))
            self._censored.append(np.zeros(len(self.algorithms), dtype=bool))

    # ------------------------------------------------------------- reading
    def features(self) -> pd.DataFrame:
        """Window features, one row per instance."""
        return pd.DataFrame(
            np.array(self._x).reshape(len(self), len(self.feature_names)),
            index=pd.Index(list(self._ids)),
            columns=pd.Index(self.feature_names),
        )

    def performance(self) -> pd.DataFrame:
        """Window performance; ``NaN`` where an algorithm was not run."""
        return pd.DataFrame(
            np.array(self._y).reshape(len(self), len(self.algorithms)),
            index=pd.Index(list(self._ids)),
            columns=pd.Index(self.algorithms),
        )

    def censored_mask(self) -> pd.DataFrame:
        """Boolean frame: ``True`` where the observed runtime was censored."""
        return pd.DataFrame(
            np.array(self._censored).reshape(len(self), len(self.algorithms)),
            index=pd.Index(list(self._ids)),
            columns=pd.Index(self.algorithms),
        )

    def recent(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Features and performance of the ``n`` most recent instances."""
        return self.features().tail(n), self.performance().tail(n)

    def n_observed(self, algorithm: str) -> int:
        """Number of window instances on which ``algorithm`` was run."""
        j = self.algorithms.index(str(algorithm))
        return int(sum(not np.isnan(y[j]) for y in self._y))

    def pair_data(
        self, algo_a: str, algo_b: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Window instances on which BOTH algorithms were observed.

        This is the training data of one pairwise (one-vs-one) classifier.

        Returns
        -------
        tuple
            ``(features, runtimes_a, runtimes_b)`` restricted to the common rows.
        """
        perf = self.performance()
        mask = perf[algo_a].notna() & perf[algo_b].notna()
        return (
            self.features().loc[mask.to_numpy()],
            perf.loc[mask, algo_a],
            perf.loc[mask, algo_b],
        )
