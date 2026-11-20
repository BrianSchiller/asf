"""
Drift detectors on a stream of scalars, typically the
:class:`~asf.online.drift.statistics.LogRuntimeResidual` of the primary run:
:class:`ADWINDetector` (the mean) and :class:`EDDMDetector` (the distance
between errors).

A censored value (lower bound) goes through ``imputation`` first: the bound
(default), dropped, or imputed above it from a normal belief with the mean
and sd of the uncensored values since the last reset.

References
----------
Bifet, A., Gavalda, R. (2007). "Learning from Time-Changing Data with
Adaptive Windowing". SDM 2007.

Baena-Garcia, M., del Campo-Avila, J., Fidalgo, R., Bifet, A., Gavalda, R.,
Morales-Bueno, R. (2006). "Early Drift Detection Method". Fourth
International Workshop on Knowledge Discovery from Data Streams.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from asf.online.drift.abstract_detector import AbstractScalarDriftDetector, DriftSignal
from asf.online.imputation import CensoredImputer, CutoffImputer


class _CensoredInput:
    """
    Imputes censored values for scalar detectors from the running mean and
    sd of the uncensored ones (no belief, ``sd = inf``, before two of them).
    ``imputation=None`` uses the bound; the rng restarts at every reset.
    """

    def __init__(
        self, imputation: CensoredImputer | None, random_state: int | None
    ) -> None:
        if imputation is not None and not isinstance(imputation, CensoredImputer):
            raise TypeError("imputation must be a CensoredImputer or None.")
        self.imputation = imputation
        self.random_state = random_state
        self.imputation_: CensoredImputer = (
            imputation if imputation is not None else CutoffImputer()
        )
        self._reset_belief()

    def _reset_belief(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self._n_exact = 0
        self._mean_exact = 0.0
        self._ss_exact = 0.0  # sum of squared deviations from the mean

    def _resolve(self, value: float, censored: bool) -> float | None:
        """The value to use (``None`` = skip); never below a censored bound."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Scalar detectors need finite values.")
        if not censored:
            self._n_exact += 1
            delta = value - self._mean_exact
            self._mean_exact += delta / self._n_exact
            self._ss_exact += delta * (value - self._mean_exact)
            return value
        mean, sd = value, math.inf
        if self.imputation_.USES_MODEL and self._n_exact >= 2:
            mean = self._mean_exact
            sd = math.sqrt(self._ss_exact / (self._n_exact - 1))
            if not self.imputation_.PREDICTIVE:  # spread of the mean itself
                sd /= math.sqrt(self._n_exact)
        imputed = float(self.imputation_(value, mean, sd, self._rng)[0])
        return None if math.isnan(imputed) else max(imputed, value)


class ADWINDetector(_CensoredInput, AbstractScalarDriftDetector):
    """
    ADWIN: adaptive windowing on the mean of a scalar stream (Bifet and
    Gavalda, 2007).

    After every value, while some split ``W = W0 . W1`` of the window has
    ``|mean(W1) - mean(W0)| >= eps_cut``, the oldest value is dropped; a
    shrinking window is the drift signal. With ``m = 1 / (1/n0 + 1/n1)``:

    * ``bound="hoeffding"`` (the rigorous test)::

          eps_cut = sqrt(ln(4 / d) / (2 m)),    d = delta / n

    * ``bound="variance"`` (eq. 2.1, the test the paper uses in practice),
      with ``var`` the variance of the values in ``W``::

          eps_cut = sqrt(2 / m * var * ln(2 / d)) + 2 / (3 m) * ln(2 / d),
          d = delta / ln(n)

    Values are clipped to ``value_range`` and rescaled to ``[0, 1]`` (for log
    residuals e.g. ``(-log 10, log 10)``). ``score`` is the window mean on the
    input scale; there is no warning zone.

    Assumptions and deviations
    --------------------------
    * Exact ADWIN, not the bucket-based ADWIN2: every split is checked, one
      pass over the window per update.
    * The oldest value is dropped one at a time, then all splits are tested
      again; no minimum sub-window length.
    * Values outside ``value_range`` are clipped, so larger deviations count
      the same.
    * ``delta=0.002`` is the MOA/river default (the paper has none).
    * ``direction="up"`` (not in the paper) only signals when the newer part
      has the larger mean: the algorithm got slower than predicted.
    * The "observed variance" of eq. 2.1 is the population variance.
    """

    def __init__(
        self,
        delta: float = 0.002,
        value_range: tuple[float, float] = (0.0, 1.0),
        bound: Literal["variance", "hoeffding"] = "variance",
        direction: Literal["both", "up"] = "both",
        imputation: CensoredImputer | None = None,
        random_state: int | None = None,
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1).")
        low, high = (float(v) for v in value_range)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError("value_range must be finite with low < high.")
        if bound not in ("variance", "hoeffding"):
            raise ValueError(f"Unknown bound: {bound!r}")
        if direction not in ("both", "up"):
            raise ValueError(f"Unknown direction: {direction!r}")
        self.delta = float(delta)
        self.value_range = (low, high)
        self.bound = bound
        self.direction = direction
        self._buffer = np.empty(64)
        super().__init__(imputation, random_state)
        self.reset()

    def reset(self) -> None:
        """Empty the window and forget the belief about censored values."""
        self._reset_belief()
        self._start = 0
        self._end = 0

    @property
    def width_(self) -> int:
        return self._end - self._start

    @property
    def estimate_(self) -> float | None:
        if self.width_ == 0:
            return None
        low, high = self.value_range
        return low + (high - low) * float(self._window().mean())

    def update(self, value: float, censored: bool = False) -> DriftSignal:
        """Add a value; drift if the window shrank (``"up"``: as values rose)."""
        resolved = self._resolve(value, censored)
        if resolved is None:
            return DriftSignal(drift=False, score=self.estimate_)
        low, high = self.value_range
        self._append((min(max(resolved, low), high) - low) / (high - low))
        drift = False
        while self.width_ >= 2:
            changed, rose = self._split_differs(self._window())
            if not changed:
                break
            drift = drift or rose or self.direction == "both"
            self._start += 1
        return DriftSignal(drift=drift, score=self.estimate_)

    def _window(self) -> np.ndarray:
        return self._buffer[self._start : self._end]

    def _append(self, x: float) -> None:
        if self._end == len(self._buffer):  # compact, grow if still full
            window = self._window().copy()
            if 2 * len(window) > len(self._buffer):
                self._buffer = np.empty(2 * len(self._buffer))
            self._buffer[: len(window)] = window
            self._start, self._end = 0, len(window)
        self._buffer[self._end] = x
        self._end += 1

    def _split_differs(self, window: np.ndarray) -> tuple[bool, bool]:
        """Whether some split reaches ``eps_cut``, and whether one of those
        has the larger mean in its newer part."""
        n = len(window)
        sums = np.cumsum(window)
        n0 = np.arange(1, n)
        n1 = n - n0
        difference = (sums[-1] - sums[:-1]) / n1 - sums[:-1] / n0
        inverse_m = 1.0 / n0 + 1.0 / n1
        if self.bound == "hoeffding":
            eps = np.sqrt(math.log(4.0 * n / self.delta) * inverse_m / 2.0)
        else:
            log_term = math.log(2.0 * math.log(n) / self.delta)
            eps = (
                np.sqrt(2.0 * inverse_m * float(window.var()) * log_term)
                + 2.0 / 3.0 * inverse_m * log_term
            )
        differs = np.abs(difference) >= eps
        return bool(differs.any()), bool((differs & (difference > 0)).any())


class EDDMDetector(_CensoredInput, AbstractScalarDriftDetector):
    """
    EDDM: early drift detection on the distance between errors (Baena-Garcia
    et al., 2006).

    A value above ``error_threshold`` is an error (default ``log(2)``: more
    than twice as slow as predicted; a 0/1 stream works as it is). With ``p``
    and ``s`` the running mean and sd of the distances between errors, at
    every error once ``min_errors`` have been seen::

        ratio = (p + 2 s) / (p_max + 2 s_max)
        ratio < alpha  ->  warning
        ratio < beta   ->  drift

    Errors getting closer lower the ratio. ``score`` is the ratio at the
    last error; ``warning`` holds until an error lifts it back to ``alpha``.
    Defaults are the paper's (``alpha=0.95``, ``beta=0.90``, 30 errors).

    Assumptions and deviations
    --------------------------
    * The one-sided error threshold is not in the paper (a timeout can only
      show a run was slower than predicted).
    * The first distance counts from the start (or last reset), as in river
      and MOA; ``s`` is the sample sd (0 for one distance), as in river.
    * ``p + 2 s`` is tracked from the first error; only the tests wait for
      ``min_errors`` (river waits 30 instances for the maximum).
    * After a drift all statistics restart, including the censored belief.
    * Censored values are imputed before the threshold; by default a timeout
      is an error only if its bound is above the threshold.
    """

    def __init__(
        self,
        error_threshold: float = math.log(2),
        alpha: float = 0.95,
        beta: float = 0.90,
        min_errors: int = 30,
        imputation: CensoredImputer | None = None,
        random_state: int | None = None,
    ) -> None:
        if not math.isfinite(error_threshold):
            raise ValueError("error_threshold must be finite.")
        if not 0.0 < beta <= alpha <= 1.0:
            raise ValueError("Need 0 < beta <= alpha <= 1.")
        if min_errors < 1:
            raise ValueError("min_errors must be >= 1.")
        self.error_threshold = float(error_threshold)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_errors = int(min_errors)
        super().__init__(imputation, random_state)
        self.reset()

    def reset(self) -> None:
        """Forget all errors and the belief about censored values."""
        self._reset_belief()
        self._n = 0  # instances seen
        self._last_error = 0  # position of the last error
        self._n_errors = 0
        self._mean = 0.0  # of the distances
        self._ss = 0.0  # their sum of squared deviations
        self._maximum = -math.inf  # of p + 2 s
        self._ratio: float | None = None
        self._warning = False

    def update(self, value: float, censored: bool = False) -> DriftSignal:
        resolved = self._resolve(value, censored)
        if resolved is None:
            return DriftSignal(False, self._warning, self._ratio)
        self._n += 1
        if not resolved > self.error_threshold:
            return DriftSignal(False, self._warning, self._ratio)

        self._n_errors += 1
        distance = self._n - self._last_error
        self._last_error = self._n
        delta = distance - self._mean
        self._mean += delta / self._n_errors
        self._ss += delta * (distance - self._mean)
        sd = math.sqrt(self._ss / (self._n_errors - 1)) if self._n_errors > 1 else 0.0
        level = self._mean + 2.0 * sd
        self._maximum = max(self._maximum, level)
        self._ratio = ratio = level / self._maximum
        tested = self._n_errors >= self.min_errors
        drift = tested and ratio < self.beta
        self._warning = tested and not drift and ratio < self.alpha
        signal = DriftSignal(drift, self._warning, ratio)
        if drift:
            self.reset()
        return signal
