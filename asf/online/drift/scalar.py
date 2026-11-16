"""
Performance-based drift detectors on a stream of scalars: ADWIN and EDDM.

Both consume one number per instance, typically the
:class:`~asf.online.drift.statistics.LogRuntimeResidual` of the primary run
(``log(observed) - log(predicted)`` runtime):

* :class:`ADWINDetector` watches the mean of the values (Bifet and Gavalda,
  2007).
* :class:`EDDMDetector` turns each value into an error ("more than
  ``error_threshold`` above the prediction") and watches the distance
  between errors (Baena-Garcia et al., 2006).

Censored runs
-------------
If the primary run hit its budget, the value is only a *lower bound*. Both
detectors pass it through a :class:`~asf.online.imputation.CensoredImputer`
first (``imputation``): use the bound itself (default), drop the instance, or
impute a value above the bound. The belief the imputing rules need is the
normal distribution with the mean and standard deviation of the uncensored
values seen since the last reset.

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
    Shared handling of censored values (lower bounds) for scalar detectors.

    Keeps the running mean and standard deviation of the uncensored values
    since the last reset and uses them as the belief of the imputation rule.
    With fewer than two uncensored values there is no belief yet (``sd =
    inf``), so the imputing rules return the bound.

    Parameters
    ----------
    imputation : CensoredImputer or None
        Rule applied to censored values; ``None`` = use the bound
        (:class:`~asf.online.imputation.CutoffImputer`).
    random_state : int or None
        Seed of the stochastic rules. The generator restarts at every reset.
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
        """
        The value the detector uses, or ``None`` to skip the instance.

        Uncensored values pass through and update the belief. Censored ones
        go through the imputation rule; the result is never below the bound.
        """
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
    ADWIN: adaptive windowing on the mean of a scalar stream.

    The detector keeps a window ``W`` of the most recent values. After every
    new value it checks all splits ``W = W0 . W1`` (old part, new part) and,
    while some split has ``|mean(W1) - mean(W0)| >= eps_cut``, drops the oldest
    value (Figure 1 of the paper). A shrinking window is the drift signal.
    With ``n0``, ``n1`` and ``n`` the lengths of ``W0``, ``W1`` and ``W`` and
    ``m = 1 / (1/n0 + 1/n1)``:

    * ``bound="hoeffding"`` (the rigorous test)::

          eps_cut = sqrt(ln(4 / d) / (2 m)),    d = delta / n

    * ``bound="variance"`` (eq. 2.1, the test the paper uses in practice),
      with ``var`` the variance of the values in ``W``::

          eps_cut = sqrt(2 / m * var * ln(2 / d)) + 2 / (3 m) * ln(2 / d),
          d = delta / ln(n)

    Both tests assume values in ``[0, 1]``; other ranges are rescaled
    (``value_range``).

    Signal: ``drift`` is ``True`` if the window shrank at this step,
    ``score`` is the mean of the window on the input scale (``None`` while
    it is empty). ADWIN has no warning zone.

    Assumptions and deviations
    --------------------------
    * Deviation (exact ADWIN): this is ADWIN, not the bucket-based ADWIN2,
      which only approximates it to save time and memory. Every split is
      checked; an update costs one pass over the window.
    * Assumption (shrinking): the paper says "drop elements from the tail";
      here one value (the oldest) is dropped at a time and all splits are
      tested again. No minimum sub-window length is needed, a very short
      sub-window cannot reach ``eps_cut``.
    * Assumption (range): the paper handles a known interval ``[a, b]`` by
      rescaling. Values outside ``value_range`` are clipped to it, so larger
      deviations all count the same.
    * Assumption (``delta``): the paper has no default; 0.002 is the default
      of MOA and river.
    * Deviation (``direction="up"``, not in the paper): the window shrinks as
      usual, but drift is only signalled if the newer part has the larger
      mean, i.e. the values rose. For runtime residuals: the algorithm got
      slower than predicted, not faster.
    * Assumption (variance): the "observed variance" of eq. 2.1 is the
      population variance of the window.

    Parameters
    ----------
    delta : float, default=0.002
        Confidence parameter in ``(0, 1)``; smaller = fewer false alarms,
        later detection.
    value_range : tuple[float, float], default=(0.0, 1.0)
        Interval ``(low, high)`` of the values; they are clipped to it and
        rescaled to ``[0, 1]``. For log runtime residuals e.g.
        ``(-math.log(10), math.log(10))`` (a factor of 10 either way).
    bound : {"variance", "hoeffding"}, default="variance"
        The threshold ``eps_cut`` (see above).
    direction : {"both", "up"}, default="both"
        ``"both"``: signal every change of the mean (paper). ``"up"``: only
        increases.
    imputation : CensoredImputer or None, default=None
        Rule for censored values (lower bounds); ``None`` = use the bound.
    random_state : int or None, default=None
        Seed for stochastic imputation rules.

    Attributes
    ----------
    width_ : int
        Current length of the window.
    estimate_ : float or None
        Mean of the window on the input scale.
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
        """
        Add one value to the window and shrink it while a split differs.

        Parameters
        ----------
        value : float
            The statistic for the current instance.
        censored : bool, default=False
            ``True`` if ``value`` is a lower bound; see ``imputation``.

        Returns
        -------
        DriftSignal
            ``drift`` if the window shrank (for ``direction="up"``: because
            the values rose); ``score`` = mean of the window.
        """
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
        """
        Test every split of ``window``.

        Returns
        -------
        tuple[bool, bool]
            Whether some split reaches ``eps_cut``, and whether one of those
            has the larger mean in its newer part.
        """
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

    def __repr__(self) -> str:
        return (
            f"ADWINDetector(delta={self.delta}, value_range={self.value_range}, "
            f"bound={self.bound!r}, direction={self.direction!r}, "
            f"imputation={self.imputation!r}, random_state={self.random_state})"
        )


class EDDMDetector(_CensoredInput, AbstractScalarDriftDetector):
    """
    EDDM: early drift detection on the distance between errors.

    EDDM is defined on a stream of classification errors. Here a value is an
    *error* if it is above ``error_threshold``; with log runtime residuals
    and the default ``log(2)``: the run took more than twice as long as
    predicted. A 0/1 error stream can be passed as it is.

    The distance between two errors is the number of instances from one error
    to the next. With ``p`` their running mean and ``s`` their running
    standard deviation, the detector stores the maximum of ``p + 2 s`` and
    compares, at every error once ``min_errors`` errors have been seen::

        ratio = (p + 2 s) / (p_max + 2 s_max)
        ratio < alpha  ->  warning
        ratio < beta   ->  drift

    Errors getting closer together lower the ratio. After a drift the
    detector starts from scratch.

    Signal: ``score`` is the ratio at the last error (``None`` before the
    first error), ``warning`` holds until an error brings the ratio back to
    ``alpha`` or above.

    Assumptions and deviations
    --------------------------
    * Deviation (real values): the threshold turning a value into an error is
      not part of the paper. It is one-sided because a timeout can only show
      that a run was slower than predicted.
    * Assumption (first distance): the first error's distance is counted from
      the start of the stream (or the last reset), as in river's and MOA's
      implementation.
    * Assumption (standard deviation): the sample standard deviation
      (``n - 1``; 0 for a single distance), as in river.
    * Assumption (maximum): ``p + 2 s`` is tracked from the first error on;
      only the thresholds wait for ``min_errors`` errors. (river tracks the
      maximum only after 30 instances and tests after more than 30 errors.)
    * Assumption (after a drift): the paper resets ``p_max`` and ``s_max``
      and learns a new model; here all statistics restart, including the
      belief about censored values.
    * Censored values are imputed (``imputation``) before the threshold. With
      the default rule a timeout is an error only if its bound is above the
      threshold.

    Parameters
    ----------
    error_threshold : float, default=log(2)
        A value above it counts as an error.
    alpha : float, default=0.95
        Warning threshold (paper: 0.95).
    beta : float, default=0.90
        Drift threshold (paper: 0.90); at most ``alpha``.
    min_errors : int, default=30
        Errors needed before the thresholds are tested (paper: 30).
    imputation : CensoredImputer or None, default=None
        Rule for censored values (lower bounds); ``None`` = use the bound.
    random_state : int or None, default=None
        Seed for stochastic imputation rules.
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
        """
        Consume one value.

        Parameters
        ----------
        value : float
            The statistic for the current instance.
        censored : bool, default=False
            ``True`` if ``value`` is a lower bound; see ``imputation``.

        Returns
        -------
        DriftSignal
            ``drift`` and ``warning`` as defined above; ``score`` = ratio.
        """
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

    def __repr__(self) -> str:
        return (
            f"EDDMDetector(error_threshold={self.error_threshold}, "
            f"alpha={self.alpha}, beta={self.beta}, min_errors={self.min_errors}, "
            f"imputation={self.imputation!r}, random_state={self.random_state})"
        )
