"""
Criteria for extra runs: reasons to run a second algorithm on an instance.

Each gate here is a :class:`~asf.online.gate.ScoreGate`: it rates an instance
with one number and fires either above a fixed threshold or, with ``rate=q``,
on the share ``q`` of recent instances with the highest ratings. Sweeping
``q`` from 0 to 1 gives every criterion (and every selector) a
cost-versus-quality curve between "never an extra run" and "always".

* :class:`~asf.online.gate.UncertaintyGate` -- the selector's uncertainty
  (needs ``PredictionRecord.uncertainty``).
* :class:`CloseCallGate` -- how close the two best algorithms are (needs
  ``PredictionRecord.scores``).
* :class:`UnfamiliarityGate` -- the detector's per-instance score (needs a
  feature detector with ``score``).
* :class:`TimeoutRiskGate` -- the predicted runtime of the chosen algorithm
  (needs ``PredictionRecord.predicted_runtime``).

A gate whose input is missing on an instance (e.g. the detector is not fitted
yet) never fires there under a threshold and fires with probability ``q``
under a rate.
"""

from __future__ import annotations

import math

from asf.online.gate import GateContext, ScoreGate


class CloseCallGate(ScoreGate):
    """
    Run extras when the two best algorithms are nearly tied.

    The gap is the difference between the highest and the second-highest
    entry of ``prediction.scores`` (votes for pairwise selectors, predicted
    performance for runtime models, the policy index for bandits); the
    smaller the gap, the more reason for an extra run. Non-finite scores are
    ignored; with fewer than two scores the instance is not rated.

    Parameters
    ----------
    max_gap : float or None, default=None
        Fire if the gap is at most ``max_gap`` (fixed rule). Exactly one of
        ``max_gap`` and ``rate`` must be given.
    n : int or None, default=1
        See :class:`~asf.online.gate.ScoreGate`.
    relative : bool, default=False
        Divide the gap by the larger absolute value of the two scores, e.g.
        to compare predicted runtimes across easy and hard instances.
    rate, window_size, random_state
        See :class:`~asf.online.gate.ScoreGate`.
    """

    def __init__(
        self,
        max_gap: float | None = None,
        n: int | None = 1,
        *,
        relative: bool = False,
        rate: float | None = None,
        window_size: int = 200,
        random_state: int | None = None,
    ) -> None:
        if max_gap is not None and max_gap < 0:
            raise ValueError("max_gap must be >= 0.")
        super().__init__(
            threshold=None if max_gap is None else -float(max_gap),
            n=n,
            rate=rate,
            window_size=window_size,
            random_state=random_state,
        )
        self.max_gap = None if max_gap is None else float(max_gap)
        self.relative = bool(relative)

    def _score(self, context: GateContext) -> float | None:
        scores = context.prediction.scores
        if scores is None:
            return None
        values = sorted(
            (float(v) for v in scores.values() if math.isfinite(v)), reverse=True
        )
        if len(values) < 2:
            return None
        gap = values[0] - values[1]
        if self.relative and gap > 0:
            gap /= max(abs(values[0]), abs(values[1]))
        return -gap


class UnfamiliarityGate(ScoreGate):
    """
    Run extras on instances the drift detector finds unfamiliar.

    The score is the per-instance score of the selector's feature detector
    (``AbstractDriftDetector.score``, higher = further from the reference
    data), so the selector needs such a detector. Until the detector is
    fitted, and for selectors without one, instances are not rated.

    Parameters
    ----------
    threshold : float or None, default=None
        Minimum detector score for extra runs (on the detector's scale).
    n, rate, window_size, random_state
        See :class:`~asf.online.gate.ScoreGate`.
    """

    needs_instance_score = True

    def _score(self, context: GateContext) -> float | None:
        return context.instance_score


class TimeoutRiskGate(ScoreGate):
    """
    Run extras when the chosen algorithm is predicted to be slow.

    The score is ``prediction.predicted_runtime``: the closer it is to the
    cutoff, the more likely the primary run times out and a second algorithm
    pays off. Selectors without a runtime prediction are not rated. Meant
    for runtime minimisation.

    Parameters
    ----------
    threshold : float or None, default=None
        Minimum predicted runtime for extra runs, on the stream's raw scale
        (e.g. ``0.8 * cutoff``). A rate needs no cutoff.
    n, rate, window_size, random_state
        See :class:`~asf.online.gate.ScoreGate`.
    """

    def _score(self, context: GateContext) -> float | None:
        runtime = context.prediction.predicted_runtime
        if runtime is None or not math.isfinite(runtime):
            return None
        return float(runtime)


__all__ = ["CloseCallGate", "UnfamiliarityGate", "TimeoutRiskGate"]
