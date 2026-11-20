"""Turning observed runtimes (or quality scores) into bandit rewards."""

from __future__ import annotations

import math
from typing import Collection, Mapping


class RewardTransform:
    """
    Map an arm's observed performance to a reward (higher is better).

    ``kind`` (PAR-k with ``par_factor`` k; ``"par"`` and ``"unit"`` need a
    finite cutoff):

    * ``"log"``: ``-log10(PAR_k + 1)``; differences between solved runs stay
      visible (as in Tornede et al.).
    * ``"par"``: ``-PAR_k / (k * cutoff)`` in ``[-1, 0]``; mostly "solved or
      not", since fast runs all get almost 0.
    * ``"unit"``: the log reward rescaled to ``[0, 1]``, for Beta posteriors.
    * ``"solved"``: 1 if solved, else 0.
    * ``"raw"``: the value when maximizing, its negation otherwise.
    * ``"auto"``: ``"raw"`` when maximizing, ``"log"`` otherwise.
    """

    KINDS = ("auto", "par", "log", "unit", "solved", "raw")

    def __init__(self, kind: str = "auto", par_factor: float = 10.0) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}.")
        self.kind = kind
        self.par_factor = float(par_factor)

    def resolve(self, cutoff: float, maximize: bool) -> str:
        """The concrete kind used for this cutoff / objective."""
        if self.kind != "auto":
            return self.kind
        return "raw" if maximize else "log"

    def solved_reward(self, runtime: float, cutoff: float) -> float:
        """
        Reward of solving in exactly ``runtime``, without penalty (``"log"``
        and ``"par"`` only): the bound for imputing a censored run.
        """
        kind = self.resolve(cutoff, maximize=False)
        if kind == "log":
            return -math.log10(float(runtime) + 1.0)
        if kind == "par":
            if not math.isfinite(cutoff):
                raise ValueError("Reward kind 'par' needs a finite cutoff.")
            return -float(runtime) / (self.par_factor * cutoff)
        raise ValueError(f"solved_reward is defined for 'log' and 'par', not {kind!r}.")

    def to_performance(
        self, reward: float, cutoff: float, maximize: bool
    ) -> float | None:
        """
        Inverse transform, e.g. an arm's mean reward -> predicted PAR-k (a
        geometric mean for ``"log"``/``"unit"``). ``None`` if not invertible
        (``"solved"``, or no finite cutoff).
        """
        kind = self.resolve(cutoff, maximize)
        if kind == "raw":
            return float(reward) if maximize else -float(reward)
        worst = self.par_factor * cutoff
        if kind == "log":
            value = 10.0 ** min(-reward, 300.0) - 1.0  # capped: no overflow
        elif kind == "par" and math.isfinite(cutoff):
            value = -reward * worst
        elif kind == "unit" and math.isfinite(cutoff):
            value = (worst + 1.0) ** (1.0 - reward) - 1.0
        else:
            return None
        return max(float(value), 0.0)

    def __call__(
        self,
        runtimes: Mapping[str, float],
        censored: Collection[str],
        cutoff: float,
        maximize: bool,
    ) -> dict[str, float]:
        """Reward per executed arm."""
        kind = self.resolve(cutoff, maximize)
        if kind == "raw":
            sign = 1.0 if maximize else -1.0
            return {a: sign * float(v) for a, v in runtimes.items()}
        if maximize:
            raise ValueError(f"Reward kind {kind!r} is for runtime minimization.")
        if kind in ("par", "unit") and not math.isfinite(cutoff):
            raise ValueError(
                f"Reward kind {kind!r} needs a finite cutoff "
                "(pass budget=<cutoff> to the selector for the warm start)."
            )

        worst = self.par_factor * cutoff
        out: dict[str, float] = {}
        for algo, value in runtimes.items():
            solved = algo not in censored and value < cutoff
            par = float(value) if solved else worst
            if kind == "solved":
                out[algo] = float(solved)
            elif kind == "par":
                out[algo] = -par / worst
            elif kind == "unit":
                out[algo] = 1.0 - math.log10(par + 1.0) / math.log10(worst + 1.0)
            else:
                out[algo] = -math.log10(par + 1.0)
        return out
