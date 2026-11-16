"""Turning observed runtimes (or quality scores) into bandit rewards."""

from __future__ import annotations

import math
from typing import Collection, Mapping


class RewardTransform:
    """
    Map an arm's observed performance to a reward (higher is better).

    Parameters
    ----------
    kind : {"auto", "par", "log", "raw"}, default="auto"
        * ``"log"``: ``-log10(PAR_k(runtime) + 1)``. Differences between
          algorithms that both solve an instance stay visible (as in
          Tornede et al.).
        * ``"par"``: ``-PAR_k(runtime) / (k * cutoff)``, in ``[-1, 0]``.
          Matches PAR-k scoring, but runs well below the cutoff all get a
          reward close to 0, so the bandit mostly learns "solved or not".
          Needs a finite cutoff.
        * ``"unit"``: ``1 - log10(PAR_k + 1) / log10(k * cutoff + 1)``, the
          log reward rescaled to ``[0, 1]`` (1 = instant, 0 = unsolved), for
          Beta posteriors. Needs a finite cutoff.
        * ``"solved"``: 1 if solved within the cutoff, else 0 (binary).
        * ``"raw"``: the value itself when maximizing, its negation otherwise.
        * ``"auto"``: ``"raw"`` when maximizing, ``"log"`` otherwise.
    par_factor : float, default=10.0
        Penalty factor ``k`` for unsolved runs.
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
        Reward of a run that solved the instance in exactly ``runtime``.

        Used as the bound when imputing censored runs: a run stopped at its
        budget would have earned at most the reward of solving at the budget.
        Only for the runtime rewards ``"log"`` and ``"par"``.

        Parameters
        ----------
        runtime : float
            The runtime (e.g. the budget a censored run was stopped at).
        cutoff : float
            The scenario cutoff (scales the ``"par"`` reward).

        Returns
        -------
        float
            The reward, without any penalty.
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
        Turn a reward back into a performance value (the inverse transform).

        Used for ``PredictionRecord.predicted_runtime``: applied to an arm's
        expected reward it gives the predicted PAR-k value on the raw scale
        (unsolved runs were learnt with their penalty). For the ``"log"`` and
        ``"unit"`` rewards this is a geometric rather than an arithmetic mean.

        Parameters
        ----------
        reward : float
            A reward, e.g. the posterior mean of an arm.
        cutoff : float
            The scenario cutoff (needed for ``"par"`` and ``"unit"``).
        maximize : bool
            Whether the scenario maximizes a quality score.

        Returns
        -------
        float or None
            The performance (runtimes are at least 0); ``None`` if the reward
            cannot be inverted (``"solved"``, or no finite cutoff).
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

        def penalised(algo: str, value: float) -> float:
            unsolved = algo in censored or value >= cutoff
            return self.par_factor * cutoff if unsolved else float(value)

        out: dict[str, float] = {}
        for algo, value in runtimes.items():
            p = penalised(algo, value)
            if kind == "solved":
                out[algo] = 0.0 if algo in censored or value >= cutoff else 1.0
            elif kind == "par":
                out[algo] = -p / (self.par_factor * cutoff)
            elif kind == "unit":
                worst = math.log10(self.par_factor * cutoff + 1.0)
                out[algo] = 1.0 - math.log10(p + 1.0) / worst
            else:
                out[algo] = -math.log10(p + 1.0)
        return out
