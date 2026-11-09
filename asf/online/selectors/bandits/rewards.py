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
        * ``"raw"``: the value itself when maximizing, its negation otherwise.
        * ``"auto"``: ``"raw"`` when maximizing, ``"log"`` otherwise.
    par_factor : float, default=10.0
        Penalty factor ``k`` for unsolved runs.
    """

    KINDS = ("auto", "par", "log", "raw")

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
        if kind == "par" and not math.isfinite(cutoff):
            raise ValueError("Reward kind 'par' needs a finite cutoff.")

        def penalised(algo: str, value: float) -> float:
            unsolved = algo in censored or value >= cutoff
            return self.par_factor * cutoff if unsolved else float(value)

        out: dict[str, float] = {}
        for algo, value in runtimes.items():
            p = penalised(algo, value)
            if kind == "par":
                out[algo] = -p / (self.par_factor * cutoff)
            else:
                out[algo] = -math.log10(p + 1.0)
        return out
