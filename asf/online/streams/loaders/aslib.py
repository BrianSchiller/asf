"""ASlib scenario -> raw (features, performance, cutoff) for streaming."""

from __future__ import annotations

import pandas as pd

from asf.scenario.aslib_reader import read_aslib_scenario


def load_aslib_raw(path: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Read an ASlib scenario WITHOUT PAR-penalising timeouts.

    ``read_aslib_scenario`` replaces timeouts by ``cutoff * 10`` by default
    (``training_par_factor=10.0``). For streaming we need raw runtimes so the
    evaluator can decide what counts as censored, hence ``None`` here.

    Returns
    -------
    tuple
        ``(features, performance, cutoff)``.
    """
    features, performance, _, _, _, maximize, cutoff, _ = read_aslib_scenario(
        path, training_par_factor=None
    )
    if maximize:
        raise ValueError("Online runtime streams assume minimisation.")
    return features, performance, float(cutoff)
