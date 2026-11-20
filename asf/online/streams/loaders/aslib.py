"""ASlib scenario -> raw (features, performance, cutoff) for streaming."""

from __future__ import annotations

import pandas as pd

from asf.scenario.aslib_reader import read_aslib_scenario


def load_aslib_raw(path: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """``(features, performance, cutoff)`` of an ASlib scenario with raw
    runtimes (no PAR penalty), so the evaluator decides what is censored."""
    features, performance, _, _, _, maximize, cutoff, _ = read_aslib_scenario(
        path, training_par_factor=None
    )
    if maximize:
        raise ValueError("Online runtime streams assume minimisation.")
    return features, performance, float(cutoff)
