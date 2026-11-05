"""
Run one (method x stream x seed) online experiment and store the results.

TODO: parse a config from ``bench/online/configs/``, build stream + selector,
call ``asf.online.evaluation.run_prequential`` and write ``OnlineResult.steps``
and ``OnlineResult.summary()`` to disk (one file per run, so runs can be
parallelised on a cluster).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
