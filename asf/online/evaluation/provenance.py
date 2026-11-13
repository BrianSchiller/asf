"""Provenance for result files: which ASF code and which configuration ran."""

from __future__ import annotations

import inspect
import json
import math
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

import asf


def asf_provenance(refresh: bool = False) -> dict[str, Any]:
    """
    Version, commit and dirty flag of the ASF code that is imported.

    The answer is computed once per process (``git status`` can be slow on
    large or network checkouts) and cached; pass ``refresh=True`` to
    recompute it.

    Parameters
    ----------
    refresh : bool, default=False
        Ignore the cached answer.

    Returns
    -------
    dict
        ``version``, ``commit``, ``dirty`` and ``commit_source``:

        * ``"git"``: ASF is imported from a git checkout; ``dirty`` is true if
          tracked files differ from ``HEAD`` or untracked files exist under
          ``asf/``.
        * ``"direct_url"``: ASF was pip-installed from a pinned git URL; the
          commit comes from the install record and ``dirty`` is ``None``.
        * ``"unknown"``: neither; ``commit`` and ``dirty`` are ``None``.
    """
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = _compute_provenance()
    return dict(_CACHE)


_CACHE: dict[str, Any] | None = None


def _compute_provenance() -> dict[str, Any]:
    try:
        version: str | None = metadata.version("asf")
    except metadata.PackageNotFoundError:
        version = None
    out: dict[str, Any] = {
        "version": version,
        "commit": None,
        "dirty": None,
        "commit_source": "unknown",
    }
    root = Path(asf.__file__).resolve().parent.parent
    if (root / ".git").exists():
        commit = _git(root, "rev-parse", "HEAD")
        tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
        untracked = _git(
            root, "status", "--porcelain", "--untracked-files=all", "--", "asf"
        )
        if commit is not None and tracked is not None and untracked is not None:
            out["commit"] = commit
            out["dirty"] = bool(tracked) or any(
                line.startswith("??") for line in untracked.splitlines()
            )
            out["commit_source"] = "git"
            return out
    try:
        record = metadata.distribution("asf").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        record = None
    if record:
        commit = json.loads(record).get("vcs_info", {}).get("commit_id")
        if commit:
            out["commit"] = commit
            out["commit_source"] = "direct_url"
    return out


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def describe(obj: Any) -> Any:
    """
    JSON-ready description of a configuration value.

    Primitives are kept (non-finite floats become ``None``), containers are
    described element-wise, classes become their qualified name, and objects
    become ``{"class": ..., "params": ...}``: online selectors with their
    captured constructor arguments, other objects with the attributes that
    match their constructor parameters (scikit-learn convention). Anything
    else (e.g. ``functools.partial``) is described by its ``repr``.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, np.bool_):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): describe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [describe(v) for v in obj]
    if isinstance(obj, type):
        return qualified_name(obj)
    params = getattr(obj, "describe_params", None)
    if callable(params):
        return {"class": qualified_name(type(obj)), "params": describe(params())}
    init = type(obj).__init__
    if init is not object.__init__ and not inspect.isbuiltin(init):
        try:
            names = [
                p.name
                for p in inspect.signature(init).parameters.values()
                if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
        except (TypeError, ValueError):
            names = []
        if all(hasattr(obj, n) for n in names) and type(obj).__module__ != "builtins":
            return {
                "class": qualified_name(type(obj)),
                "params": {n: describe(getattr(obj, n)) for n in names},
            }
    return repr(obj)


def qualified_name(cls: type) -> str:
    """``module.QualName`` of a class."""
    return f"{cls.__module__}.{cls.__qualname__}"
