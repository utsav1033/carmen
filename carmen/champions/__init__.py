"""Kernels carmen already wrote and proved, shipped with the package.

Each folder is one model: the champion kernels from a targeted run (`carmen run <op> --for
<model>`), the config each won with, and the in-model result it was published with. They let
anyone reproduce that result with `carmen speedup`, with no API key and no Carmy calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..backends import Kernel

HERE = Path(__file__).resolve().parent


def published() -> dict:
    """Judge and per-kernel results published in the README (measured on an M4)."""
    return json.loads((HERE / "published.json").read_text())


def available() -> list[str]:
    return sorted(p.name for p in HERE.iterdir() if p.is_dir() and (p / "result.json").exists())


def result(name: str) -> dict:
    return json.loads((HERE / name / "result.json").read_text())


def results() -> list[dict]:
    """Every published in-model result, newest first."""
    return sorted((result(n) | {"id": n} for n in available()), key=lambda r: r.get("date", ""), reverse=True)


def kernels(name: str) -> dict[str, tuple[Kernel, dict, str]]:
    """{op: (kernel, the config it won with, where it came from)} for one bundled model."""
    out = {}
    for f in sorted((HERE / name).glob("*.json")):
        if f.name == "result.json":
            continue
        d = json.loads(f.read_text())
        out[f.stem] = (Kernel.from_json(d), d["config"], d.get("from", ""))
    return out


def for_model(model: str) -> str | None:
    """The bundled set made for this model id, if any."""
    return next((n for n in available() if result(n)["model"] == model), None)
