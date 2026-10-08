"""Minimal .env loader: KEY=value lines, # comments, optional quotes, optional `export`.

Variables already set in the environment win, so a shell `export` always overrides `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path


def load(path: str | Path = ".env") -> list[str]:
    """Load `path` into os.environ. Returns the names it set."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for lineno, raw in enumerate(p.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            raise ValueError(f"{p}:{lineno}: expected KEY=value, got {raw!r}")
        key, value = (s.strip() for s in line.split("=", 1))
        if not key.isidentifier():
            raise ValueError(f"{p}:{lineno}: invalid variable name {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
