"""Minimal .env loader: KEY=value lines, # comments, optional quotes, optional `export`.

Variables already set in the environment win, so a shell `export` always overrides `.env`.
Order: your shell, then ./.env in the folder you run carmen from, then the per-user file
`carmen key` writes (~/.config/carmen/.env), so a project can override your default key.
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


def user_env() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "carmen" / ".env"


def save(key: str, value: str, path: Path | None = None) -> Path:
    """Set KEY=value in the per-user file (created readable by you only), keeping its other lines."""
    p = Path(path or user_env())
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [l for l in (p.read_text().splitlines() if p.exists() else [])
             if not l.strip().removeprefix("export ").startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    p.touch(mode=0o600, exist_ok=True)
    os.chmod(p, 0o600)
    p.write_text("\n".join(lines) + "\n")
    os.environ[key] = value
    return p


def remove(key: str, path: Path | None = None) -> bool:
    p = Path(path or user_env())
    if not p.exists():
        return False
    lines = p.read_text().splitlines()
    keep = [l for l in lines if not l.strip().removeprefix("export ").startswith(f"{key}=")]
    p.write_text("\n".join(keep) + ("\n" if keep else ""))
    return len(keep) != len(lines)


def masked(value: str) -> str:
    return f"…{value[-4:]}" if len(value) > 8 else "set"
