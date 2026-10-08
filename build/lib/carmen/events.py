"""Append-only run log. The harness writes events; a UI (or `carmen report`) only reads them."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Run:
    def __init__(self, root: Path, run_id: str | None = None):
        self.id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = self.dir / "events.jsonl"

    def log(self, type_: str, **data) -> None:
        with self._events.open("a") as f:
            f.write(json.dumps({"t": time.time(), "type": type_, **data}, default=float) + "\n")

    def save(self, rel: str, obj) -> Path:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=float))
        return path


def read_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
