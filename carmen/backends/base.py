"""What a kernel is, and what a chip adapter must provide.

A kernel is Metal (or CUDA, later) *source text* plus launch configs. It is never
Python: Carmy cannot run code inside the judge, only submit a kernel body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

# Extra elements allocated after the real output. They start as NaN and must stay NaN:
# a write here means the kernel wrote past the end of its output.
PAD = 64

MAX_SOURCE_CHARS = 30_000
MAX_CONFIGS = 6
BANNED_TOKENS = ("#include", "asm(", "__asm", "#pragma")


@dataclass
class Kernel:
    source: str
    header: str = ""
    # Each config maps template constant -> int. "TG" (threads per threadgroup) is required.
    configs: list[dict[str, int]] = field(default_factory=lambda: [{"TG": 256}])
    plan: str = ""

    def digest(self) -> str:
        blob = json.dumps([self.source, self.header], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_json(self) -> dict:
        return {"source": self.source, "header": self.header, "configs": self.configs, "plan": self.plan}

    @staticmethod
    def from_json(d: dict) -> "Kernel":
        return Kernel(d["source"], d.get("header", ""), d.get("configs") or [{"TG": 256}], d.get("plan", ""))


def static_check(kernel: Kernel) -> str | None:
    """Cheap checks before touching the GPU. Returns an error message or None."""
    text = kernel.source + "\n" + kernel.header
    if not kernel.source.strip():
        return "empty kernel source"
    if len(text) > MAX_SOURCE_CHARS:
        return f"kernel is {len(text)} chars; limit is {MAX_SOURCE_CHARS}"
    for tok in BANNED_TOKENS:
        if tok in text:
            return f"banned token {tok!r} (MLX already includes the Metal standard library)"
    if not 1 <= len(kernel.configs) <= MAX_CONFIGS:
        return f"give between 1 and {MAX_CONFIGS} configs"
    for cfg in kernel.configs:
        tg = cfg.get("TG")
        if not isinstance(tg, int) or tg < 32 or tg > 1024 or tg % 32:
            return f"config {cfg}: TG must be a multiple of 32 between 32 and 1024"
        for k, v in cfg.items():
            if not k.isidentifier() or not isinstance(v, int) or not 0 <= v <= 65536:
                return f"config {cfg}: constants must be identifiers mapped to ints in [0, 65536]"
    return None


class Adapter(Protocol):
    """The only chip-specific surface. Adding a new chip means writing one of these."""

    name: str

    def chip(self) -> str: ...

    def build(self, op, kernel: Kernel):
        """Compile the kernel. Raise on compile errors."""

    def upload(self, inputs: dict[str, np.ndarray]):
        """Move numpy inputs to the device once, so timing excludes transfers."""

    def launch(self, built, op, config: dict, dev_inputs, rows: int, n: int, dtype: str,
               inner: int = 1) -> Callable[[], None]:
        """Return a function that runs the kernel `inner` times back to back and blocks until done.
        For timing only: must not add work the stock op doesn't do (e.g. no NaN pre-fill)."""

    def run(self, built, op, config: dict, dev_inputs, rows: int, n: int, dtype: str) -> np.ndarray:
        """Run once; return the flat output (rows * n + PAD elements), NaN-poisoned before launch."""

    def baseline(self, op, dev_inputs) -> Callable[[], np.ndarray]:
        """Return a function that runs the stock library op, blocks, and returns its output."""

    def baseline_launch(self, op, dev_inputs, inner: int = 1, compiled: bool = False) -> Callable[[], None]:
        """Return a function that runs the stock library op `inner` times and blocks (for timing).
        `compiled=True` wraps it in the library's graph compiler (mx.compile), which fuses element-wise ops."""

    def measure_peak_gbps(self) -> float:
        """Practical memory bandwidth from a copy kernel."""
