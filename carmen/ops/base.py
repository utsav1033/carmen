"""The op spec: what an operation *means*, independent of any chip or kernel.

A spec gives the judge everything it needs to decide whether a kernel is right:
a float64 reference, properties any correct output must have, and generators
for the inputs where bugs live. The judge itself contains no op-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

DTYPES = ("float32", "float16")


@dataclass(frozen=True)
class Case:
    """A test input described by a recipe, not by data.

    Cases are regenerated deterministically from (kind, rows, n, dtype, seed),
    so they are cheap to send to the worker process and easy to shrink.
    """

    kind: str
    rows: int
    n: int
    dtype: str
    seed: int
    family: str = ""

    def with_shape(self, rows: int, n: int) -> "Case":
        return Case(self.kind, rows, n, self.dtype, self.seed, self.family)

    def label(self) -> str:
        return f"{self.kind}[{self.rows}x{self.n}, {self.dtype}]"

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_json(d: dict) -> "Case":
        return Case(**d)


@dataclass
class OpSpec:
    """Everything the judge knows about one op. Row-wise ops over a 2D (rows, n) input."""

    name: str
    summary: str
    # Names of the kernel's input buffers, in call order. The output is always `out`.
    input_names: tuple[str, ...]
    # Text for Carmy: the exact math, buffer layout and edge-case semantics.
    contract: str
    # kind -> function(rng, rows, n, dtype) -> dict of numpy inputs
    generators: dict[str, Callable[..., dict[str, np.ndarray]]]
    # float64 answer key, shape (rows, n)
    reference: Callable[[dict[str, np.ndarray]], np.ndarray]
    # The stock implementation users would otherwise call: fn(mx, inputs_on_device) -> mx.array
    mlx_baseline: Callable
    # Extra properties a correct output must have: fn(inputs, out, ref) -> list of failure strings
    invariants: Callable[[dict, np.ndarray, np.ndarray], list[str]]
    # Visible test battery (Carmy sees failures on these)
    visible: list[Case] = field(default_factory=list)
    # Kinds and families used for fresh per-round fuzz and for the hidden sampler
    fuzz_kinds: tuple[str, ...] = ()
    hidden_kinds: tuple[str, ...] = ()
    # (rows, n) shapes used for timing
    timing_shapes: tuple[tuple[int, int], ...] = ()
    # The KernelBench-style check we compare against: one shape, uniform [0,1), allclose 1e-2
    naive: Case | None = None

    def materialize(self, case: Case) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(case.seed)
        return self.generators[case.kind](rng, case.rows, case.n, case.dtype)

    def bytes_moved(self, rows: int, n: int, dtype: str) -> int:
        """Minimum memory traffic: read every input once, write the output once."""
        item = np.dtype(dtype).itemsize
        per_row_inputs = sum(1 for name in self.input_names if name != "scale")
        return rows * n * item * (per_row_inputs + 1)


def off_grid_n(rng: np.random.Generator, lo: int, hi: int) -> int:
    """A row length a model can't guess: not a power of two, not a multiple of 32."""
    while True:
        n = int(rng.integers(lo, hi))
        if n & (n - 1) and n % 32 and (n + 1) & n and (n - 1) & (n - 2):
            return n


def fuzz_cases(spec: OpSpec, seed: int, count: int = 8) -> list[Case]:
    """Fresh visible cases for one round. Seeds are revealed to Carmy only through failures."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        kind = spec.fuzz_kinds[i % len(spec.fuzz_kinds)]
        n = off_grid_n(rng, 2, 6000)
        rows = int(rng.integers(1, 48))
        out.append(Case(kind, rows, n, DTYPES[i % 2], int(rng.integers(2**31)), "fuzz"))
    return out


def hidden_cases(spec: OpSpec, secret_seed: int, count: int = 24) -> list[Case]:
    """Inputs Carmy never sees: off-grid sizes and data statistics drawn fresh each round."""
    rng = np.random.default_rng(secret_seed)
    out = []
    for i in range(count):
        kind = spec.hidden_kinds[i % len(spec.hidden_kinds)]
        n = off_grid_n(rng, 3, 20000)
        rows = int(rng.integers(1, 64))
        out.append(Case(kind, rows, n, DTYPES[i % 2], int(rng.integers(2**31)), "hidden"))
    return out


def hidden_timing_shapes(secret_seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(secret_seed ^ 0x5EED)
    return [(int(rng.integers(300, 3000)), off_grid_n(rng, 700, 3000)),
            (int(rng.integers(64, 600)), off_grid_n(rng, 5000, 30000))]
