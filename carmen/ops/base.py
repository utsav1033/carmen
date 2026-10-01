"""The op spec: what an operation *means*, independent of any chip or kernel.

A spec gives the judge everything it needs to decide whether a kernel is right:
a float64 reference, properties any correct output must have, and generators
for the inputs where bugs live. The judge itself contains no op-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

DTYPES = ("float32", "float16", "bfloat16")
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}


def to_bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), kept as float32.

    numpy has no bfloat16, so bf16 inputs live as float32 arrays holding exactly
    representable bf16 values; the adapter casts them on upload, losslessly."""
    a = np.asarray(a, dtype=np.float32)
    b = a.view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    return np.where(np.isnan(a), a, r)


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
    # Third dimension for ops that have one (matmul's K, the length of each dot product). 0 = none.
    inner: int = 0

    def __post_init__(self):
        # The harness never launches on an empty tensor: zero threadgroups would mean the
        # kernel never runs, so there would be nothing to judge.
        if self.rows < 1 or self.n < 1 or self.inner < 0:
            raise ValueError(f"empty case {self.rows}x{self.n}: rows and n must be >= 1")

    def with_shape(self, rows: int, n: int) -> "Case":
        """A smaller case of the same kind. For 3-D ops the inner length shrinks with n."""
        return Case(self.kind, rows, n, self.dtype, self.seed, self.family, n if self.inner else 0)

    def label(self) -> str:
        dims = f"{self.rows}x{self.n}" + (f"x{self.inner}" if self.inner else "")
        return f"{self.kind}[{dims}, {self.dtype}]"

    @property
    def accum_len(self) -> int:
        """How many terms each output sums over: sets the float32 accumulation tolerance."""
        return self.inner or self.n

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
    # Plain-English note for the app: what this is for in a model, and where a custom kernel can win
    about: str = ""
    # Inputs read once per row (shape (rows, n)). None = every input except `scale`.
    # Per-column vectors like layernorm's weight are tiny and stay in cache, so they don't count.
    streamed: tuple[str, ...] | None = None
    # 2 for row-wise ops (rows, n); 3 for ops with an inner dimension (matmul: M, N, K).
    dims: int = 2
    # Launch shape the harness uses: fn(rows, n, config) -> (grid, threadgroup). None = one
    # threadgroup of TG threads per row. Carmy never picks the grid.
    launch: Callable | None = None
    # Template constants every config must define, beyond TG (e.g. matmul's tile BM, BN).
    required_params: tuple[str, ...] = ()
    # Minimum memory traffic and arithmetic for one launch: fn(rows, n, inner, itemsize) -> bytes / flops.
    bytes_fn: Callable | None = None
    flops_fn: Callable | None = None

    def grid(self, rows: int, n: int, config: dict) -> tuple[tuple, tuple]:
        if self.launch:
            return self.launch(rows, n, config)
        return (rows * config["TG"], 1, 1), (config["TG"], 1, 1)

    def check_config(self, config: dict) -> str | None:
        missing = [k for k in self.required_params if k not in config]
        if missing:
            return f"config {config}: {self.name} needs {', '.join(missing)}"
        bad = [k for k in self.required_params if config[k] < 1]
        return f"config {config}: {', '.join(bad)} must be at least 1" if bad else None

    def materialize(self, case: Case) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(case.seed)
        gen = self.generators[case.kind]
        extra = {"inner": case.inner} if self.dims == 3 else {}
        if case.dtype != "bfloat16":
            return gen(rng, case.rows, case.n, case.dtype, **extra)
        d = gen(rng, case.rows, case.n, "float32", **extra)
        return {k: v if k == "scale" else to_bf16(v) for k, v in d.items()}

    def bytes_moved(self, rows: int, n: int, dtype: str, inner: int = 0) -> int:
        """Minimum memory traffic: read every input once, write the output once."""
        item = ITEMSIZE[dtype]
        if self.bytes_fn:
            return self.bytes_fn(rows, n, inner, item)
        streamed = self.streamed if self.streamed is not None else [k for k in self.input_names if k != "scale"]
        per_row_inputs = len(streamed)
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
        if spec.dims == 3:
            rows, n, inner = int(rng.integers(1, 160)), off_grid_n(rng, 2, 700), off_grid_n(rng, 2, 700)
        else:
            rows, n, inner = int(rng.integers(1, 48)), off_grid_n(rng, 2, 6000), 0
        out.append(Case(kind, rows, n, DTYPES[i % len(DTYPES)], int(rng.integers(2**31)), "fuzz", inner))
    return out


def hidden_cases(spec: OpSpec, secret_seed: int, count: int = 24) -> list[Case]:
    """Inputs Carmy never sees: off-grid sizes and data statistics drawn fresh each round."""
    rng = np.random.default_rng(secret_seed)
    out = []
    for i in range(count):
        kind = spec.hidden_kinds[i % len(spec.hidden_kinds)]
        if spec.dims == 3:
            rows, n, inner = int(rng.integers(1, 256)), off_grid_n(rng, 3, 1200), off_grid_n(rng, 3, 1200)
        else:
            rows, n, inner = int(rng.integers(1, 64)), off_grid_n(rng, 3, 20000), 0
        out.append(Case(kind, rows, n, DTYPES[i % len(DTYPES)], int(rng.integers(2**31)), "hidden", inner))
    return out


def hidden_timing_shapes(secret_seed: int, dims: int = 2) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(secret_seed ^ 0x5EED)
    if dims == 3:
        return [(int(rng.integers(300, 1500)), off_grid_n(rng, 300, 1500), off_grid_n(rng, 300, 1500)),
                (int(rng.integers(8, 64)), off_grid_n(rng, 2000, 4500), off_grid_n(rng, 2000, 4500))]
    return [(int(rng.integers(300, 3000)), off_grid_n(rng, 700, 3000)),
            (int(rng.integers(64, 600)), off_grid_n(rng, 5000, 30000))]
