"""matmul: C = A @ B. The first non-row-wise op: a 2-D grid of output tiles and an inner
dimension K, with its own class of bugs (tile edges in two directions, the K tail,
leading dimensions that only look right on square inputs)."""

from __future__ import annotations

import math

import numpy as np

from .base import Case, OpSpec

CONTRACT = """\
Compute C = A @ B: out[r, c] = sum_k A[r, k] * B[k, c].

Buffers (row-major, contiguous):
  a   : input,  shape (M, K), element type T (float, half or bfloat)
  b   : input,  shape (K, N), element type T
  out : output, flat, at least M * N elements of type T; write out[r * N + c]
M = a_shape[0], K = a_shape[1], N = b_shape[1].

How the harness launches it (different from the row-wise ops):
  - Each config must define template constants BM and BN: the output tile one threadgroup computes.
  - grid = (ceil(N / BN) * TG, ceil(M / BM), 1), threadgroup = (TG, 1, 1).
    threadgroup_position_in_grid.x is the tile's column index, .y its row index.
  - Tiles on the right and bottom edges are partial: M and N need not be multiples of BM, BN,
    and K need not be a multiple of anything.

Semantics that the judge checks exactly:
  - Accumulate in float32 even when T is half or bfloat.
  - Every one of the M * N outputs must be written. Nothing past M * N may be written.
"""


def _ab(a, b, dtype):
    return {"a": a.astype(dtype), "b": b.astype(dtype)}


def _normal(rng, rows, n, dtype, inner):
    return _ab(rng.standard_normal((rows, inner)), rng.standard_normal((inner, n)), dtype)


def _uniform01(rng, rows, n, dtype, inner):
    return _ab(rng.random((rows, inner)), rng.random((inner, n)), dtype)


def _large(rng, rows, n, dtype, inner):
    return _ab(rng.standard_normal((rows, inner)) * 30, rng.standard_normal((inner, n)) * 30, dtype)


def _ones(rng, rows, n, dtype, inner):
    """Every output is exactly K: a dropped or doubled term is visible in every element."""
    return _ab(np.ones((rows, inner)), np.ones((inner, n)), dtype)


def _sparse(rng, rows, n, dtype, inner):
    a = rng.standard_normal((rows, inner)) * (rng.random((rows, inner)) < 0.3)
    b = rng.standard_normal((inner, n)) * (rng.random((inner, n)) < 0.3)
    return _ab(a, b, dtype)


def _skewed(rng, rows, n, dtype, inner):
    def m(shape):
        return np.exp(rng.standard_normal(shape)) * rng.choice([-1.0, 1.0], shape)
    return _ab(m((rows, inner)), m((inner, n)), dtype)


def _ramp(rng, rows, n, dtype, inner):
    """A depends on its row, B on its column: swapped or transposed indexing changes every value."""
    a = np.arange(rows)[:, None] * 0.01 + rng.standard_normal((rows, inner)) * 0.1
    b = np.arange(n)[None, :] * 0.01 + rng.standard_normal((inner, n)) * 0.1
    return _ab(a, b, dtype)


def reference(inputs):
    return inputs["a"].astype(np.float64) @ inputs["b"].astype(np.float64)


def mlx_baseline(mx, inputs):
    return mx.matmul(inputs["a"], inputs["b"])


def invariants(inputs, out, ref):
    return []


def launch(rows, n, config):
    tg = config["TG"]
    return (math.ceil(n / config["BN"]) * tg, math.ceil(rows / config["BM"]), 1), (tg, 1, 1)


SHAPES = [(1, 1, 1), (1, 7, 3), (7, 1, 5), (33, 31, 17), (31, 33, 65), (64, 64, 64), (65, 63, 129),
          (127, 129, 33), (257, 255, 31), (5, 513, 257), (40, 24, 1025)]

VISIBLE = (
    [Case("normal", m, n, dt, 800 + m + n + k, "shape", k) for m, n, k in SHAPES for dt in ("float32", "float16")]
    + [Case("normal", 3, 70, "float32", 9, "shape", 4097)]
    + [Case(kind, 65, 67, dt, 53, "values", 129)
       for kind in ("large", "ones", "sparse", "skewed", "ramp") for dt in ("float32", "float16")]
    + [Case("normal", m, n, "bfloat16", 900 + m, "bf16", k) for m, n, k in ((1, 7, 3), (33, 31, 17), (65, 63, 129))]
    + [Case(kind, 65, 67, "bfloat16", 59, "bf16", 129) for kind in ("ones", "ramp")]
)

SPEC = OpSpec(
    name="matmul",
    summary="C = A @ B, tiled (the core of every model)",
    input_names=("a", "b"),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "large": _large, "ones": _ones,
        "sparse": _sparse, "skewed": _skewed, "ramp": _ramp,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "ramp", "sparse", "ones"),
    hidden_kinds=("skewed", "ramp", "normal", "large", "sparse"),
    timing_shapes=((1024, 1024, 1024), (2048, 2048, 512), (32, 4096, 4096)),
    # KernelBench-style: one square shape. Square inputs hide leading-dimension bugs.
    naive=Case("uniform01", 256, 256, "float32", 42, "naive", 256),
    dims=3,
    launch=launch,
    required_params=("BM", "BN"),
    bytes_fn=lambda m, n, k, item: (m * k + k * n + m * n) * item,
    flops_fn=lambda m, n, k: 2 * m * n * k,
    about="The operation at the heart of every model: most of an LLM's time is spent multiplying "
          "matrices. Unlike the row-wise kernels, it's limited by arithmetic, not memory, and the "
          "bugs are different: partial tiles on two edges, a dropped last term of the dot product, "
          "indexing that only looks right on square matrices. MLX's matmul is one of its most tuned "
          "kernels, so even matching it would be a strong result.",
)
