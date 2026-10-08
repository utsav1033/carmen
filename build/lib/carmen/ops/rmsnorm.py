"""rmsnorm(x) over the last axis, with a learned weight (Llama-style)."""

from __future__ import annotations

import numpy as np

from .base import Case, OpSpec
from .softmax import LENGTHS

EPS = 1e-5

CONTRACT = """\
Compute y = x / sqrt(mean(x^2) + eps) * w over each row, with eps = 1e-5 and mean(x^2) = sum_i x[r, i]^2 / n.

Buffers (row-major, contiguous):
  x   : input,  shape (rows, n), element type T (float, half or bfloat)
  w   : input,  shape (n,), element type T (weight, shared by every row)
  out : output, flat, at least rows * n elements of type T; write out[r * n + i]
Row length n = x_shape[1]. Rows = x_shape[0].

Semantics that the judge checks exactly:
  - Square and accumulate in float32 even when T is half or bfloat (x^2 overflows half for |x| > 256).
  - An all-zero row outputs 0.
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""


def _w(rng, n, dtype):
    return (1 + 0.1 * rng.standard_normal(n)).astype(dtype)


def _normal(rng, rows, n, dtype):
    return {"x": rng.standard_normal((rows, n)).astype(dtype), "w": _w(rng, n, dtype)}


def _uniform01(rng, rows, n, dtype):
    return {"x": rng.random((rows, n)).astype(dtype), "w": _w(rng, n, dtype)}


def _large(rng, rows, n, dtype):
    return {"x": (rng.standard_normal((rows, n)) * 300).astype(dtype), "w": _w(rng, n, dtype)}


def _small(rng, rows, n, dtype):
    return {"x": (rng.standard_normal((rows, n)) * 1e-3).astype(dtype), "w": _w(rng, n, dtype)}


def _zero_rows(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[::2] = 0.0
    return {"x": x.astype(dtype), "w": _w(rng, n, dtype)}


def _wild_params(rng, rows, n, dtype):
    w = rng.standard_normal(n) * 3
    w[rng.random(n) < 0.2] = 0.0
    return {"x": rng.standard_normal((rows, n)).astype(dtype), "w": w.astype(dtype)}


def _outlier(rng, rows, n, dtype):
    """One huge activation per row, as in real LLM hidden states."""
    x = rng.standard_normal((rows, n))
    x[np.arange(rows), rng.integers(0, n, rows)] = 200.0
    return {"x": x.astype(dtype), "w": _w(rng, n, dtype)}


def reference(inputs):
    x = inputs["x"].astype(np.float64)
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + EPS) * inputs["w"].astype(np.float64)


def mlx_baseline(mx, inputs):
    return mx.fast.rms_norm(inputs["x"], inputs["w"], EPS)


def invariants(inputs, out, ref):
    return []


VISIBLE = (
    [Case("normal", 5, n, dt, 400 + n, "shape") for n in LENGTHS for dt in ("float32", "float16")]
    + [Case("normal", 3, 70001, "float32", 7, "shape")]
    + [Case(k, 7, 1537, dt, 31, "values")
       for k in ("large", "small", "zero_rows", "wild_params", "outlier")
       for dt in ("float32", "float16")]
    + [Case("normal", 300, 129, "float32", 37, "many_rows")]
    + [Case("normal", 5, n, "bfloat16", 700 + n, "bf16") for n in (1, 33, 257, 4097)]
    + [Case(k, 7, 1537, "bfloat16", 47, "bf16") for k in ('large', 'outlier', 'small')]
)

SPEC = OpSpec(
    name="rmsnorm",
    summary="Llama-style normalize: divide each row by its size",
    input_names=("x", "w"),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "large": _large, "small": _small,
        "zero_rows": _zero_rows, "wild_params": _wild_params, "outlier": _outlier,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "large", "small", "outlier"),
    hidden_kinds=("outlier", "large", "wild_params", "normal", "zero_rows"),
    timing_shapes=((4096, 1024), (1024, 4096), (128, 32768)),
    naive=Case("uniform01", 64, 1000, "float32", 42, "naive"),
    streamed=("x",),
    about="The cheaper cousin of layernorm, used by Llama, Mistral and most new models. It divides "
          "each row by its average size, then applies a learned scale. The trap: in half precision, "
          "squaring a big activation overflows to infinity unless the sum is kept in float32. MLX "
          "already fuses this, so a tie is the honest goal.",
)
