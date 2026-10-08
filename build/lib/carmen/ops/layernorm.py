"""layernorm(x) over the last axis, with a learned weight and bias."""

from __future__ import annotations

import numpy as np

from .base import Case, OpSpec
from .softmax import LENGTHS

EPS = 1e-5

CONTRACT = """\
Compute y = (x - mean) / sqrt(var + eps) * w + b over each row, with eps = 1e-5.
mean = sum_i x[r, i] / n and var = sum_i (x[r, i] - mean)^2 / n (biased: divide by n, not n - 1).

Buffers (row-major, contiguous):
  x   : input,  shape (rows, n), element type T (float, half or bfloat)
  w   : input,  shape (n,), element type T (weight, shared by every row)
  b   : input,  shape (n,), element type T (bias, shared by every row)
  out : output, flat, at least rows * n elements of type T; write out[r * n + i]
Row length n = x_shape[1]. Rows = x_shape[0].

Semantics that the judge checks exactly:
  - Accumulate in float32 even when T is half or bfloat.
  - Rows can sit far from zero (a large common offset), so compute the variance from (x - mean),
    not as E[x^2] - mean^2, which cancels catastrophically.
  - A row with zero variance (constant, or n = 1) outputs b.
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""


def _params(rng, n, dtype):
    return {"w": (1 + 0.1 * rng.standard_normal(n)).astype(dtype), "b": (0.1 * rng.standard_normal(n)).astype(dtype)}


def _normal(rng, rows, n, dtype):
    return {"x": rng.standard_normal((rows, n)).astype(dtype), **_params(rng, n, dtype)}


def _uniform01(rng, rows, n, dtype):
    return {"x": rng.random((rows, n)).astype(dtype), **_params(rng, n, dtype)}


def _offset(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n)) + rng.uniform(-50, 50, (rows, 1))
    return {"x": x.astype(dtype), **_params(rng, n, dtype)}


def _large(rng, rows, n, dtype):
    return {"x": (rng.standard_normal((rows, n)) * 300).astype(dtype), **_params(rng, n, dtype)}


def _small(rng, rows, n, dtype):
    """Variance near eps: a kernel that drops or misplaces eps is visibly wrong here."""
    return {"x": (rng.standard_normal((rows, n)) * 1e-3).astype(dtype), **_params(rng, n, dtype)}


def _zero_rows(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[::2] = 0.0
    return {"x": x.astype(dtype), **_params(rng, n, dtype)}


def _wild_params(rng, rows, n, dtype):
    w = rng.standard_normal(n) * 3
    w[rng.random(n) < 0.2] = 0.0
    return {"x": rng.standard_normal((rows, n)).astype(dtype), "w": w.astype(dtype),
            "b": rng.standard_normal(n).astype(dtype)}


def _skewed(rng, rows, n, dtype):
    x = np.exp(rng.standard_normal((rows, n)) * 1.5) * rng.choice([-1.0, 1.0], (rows, n))
    return {"x": x.astype(dtype), **_params(rng, n, dtype)}


def reference(inputs):
    x = inputs["x"].astype(np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * inputs["w"].astype(np.float64) + inputs["b"].astype(np.float64)


def mlx_baseline(mx, inputs):
    return mx.fast.layer_norm(inputs["x"], inputs["w"], inputs["b"], EPS)


def invariants(inputs, out, ref):
    return []


VISIBLE = (
    [Case("normal", 5, n, dt, 300 + n, "shape") for n in LENGTHS for dt in ("float32", "float16")]
    + [Case("normal", 3, 70001, "float32", 7, "shape")]
    + [Case(k, 7, 1537, dt, 23, "values")
       for k in ("offset", "large", "small", "zero_rows", "wild_params")
       for dt in ("float32", "float16")]
    + [Case("normal", 300, 129, "float32", 29, "many_rows")]
    + [Case("normal", 5, n, "bfloat16", 700 + n, "bf16") for n in (1, 33, 257, 4097)]
    + [Case(k, 7, 1537, "bfloat16", 47, "bf16") for k in ('offset', 'small', 'zero_rows')]
)

SPEC = OpSpec(
    name="layernorm",
    summary="normalize each row, then scale and shift",
    input_names=("x", "w", "b"),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "offset": _offset, "large": _large, "small": _small,
        "zero_rows": _zero_rows, "wild_params": _wild_params, "skewed": _skewed,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "offset", "small", "wild_params"),
    hidden_kinds=("skewed", "offset", "large", "normal", "zero_rows"),
    timing_shapes=((4096, 1024), (1024, 4096), (128, 32768)),
    naive=Case("uniform01", 64, 1000, "float32", 42, "naive"),
    streamed=("x",),
    about="Every transformer layer runs this before attention and before the MLP. It recenters each "
          "row of numbers to mean 0 and spread 1, then applies a learned scale and shift, which keeps "
          "the numbers stable as they flow through the model. The trap: rows far from zero, where the "
          "quick formula for variance silently loses all precision. MLX already fuses this into one "
          "kernel, so expect Carmy to tie it, not beat it.",
)
