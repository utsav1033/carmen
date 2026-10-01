"""rmsnorm(x + res): the residual add fused into the norm, as in every Llama-style layer."""

from __future__ import annotations

import numpy as np

from .base import Case, OpSpec
from .softmax import LENGTHS
from . import rmsnorm as rn

EPS = rn.EPS

CONTRACT = """\
Compute h = x + res, then y = h / sqrt(mean(h^2) + eps) * w over each row, with eps = 1e-5.
Only y is written (this version does not write h back).

Buffers (row-major, contiguous):
  x   : input,  shape (rows, n), element type T (float, half or bfloat)
  res : input,  shape (rows, n), element type T (the residual stream)
  w   : input,  shape (n,), element type T (weight, shared by every row)
  out : output, flat, at least rows * n elements of type T; write out[r * n + i]
Row length n = x_shape[1]. Rows = x_shape[0].

Semantics that the judge checks exactly:
  - Compute h = float(x) + float(res) in float32; square and accumulate in float32 even when T is half or bfloat.
  - x and res can be large with opposite signs, so h can be much smaller than either.
  - A row where h is all zero outputs 0.
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""


def _pair(rng, rows, n, dtype, sx=1.0, sr=1.0):
    return {"x": (rng.standard_normal((rows, n)) * sx).astype(dtype),
            "res": (rng.standard_normal((rows, n)) * sr).astype(dtype), "w": rn._w(rng, n, dtype)}


def _normal(rng, rows, n, dtype):
    return _pair(rng, rows, n, dtype)


def _uniform01(rng, rows, n, dtype):
    return {"x": rng.random((rows, n)).astype(dtype), "res": rng.random((rows, n)).astype(dtype),
            "w": rn._w(rng, n, dtype)}


def _big_residual(rng, rows, n, dtype):
    """Residual streams grow through a model: res dominates x."""
    return _pair(rng, rows, n, dtype, sx=1.0, sr=100.0)


def _cancel(rng, rows, n, dtype):
    """x close to -res, so h is small: a kernel that ignores x or res is far off."""
    res = rng.standard_normal((rows, n)) * 50
    x = -res + rng.standard_normal((rows, n))
    return {"x": x.astype(dtype), "res": res.astype(dtype), "w": rn._w(rng, n, dtype)}


def _zero_sum_rows(rng, rows, n, dtype):
    d = _pair(rng, rows, n, dtype)
    d["x"][::2] = 0
    d["res"][::2] = 0
    return d


def _small(rng, rows, n, dtype):
    return _pair(rng, rows, n, dtype, sx=1e-3, sr=1e-3)


def _outlier(rng, rows, n, dtype):
    d = _pair(rng, rows, n, dtype)
    d["res"][np.arange(rows), rng.integers(0, n, rows)] = 200.0
    return d


def reference(inputs):
    h = inputs["x"].astype(np.float64) + inputs["res"].astype(np.float64)
    return rn.reference({"x": h, "w": inputs["w"]})


def mlx_baseline(mx, inputs):
    return mx.fast.rms_norm(inputs["x"] + inputs["res"], inputs["w"], EPS)


def invariants(inputs, out, ref):
    return []


VISIBLE = (
    [Case("normal", 5, n, dt, 500 + n, "shape") for n in LENGTHS for dt in ("float32", "float16")]
    + [Case("normal", 3, 70001, "float32", 7, "shape")]
    + [Case(k, 7, 1537, dt, 41, "values")
       for k in ("big_residual", "cancel", "zero_sum_rows", "small", "outlier")
       for dt in ("float32", "float16")]
    + [Case("normal", 300, 129, "float32", 43, "many_rows")]
    + [Case("normal", 5, n, "bfloat16", 700 + n, "bf16") for n in (1, 33, 257, 4097)]
    + [Case(k, 7, 1537, "bfloat16", 47, "bf16") for k in ('cancel', 'big_residual', 'outlier')]
)

SPEC = OpSpec(
    name="add_rmsnorm",
    summary="residual add + rmsnorm, fused (every Llama layer)",
    input_names=("x", "res", "w"),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "big_residual": _big_residual, "cancel": _cancel,
        "zero_sum_rows": _zero_sum_rows, "small": _small, "outlier": _outlier,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "big_residual", "cancel", "outlier"),
    hidden_kinds=("cancel", "outlier", "big_residual", "normal", "zero_sum_rows"),
    timing_shapes=((4096, 1024), (1024, 4096), (128, 32768)),
    naive=Case("uniform01", 64, 1000, "float32", 42, "naive"),
    streamed=("x", "res"),
    about="Every layer of a Llama-style model adds its output back onto the residual stream, then "
          "normalizes it for the next layer. MLX runs that as 2 kernels: the add writes the sum to "
          "memory, then rmsnorm reads it back. One fused kernel never writes the sum, so it moves "
          "3 tables of data instead of 5. This version writes only the normalized output.",
)
