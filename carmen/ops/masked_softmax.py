"""softmax(x * scale + mask): the fused attention-score op.

Stock MLX runs this as separate passes (multiply, add, softmax), so the data
crosses memory several times. A fused kernel reads x and mask once and writes
once; that is where a custom kernel can honestly beat the library.
"""

from __future__ import annotations

import numpy as np

from .base import Case, OpSpec
from . import softmax as sm

CONTRACT = """\
Compute y = softmax(x * scale + mask) over each row, where mask is additive (0 or -inf, but any finite value is allowed).

Buffers (row-major, contiguous):
  x     : input,  shape (rows, n), element type T (float, half or bfloat)
  mask  : input,  shape (rows, n), element type T
  scale : input,  shape (1,), float32 scalar
  out   : output, flat, at least rows * n elements of type T; write out[r * n + i]
Row length n = x_shape[1]. Rows = x_shape[0].

Semantics that the judge checks exactly:
  - Compute v = float(x) * scale + float(mask) in float32, then a numerically safe softmax of v.
  - Masked (-inf) positions output exactly 0. A fully masked row produces NaN in every position (IEEE, as in PyTorch).
  - Accumulate in float32 even when T is half or bfloat.
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""


def _with_mask(x, mask, scale=0.125):
    return {"x": x, "mask": mask, "scale": np.array([scale], dtype=np.float32)}


def _random_mask(rng, rows, n, p):
    m = np.where(rng.random((rows, n)) < p, -np.inf, 0.0)
    m[:, rng.integers(n)] = 0.0  # keep one unmasked position per row
    return m


def _normal(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n)).astype(dtype)
    return _with_mask(x, _random_mask(rng, rows, n, 0.2).astype(dtype))


def _uniform01(rng, rows, n, dtype):
    x = rng.random((rows, n)).astype(dtype)
    return _with_mask(x, np.zeros((rows, n), dtype=dtype), 1.0)


def _causal(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n)).astype(dtype)
    cutoff = np.maximum(1, (np.arange(rows)[:, None] + 1) * n // max(rows, 1))
    mask = np.where(np.arange(n)[None, :] < cutoff, 0.0, -np.inf).astype(dtype)
    return _with_mask(x, mask)


def _big(rng, rows, n, dtype):
    x = rng.uniform(-1e4, 1e4, (rows, n)).astype(dtype)
    return _with_mask(x, _random_mask(rng, rows, n, 0.1).astype(dtype), 1.0)


def _fully_masked_row(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n)).astype(dtype)
    mask = _random_mask(rng, rows, n, 0.2)
    mask[0] = -np.inf
    return _with_mask(x, mask.astype(dtype))


def _bias_mask(rng, rows, n, dtype):
    """Finite additive bias (ALiBi-style), no -inf."""
    x = rng.standard_normal((rows, n)).astype(dtype)
    mask = (-0.05 * np.abs(np.arange(n)[None, :] - rng.integers(0, n, (rows, 1)))).astype(dtype)
    return _with_mask(x, mask, 0.3)


def _peaked(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[np.arange(rows), rng.integers(0, n, rows)] += 300.0
    return _with_mask(x.astype(dtype), _random_mask(rng, rows, n, 0.3).astype(dtype))


def reference(inputs):
    x = inputs["x"].astype(np.float64)
    v = x * float(inputs["scale"][0]) + inputs["mask"].astype(np.float64)
    return sm.reference({"x": v})


def mlx_baseline(mx, inputs):
    return mx.softmax(inputs["x"] * inputs["scale"].astype(inputs["x"].dtype) + inputs["mask"], axis=-1)


def invariants(inputs, out, ref):
    fails = sm.invariants(inputs, out, ref)
    masked = np.isneginf(inputs["mask"].astype(np.float64)) & np.isfinite(ref)
    if masked.any() and (out[masked] != 0).any():
        fails.append("masked positions are not exactly 0")
    return fails


VISIBLE = (
    [Case("normal", 5, n, dt, 200 + n, "shape") for n in sm.LENGTHS for dt in ("float32", "float16")]
    + [Case(k, 9, 1537, dt, 17, "values")
       for k in ("causal", "big", "fully_masked_row", "bias_mask")
       for dt in ("float32", "float16")]
    + [Case("normal", 256, 129, "float32", 19, "many_rows")]
    + [Case("normal", 5, n, "bfloat16", 700 + n, "bf16") for n in (1, 33, 257, 4097)]
    + [Case(k, 7, 1537, "bfloat16", 47, "bf16") for k in ('causal', 'fully_masked_row', 'big')]
)

SPEC = OpSpec(
    name="masked_softmax",
    summary="attention: scale, hide future words, softmax",
    input_names=("x", "mask", "scale"),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "causal": _causal, "big": _big,
        "fully_masked_row": _fully_masked_row, "bias_mask": _bias_mask, "peaked": _peaked,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "causal", "big", "bias_mask"),
    hidden_kinds=("peaked", "causal", "bias_mask", "normal"),
    timing_shapes=((4096, 1024), (1024, 4096), (128, 32768)),
    naive=Case("uniform01", 64, 1000, "float32", 42, "naive"),
    about="The attention step that asks which past words matter: scale the scores, hide the words the "
          "model may not look at (the future), then softmax. MLX does this in 3 trips through memory. "
          "One fused kernel does it in 1, and memory trips are the slow part. That is where Carmy wins.",
)
