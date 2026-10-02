"""residual_rmsnorm: h = x + res and y = rmsnorm(h) * w, both written.

This is the version a real model can use. Every Llama/Qwen-style layer adds a block's
output onto the residual stream and normalizes the sum for the next block, and needs
both: h is the new residual, y feeds the next matmul. Stock MLX runs the add and the
norm as two kernels (5 trips through memory); one fused kernel makes 4. eps is an
input because models differ (Qwen 1e-6, Llama 1e-5).
"""

from __future__ import annotations

import numpy as np

from . import add_rmsnorm as ar
from . import rmsnorm as rn
from .base import Case, OpSpec
from .softmax import LENGTHS

CONTRACT = """\
Compute, for each row: h = x + res, then y = h / sqrt(mean(h^2) + eps) * w, and write BOTH.

Buffers (row-major, contiguous):
  x   : input,  shape (rows, n), element type T (float, half or bfloat)
  res : input,  shape (rows, n), element type T
  w   : input,  shape (n,), element type T (weight, shared by every row)
  eps : input,  shape (1,), float32 (e.g. 1e-6 or 1e-5; read eps[0])
  out : output, flat, 2 * rows * n elements of type T, two blocks back to back:
        out[r * n + i]          = h[r, i]   (the new residual)
        out[(rows + r) * n + i] = y[r, i]   (the normalized output)
Row length n = x_shape[1]. Rows = x_shape[0].

Semantics that the judge checks exactly:
  - Compute h = float(x) + float(res) in float32; square and accumulate in float32 even when T is half or bfloat.
  - A row where h is all zero gives y = 0.
  - Every one of the 2 * rows * n outputs must be written. Nothing past 2 * rows * n may be written.
"""

EPS_CHOICES = (1e-5, 1e-6)


def _with_eps(gen):
    def g(rng, rows, n, dtype):
        d = gen(rng, rows, n, dtype)
        d["eps"] = np.array([EPS_CHOICES[int(rng.integers(len(EPS_CHOICES)))]], dtype=np.float32)
        return d
    return g


def _tiny_rows(rng, rows, n, dtype):
    """Values ~1e-3: mean(h^2) ~1e-6, the same size as eps, so a wrong or missing eps shows."""
    return ar._pair(rng, rows, n, dtype, sx=1e-3, sr=1e-3)


GENERATORS = {k: _with_eps(v) for k, v in {**ar.SPEC.generators, "tiny_rows": _tiny_rows}.items()}


def reference(inputs):
    h = inputs["x"].astype(np.float64) + inputs["res"].astype(np.float64)
    eps = float(inputs["eps"][0])
    y = h / np.sqrt((h ** 2).mean(axis=-1, keepdims=True) + eps) * inputs["w"].astype(np.float64)
    return np.concatenate([h, y])


def mlx_baseline(mx, inputs):
    eps = inputs["eps"]
    eps = eps if isinstance(eps, float) else float(eps.item())  # a constant when compiled
    h = inputs["x"] + inputs["res"]
    return h, mx.fast.rms_norm(h, inputs["w"], eps)


def invariants(inputs, out, ref):
    return []


# Model shapes first: Qwen2.5-0.5B (896) and Llama-3.2-1B (2048) hidden sizes, 1 token (decode) and a prompt.
VISIBLE = (
    [Case("normal", 5, n, dt, 1200 + n, "shape") for n in LENGTHS for dt in ("float32", "float16")]
    + [Case("normal", r, n, dt, 1300 + r, "model", ) for r, n in ((1, 896), (512, 896), (1, 2048), (64, 2048))
       for dt in ("float16", "bfloat16")]
    + [Case(k, 7, 1537, dt, 73, "values")
       for k in ("big_residual", "cancel", "zero_sum_rows", "tiny_rows", "outlier")
       for dt in ("float32", "float16")]
    + [Case("normal", 300, 129, "float32", 79, "many_rows")]
    + [Case(k, 7, 896, "bfloat16", 83, "bf16") for k in ("cancel", "big_residual", "tiny_rows")]
)

SPEC = OpSpec(
    name="residual_rmsnorm",
    summary="residual add + rmsnorm, both outputs (drop-in for a model)",
    input_names=("x", "res", "w", "eps"),
    contract=CONTRACT,
    generators=GENERATORS,
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "cancel", "tiny_rows", "outlier"),
    hidden_kinds=("cancel", "outlier", "big_residual", "normal", "tiny_rows"),
    # Qwen2.5-0.5B decode and prefill shapes, then a bigger batch.
    timing_shapes=((1, 896), (512, 896), (4096, 2048)),
    naive=Case("uniform01", 64, 1000, "float32", 42, "naive"),
    bytes_fn=lambda rows, n, inner, item: rows * n * item * 4,  # read x, res; write h, y
    outputs=2,
    about="The residual add and the norm that every Llama- or Qwen-style layer runs twice, written as "
          "one kernel that returns both results, so it can be dropped straight into a real model. "
          "MLX runs it as 2 kernels; at one word at a time, each kernel costs ~10-25 us of pure "
          "overhead, so this is about running fewer kernels, not faster math.",
)
