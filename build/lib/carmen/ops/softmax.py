"""softmax(x) over the last axis."""

from __future__ import annotations

import numpy as np

from .base import Case, OpSpec

CONTRACT = """\
Compute y = softmax(x) over each row: y[r, i] = exp(x[r, i] - max_r) / sum_j exp(x[r, j] - max_r).

Buffers (row-major, contiguous):
  x   : input,  shape (rows, n), element type T (float, half or bfloat)
  out : output, flat, at least rows * n elements of type T; write out[r * n + i]
Row length n = x_shape[1]. Rows = x_shape[0].

Semantics that the judge checks exactly:
  - -inf entries contribute 0. A row that is entirely -inf produces NaN in every position (IEEE, as in PyTorch).
  - Inputs can be as large as +/-1e4 (so exp overflows unless you subtract the row max first).
  - Accumulate in float32 even when T is half or bfloat.
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""


def _normal(rng, rows, n, dtype):
    return {"x": rng.standard_normal((rows, n)).astype(dtype)}


def _uniform01(rng, rows, n, dtype):
    return {"x": rng.random((rows, n)).astype(dtype)}


def _big(rng, rows, n, dtype):
    return {"x": rng.uniform(-1e4, 1e4, (rows, n)).astype(dtype)}


def _very_negative(rng, rows, n, dtype):
    return {"x": (rng.standard_normal((rows, n)) * 3 - 1e4).astype(dtype)}


def _constant(rng, rows, n, dtype):
    return {"x": np.full((rows, n), 3.7, dtype=dtype)}


def _neg_inf_mask(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[rng.random((rows, n)) < 0.3] = -np.inf
    x[:, rng.integers(n)] = rng.standard_normal(rows)  # keep one finite value per row
    return {"x": x.astype(dtype)}


def _all_neg_inf_row(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[0] = -np.inf
    return {"x": x.astype(dtype)}


def _spike_last(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[:, -1] += 20.0
    return {"x": x.astype(dtype)}


def _peaked(rng, rows, n, dtype):
    x = rng.standard_normal((rows, n))
    x[np.arange(rows), rng.integers(0, n, rows)] += 40.0
    return {"x": x.astype(dtype)}


def _skewed(rng, rows, n, dtype):
    mag = np.exp(rng.standard_normal((rows, n)) * 1.5)
    return {"x": (mag * rng.choice([-1.0, 1.0], (rows, n))).astype(dtype)}


def _offset(rng, rows, n, dtype):
    return {"x": (rng.standard_normal((rows, n)) + rng.uniform(-3e3, 3e3, (rows, 1))).astype(dtype)}


def reference(inputs):
    x = inputs["x"].astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=-1, keepdims=True)


def mlx_baseline(mx, inputs):
    return mx.softmax(inputs["x"], axis=-1)


def invariants(inputs, out, ref):
    fails = []
    finite = np.isfinite(ref).all(axis=-1)
    if not finite.any():
        return fails
    o = out[finite]
    if (o < 0).any():
        fails.append("negative probabilities")
    sums = o.sum(axis=-1)
    if (sums < 0.5).any():
        fails.append(f"row sums far below 1 (min {sums.min():.3g}); output looks empty")
    return fails


LENGTHS = (1, 2, 3, 7, 31, 33, 255, 257, 1025, 4097)

VISIBLE = (
    [Case("normal", 5, n, dt, 100 + n, "shape") for n in LENGTHS for dt in ("float32", "float16")]
    + [Case("normal", 3, 70001, "float32", 7, "shape")]
    + [Case(k, 7, 1537, dt, 11, "values")
       for k in ("big", "very_negative", "constant", "neg_inf_mask", "all_neg_inf_row", "spike_last")
       for dt in ("float32", "float16")]
    + [Case("normal", 300, 129, "float32", 13, "many_rows")]
    + [Case("normal", 5, n, "bfloat16", 700 + n, "bf16") for n in (1, 33, 257, 4097)]
    + [Case(k, 7, 1537, "bfloat16", 47, "bf16") for k in ('big', 'neg_inf_mask', 'all_neg_inf_row')]
)

SPEC = OpSpec(
    name="softmax",
    summary="raw scores → percentages that add up to 100%",
    input_names=("x",),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "big": _big, "very_negative": _very_negative,
        "constant": _constant, "neg_inf_mask": _neg_inf_mask, "all_neg_inf_row": _all_neg_inf_row,
        "spike_last": _spike_last, "peaked": _peaked, "skewed": _skewed, "offset": _offset,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "big", "neg_inf_mask", "spike_last"),
    hidden_kinds=("peaked", "skewed", "offset", "neg_inf_mask", "very_negative", "normal"),
    timing_shapes=((4096, 1024), (1024, 4096), (128, 32768)),
    naive=Case("uniform01", 64, 1000, "float32", 42, "naive"),
    about="Turns a row of raw scores into percentages that add up to 100%. Inside attention, it decides "
          "how much each past word matters to the next one. The trap: big scores overflow exp() unless "
          "you subtract the row's max first. MLX's version is already hand-tuned, so matching it is the "
          "honest bar.",
)
