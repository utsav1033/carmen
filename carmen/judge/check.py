"""Compare one output against the float64 answer key, and say *where* it is wrong.

Tolerance rule, fixed in advance (not tuned per kernel):

    allowed = min( max(K * baseline_error, floor), CAP * floor )
    floor   = 2 * eps(output dtype) + C * eps(float32) * sqrt(n)

`floor` is output rounding plus float32 accumulation over n terms. Errors are
measured relative to each row's largest reference value, because softmax outputs
are ~1/n and an absolute tolerance would let an all-zeros kernel pass (a known
KernelBench hole). The stock library's own error on the same input can loosen the
bar up to CAP * floor, never further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = {"float32": 2.0**-23, "float16": 2.0**-10, "bfloat16": 2.0**-7}
K_BASELINE = 4.0
C_ACCUM = 8.0
CAP = 16.0


def floor_tol(dtype: str, n: int) -> float:
    return 2 * EPS[dtype] + C_ACCUM * EPS["float32"] * math.sqrt(n)


def tolerance(dtype: str, n: int, baseline_err: float) -> float:
    f = floor_tol(dtype, n)
    return min(max(K_BASELINE * baseline_err, f), CAP * f)


@dataclass
class Comparison:
    ok: bool
    patterns: list[str] = field(default_factory=list)
    where: str = ""
    max_err: float = 0.0
    tol: float = 0.0
    first_bad: tuple | None = None

    def to_json(self) -> dict:
        return {"ok": self.ok, "patterns": self.patterns, "where": self.where,
                "max_err": _num(self.max_err), "tol": _num(self.tol), "first_bad": self.first_bad}


def _num(x: float) -> float | str:
    return x if math.isfinite(x) else str(x)


def row_relative_error(out: np.ndarray, ref: np.ndarray) -> float:
    """Largest |out - ref| / max|ref row|, over elements where both are finite."""
    both = np.isfinite(ref) & np.isfinite(out)
    if not both.any():
        return 0.0
    scale = np.where(np.isfinite(ref), np.abs(ref), 0.0).max(axis=-1, keepdims=True)
    rel = np.where(both, np.abs(out - ref) / np.maximum(scale, 1e-30), 0.0)
    return float(rel.max())


def compare(out_flat: np.ndarray, ref: np.ndarray, dtype: str, tol: float, pad: int = 0) -> Comparison:
    rows, n = ref.shape
    out_flat = np.asarray(out_flat)
    patterns = []
    if pad:
        tail = out_flat[rows * n: rows * n + pad].astype(np.float64)
        if not np.isnan(tail).all():
            patterns.append("wrote past the end of the output")
    out = out_flat[: rows * n].reshape(rows, n).astype(np.float64)

    ref_nan, out_nan = np.isnan(ref), np.isnan(out)
    ref_inf, out_inf = np.isinf(ref), np.isinf(out)
    nan_bad = out_nan & ~ref_nan
    nan_missing = ref_nan & ~out_nan
    inf_bad = ~out_nan & ~ref_nan & ((out_inf != ref_inf) | (ref_inf & (np.sign(out) != np.sign(ref))))

    scale = np.where(np.isfinite(ref), np.abs(ref), 0.0).max(axis=-1, keepdims=True)
    both = np.isfinite(ref) & np.isfinite(out)
    rel = np.where(both, np.abs(out - ref) / np.maximum(scale, 1e-30), 0.0)
    val_bad = rel > tol

    if nan_bad.any():
        patterns.append("NaN where a number was expected (unwritten output, or overflow)")
    if nan_missing.any():
        patterns.append("a number where NaN was expected (e.g. a fully masked / all -inf row)")
    if inf_bad.any():
        patterns.append("infinity mismatch")
    if val_bad.any():
        patterns.append("values outside tolerance")

    bad = nan_bad | nan_missing | inf_bad | val_bad
    max_err = float(rel.max()) if both.any() else 0.0
    if nan_bad.any() or inf_bad.any():
        max_err = math.inf
    if not patterns:
        return Comparison(True, max_err=max_err, tol=tol)
    first = None
    if bad.any():
        r, c = map(int, np.argwhere(bad)[0])
        first = (r, c, _num(float(out[r, c])), _num(float(ref[r, c])))
    return Comparison(False, patterns, locate(bad), max_err, tol, first)


def locate(bad: np.ndarray) -> str:
    """Turn a mask of wrong elements into a sentence a model can act on."""
    if not bad.any():
        return "outside the output buffer"
    rows, n = bad.shape
    bad_rows = bad.any(axis=1)
    cols = np.nonzero(bad)[1]
    tail_width = max(1, min(64, n // 4))
    if n > 8 and cols.min() >= n - tail_width:
        return f"only the last elements of rows (columns >= {int(cols.min())} of {n}): tail handling"
    if bad[bad_rows].all(axis=1).all():
        k = int(bad_rows.sum())
        if k == rows:
            return "every element of every row"
        return f"entire rows ({k} of {rows}) are wrong while the others are right"
    count = int(bad.sum())
    share = f"{count} element{'s' if count > 1 else ''}" if bad.mean() < 0.001 else f"{bad.mean():.1%} of elements"
    return f"{share}, scattered across {int(bad_rows.sum())} of {rows} rows"


NAIVE_LOOSE = 1e-2   # KernelBench v0/v0.1 default, and its current fp16/bf16 default
NAIVE_STRICT = 1e-4  # KernelBench's current fp32 default


def naive_check(out_flat: np.ndarray, ref: np.ndarray, tol: float = NAIVE_LOOSE) -> bool:
    """What a KernelBench-style harness does: one shape, allclose with atol = rtol = tol."""
    rows, n = ref.shape
    out = np.asarray(out_flat)[: rows * n].reshape(rows, n).astype(np.float64)
    return bool(np.allclose(out, ref, atol=tol, rtol=tol))
