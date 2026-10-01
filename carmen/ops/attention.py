"""Fused causal attention, one head: out = softmax(q k^T / sqrt(D) + causal) v.

The hardest kernel in carmen: two matmuls and a softmax in one pass, without
writing the Lq x Lk score matrix to memory. Where the bugs live: the causal
boundary (off by one, and the offset when there are more keys than queries,
as with a KV cache), the online-softmax rescale, the 1/sqrt(D) scale, and
head sizes that aren't a multiple of the tile.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Case, OpSpec, off_grid_n

CONTRACT = """\
Compute causal attention for one head: out[i] = sum_j p[i, j] * v[j], where
  s[i, j] = dot(q[i], k[j]) / sqrt(D)
  p[i, :] = softmax over the visible keys j = 0 .. i + (Lk - Lq)   (keys after that are masked out)
The offset Lk - Lq >= 0 means the last query sees every key (as with a KV cache).

Buffers (row-major, contiguous):
  q   : input,  shape (Lq, D), element type T (float, half or bfloat)
  k   : input,  shape (Lk, D), element type T
  v   : input,  shape (Lk, D), element type T
  out : output, flat, at least Lq * D elements of type T; write out[i * D + d]
Lq = q_shape[0], D = q_shape[1], Lk = k_shape[0]. Lk >= Lq. D is any size from 1 up (often 64, 80, 128).

How the harness launches it:
  - Each config must define template constant BQ: the number of query rows one threadgroup handles.
  - grid = (ceil(Lq / BQ) * TG, 1, 1), threadgroup = (TG, 1, 1).
    threadgroup_position_in_grid.x is the index of the block of BQ queries.

Semantics that the judge checks exactly:
  - Accumulate scores, the softmax and the output in float32 even when T is half or bfloat.
  - Scores can be large (|s| in the hundreds): use a numerically safe softmax.
  - Every one of the Lq * D outputs must be written. Nothing past Lq * D may be written.
"""


def _qkv(rng, lq, d, lk, dtype, qk_scale=1.0, v=None):
    q = rng.standard_normal((lq, d)) * qk_scale
    k = rng.standard_normal((lk, d)) * qk_scale
    v = rng.standard_normal((lk, d)) if v is None else v
    out = {"q": q.astype(dtype), "k": k.astype(dtype), "v": v.astype(dtype)}
    if lq != lk:
        # The stock op's built-in causal mask is only used when Lq == Lk; for Lq < Lk the baseline
        # gets this explicit mask, so its semantics match the contract exactly. Never passed to the kernel.
        visible = np.arange(lk)[None, :] <= np.arange(lq)[:, None] + (lk - lq)
        out["causal_mask"] = np.where(visible, 0.0, -np.inf).astype(dtype)
    return out


def _normal(rng, rows, n, dtype, inner):
    return _qkv(rng, rows, n, inner, dtype)


def _uniform01(rng, rows, n, dtype, inner):
    return {k: v for k, v in _qkv(rng, rows, n, inner, dtype).items()} | {
        "q": rng.random((rows, n)).astype(dtype), "k": rng.random((inner, n)).astype(dtype)}


def _large_scores(rng, rows, n, dtype, inner):
    """Scores in the hundreds: exp overflows without the max subtraction or the online rescale."""
    return _qkv(rng, rows, n, inner, dtype, qk_scale=4.0)


def _peaked(rng, rows, n, dtype, inner):
    """Each query matches one visible key far better than the rest."""
    d = _qkv(rng, rows, n, inner, dtype)
    q = d["q"].astype(np.float64)
    k = d["k"].astype(np.float64)
    for i in range(rows):
        j = rng.integers(0, i + (inner - rows) + 1)
        k[j] = q[i] * 3
    d["k"] = k.astype(dtype)
    return d


def _ramp_v(rng, rows, n, dtype, inner):
    """v grows with the key index, so seeing one key too many or too few shifts every output."""
    v = np.arange(inner)[:, None] / max(inner, 1) * 4 + rng.standard_normal((inner, n)) * 0.01
    return _qkv(rng, rows, n, inner, dtype, v=v)


def reference(inputs):
    q, k, v = (inputs[x].astype(np.float64) for x in ("q", "k", "v"))
    lq, d = q.shape
    lk = k.shape[0]
    s = q @ k.T / math.sqrt(d)
    visible = np.arange(lk)[None, :] <= np.arange(lq)[:, None] + (lk - lq)
    s = np.where(visible, s, -np.inf)
    with np.errstate(invalid="ignore"):
        s = s - s.max(axis=-1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(axis=-1, keepdims=True)
    return p @ v


def mlx_baseline(mx, inputs):
    q, k, v = (inputs[x][None, None] for x in ("q", "k", "v"))
    scale = 1.0 / math.sqrt(q.shape[-1])
    mask = inputs["causal_mask"] if "causal_mask" in inputs else "causal"
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)[0, 0]


def invariants(inputs, out, ref):
    return []


def launch(rows, n, config):
    tg = config["TG"]
    return (math.ceil(rows / config["BQ"]) * tg, 1, 1), (tg, 1, 1)


HEAD_DIMS = (64, 128, 80, 96)


def sizes(rng, purpose):
    d = int(rng.choice(HEAD_DIMS)) if rng.random() < 0.6 else off_grid_n(rng, 3, 160)
    if purpose == "timing":
        lq = int(rng.integers(300, 1200))
        return lq, int(rng.choice((64, 128))), lq + int(rng.integers(0, 1500))
    lq = int(rng.integers(1, 200 if purpose == "fuzz" else 300))
    return lq, d, lq + int(rng.integers(0, 600 if purpose == "fuzz" else 900))


# (Lq, D, Lk): tails in every dimension, Lq < Lk (cache offset), and one long key sequence.
SHAPES = [(1, 1, 1), (1, 8, 1), (1, 64, 7), (3, 33, 3), (7, 64, 33), (33, 64, 33), (31, 80, 65),
          (64, 128, 64), (65, 128, 129), (16, 64, 1025), (127, 40, 255), (5, 96, 4097)]

VISIBLE = (
    [Case("normal", lq, d, dt, 1000 + lq + d + lk, "shape", lk) for lq, d, lk in SHAPES for dt in ("float32", "float16")]
    + [Case(kind, 33, 64, dt, 61, "values", 65)
       for kind in ("large_scores", "peaked", "ramp_v", "uniform01") for dt in ("float32", "float16")]
    + [Case(kind, 64, 64, "float32", 67, "values", 64) for kind in ("large_scores", "ramp_v")]
    + [Case("normal", lq, d, "bfloat16", 1100 + lq, "bf16", lk) for lq, d, lk in ((1, 64, 7), (33, 64, 33), (31, 80, 65))]
    + [Case(kind, 33, 64, "bfloat16", 71, "bf16", 65) for kind in ("large_scores", "ramp_v")]
)

SPEC = OpSpec(
    name="attention",
    summary="fused causal attention, FlashAttention-style",
    input_names=("q", "k", "v"),
    contract=CONTRACT,
    generators={
        "normal": _normal, "uniform01": _uniform01, "large_scores": _large_scores,
        "peaked": _peaked, "ramp_v": _ramp_v,
    },
    reference=reference,
    mlx_baseline=mlx_baseline,
    invariants=invariants,
    visible=VISIBLE,
    fuzz_kinds=("normal", "large_scores", "ramp_v", "peaked"),
    hidden_kinds=("peaked", "large_scores", "normal", "ramp_v"),
    # Prefill shapes with Lq == Lk, so the baseline uses MLX's built-in causal mask (no mask read).
    timing_shapes=((512, 64, 512), (1024, 128, 1024), (2048, 64, 2048)),
    # KernelBench-style: one square shape (Lq == Lk), where a missing cache offset is invisible.
    naive=Case("uniform01", 128, 64, "float32", 42, "naive", 128),
    dims=3,
    launch=launch,
    required_params=("BQ",),
    bytes_fn=lambda lq, d, lk, item: (2 * lq * d + 2 * lk * d) * item,
    flops_fn=lambda lq, d, lk: 4 * lq * lk * d // 2,  # QK^T and PV, about half the pairs causal-visible
    sizes=sizes,
    about="The heart of every transformer: each word scores every earlier word, softmaxes the scores "
          "and takes a weighted sum, in one kernel that never writes the big score table to memory. "
          "It's where kernels break: the causal boundary off by one, the offset when there are more "
          "keys than queries (a KV cache), the running-max rescale. MLX's fused attention is heavily "
          "tuned, so this is the hardest test in carmen, for Carmy and for the judge.",
)
