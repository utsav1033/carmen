"""mlp_up and mlp_down: a Llama/Qwen MLP block at decode, in 2 kernels instead of 8.

After attention, every layer of a Llama- or Qwen-style model runs, one MLX kernel each:

    h = x + attn                      add
    hn = rmsnorm(h) * w               norm
    g = Wg @ hn ; u = Wu @ hn         2 quantized matmuls (4-bit weights)
    a = silu(g) * u                   silu, multiply
    out = h + Wd @ a                  quantized matmul, add

At decode (one word) each of those is a tiny kernel, and launching kernels is most of the
cost: Qwen 0.5B runs at 44% of its speed limit. Two kernels can do it all:

    mlp_up   : a = silu(Wg @ rmsnorm(x + res)) * (Wu @ rmsnorm(x + res))
    mlp_down : out = x + res + Wd @ a

h never goes to memory: each kernel recomputes x + res where it needs it (it's tiny).

Weights are in MLX's 4-bit affine format, exactly as mlx-lm loads them, so the kernels run
on a real model's weights with no conversion: per output row, uint32 words each holding 8
4-bit values (element k is bits 4*(k%8).. of word k/8, lowest bits first), and one float
scale and bias per group of 64 consecutive values: w = scale * q + bias.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Case, OpSpec, off_grid_n

GROUP = 64
BITS = 4

_FORMAT = """\
4-bit weight format (MLX affine quantization, group size 64, as mlx-lm loads models):
  For a weight matrix W of shape (rows_out, K), K a multiple of 64:
  q      : uint32, shape (rows_out, K / 8). Element W[j, k] is the 4-bit value
           (q[j * (K/8) + k/8] >> (4 * (k % 8))) & 0xF   (lowest bits first)
  scales : type T, shape (rows_out, K / 64)
  biases : type T, shape (rows_out, K / 64)
  W[j, k] = float(scales[j * (K/64) + k/64]) * float(that 4-bit value) + float(biases[j * (K/64) + k/64])
"""

_LAUNCH = """\
How the harness launches it:
  - Each config must define template constant BN: how many output columns one threadgroup computes.
  - grid = (ceil(n / BN) * TG, rows, 1), threadgroup = (TG, 1, 1).
    threadgroup_position_in_grid.x is the column tile, .y the row (the token).
  - This is a decode kernel: rows is small (1 when writing one word, at most a few).
    The last tile is partial: n need not be a multiple of BN.
"""

UP_CONTRACT = f"""\
Compute, for each row r (a token) and output column j:
  h  = x[r] + res[r]                                (length K vector)
  hn = h / sqrt(mean(h^2) + eps) * w                (rmsnorm)
  g  = sum_k Wg[j, k] * hn[k] ;  u = sum_k Wu[j, k] * hn[k]
  out[r, j] = silu(g) * u,  where silu(g) = g / (1 + exp(-g))

Buffers:
  x, res : input, shape (rows, K), type T (float, half or bfloat)
  w      : input, shape (K,), type T (the norm weight)
  eps    : input, shape (1,), float32 (read eps[0])
  gq, gs, gb : the gate weight Wg (n, K) in the 4-bit format below
  uq, us, ub : the up weight Wu (n, K) in the same format
  out    : output, flat, rows * n elements of type T; write out[r * n + j]
K = x_shape[1] (a multiple of 64), n = gq_shape[0], rows = x_shape[0].

{_FORMAT}
{_LAUNCH}
Semantics that the judge checks exactly:
  - Accumulate in float32 even when T is half or bfloat.
  - silu must not overflow: |g| can be large (exp(-g) is inf for g < -88; the answer is then 0, not NaN).
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""

DOWN_CONTRACT = f"""\
Compute, for each row r (a token) and output column j:
  out[r, j] = x[r, j] + res[r, j] + sum_k Wd[j, k] * act[r, k]

Buffers:
  act    : input, shape (rows, K), type T (the MLP activation, silu(g) * u)
  x, res : input, shape (rows, n), type T (residual stream and attention output: out = their sum + Wd @ act)
  dq, ds, db : the down weight Wd (n, K) in the 4-bit format below
  out    : output, flat, rows * n elements of type T; write out[r * n + j]
K = act_shape[1] (a multiple of 64), n = dq_shape[0], rows = act_shape[0].

{_FORMAT}
{_LAUNCH}
Semantics that the judge checks exactly:
  - Accumulate in float32 even when T is half or bfloat. Add x and res in float32 too.
  - Every one of the rows * n outputs must be written. Nothing past rows * n may be written.
"""


# ── 4-bit affine quantization, the MLX layout ────────────────────────────────────
def quantize(w: np.ndarray, dtype: str, flat_groups: bool = False):
    """(q uint32 (rows, K/8), scales, biases) for w of shape (rows, K). Scales and biases are
    rounded to `dtype` here; the reference dequantizes the stored values, so it is exact."""
    rows, k = w.shape
    g = w.reshape(rows, k // GROUP, GROUP)
    lo, hi = g.min(axis=-1), g.max(axis=-1)
    scale = (hi - lo) / 15.0
    if flat_groups:  # constant groups: scale 0, the whole weight lives in the bias
        scale = np.zeros_like(scale)
    s = scale.astype(dtype if dtype != "bfloat16" else "float32")
    b = lo.astype(dtype if dtype != "bfloat16" else "float32")
    safe = np.where(s.astype(np.float64) > 0, s.astype(np.float64), 1.0)[..., None]
    q = np.clip(np.rint((g - b.astype(np.float64)[..., None]) / safe), 0, 15).astype(np.uint32)
    q = q.reshape(rows, k // 8, 8)
    packed = np.zeros((rows, k // 8), dtype=np.uint32)
    for t in range(8):
        packed |= q[..., t] << np.uint32(4 * t)
    return packed, s, b


def dequantize(q: np.ndarray, s: np.ndarray, b: np.ndarray) -> np.ndarray:
    rows, words = q.shape
    vals = np.stack([(q >> np.uint32(4 * t)) & np.uint32(0xF) for t in range(8)], axis=-1)
    vals = vals.reshape(rows, words * 8).astype(np.float64)
    return np.repeat(s.astype(np.float64), GROUP, axis=1) * vals + np.repeat(b.astype(np.float64), GROUP, axis=1)


def _weights(rng, rows_out, k, dtype, scale=None, ramp=False, flat=False):
    w = rng.standard_normal((rows_out, k)) * (scale if scale is not None else 1 / math.sqrt(k))
    if ramp:  # each output row has its own offset: swapped row indexing changes every value
        w = w + np.arange(rows_out)[:, None] * (0.5 / rows_out)
    return quantize(w, dtype, flat_groups=flat)


def _act(rng, shape, dtype, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(dtype)


# ── mlp_up ─────────────────────────────────────────────────────────────────────────
EPS_CHOICES = (1e-5, 1e-6)


def _up(rng, rows, n, dtype, inner, sx=1.0, sr=1.0, wscale=None, ramp=False, flat=False, outliers=False):
    x = rng.standard_normal((rows, inner)) * sx
    if outliers:  # LLM activations have a few huge channels
        x[:, rng.choice(inner, size=max(1, inner // 64), replace=False)] *= 200
    d = {"x": x.astype(dtype), "res": _act(rng, (rows, inner), dtype, sr),
         "w": (1 + 0.3 * rng.standard_normal(inner)).astype(dtype),
         "eps": np.array([EPS_CHOICES[int(rng.integers(2))]], dtype=np.float32)}
    for p in ("g", "u"):
        q, s, b = _weights(rng, n, inner, dtype, wscale, ramp, flat)
        d[p + "q"], d[p + "s"], d[p + "b"] = q, s, b
    return d


UP_GENERATORS = {
    "normal": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner),
    "uniform01": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner, sx=0.5, sr=0.5),
    # |g| in the tens to hundreds: exp(-g) overflows for the negative ones
    "big_act": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner, wscale=40 / math.sqrt(inner)),
    "ramp": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner, ramp=True),
    # whole groups constant: scale 0, the weight is all bias (a kernel that drops biases is wrong)
    "flat_groups": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner, flat=True),
    "outlier": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner, outliers=True),
    # mean(h^2) ~ 1e-6, the size of eps: a wrong or missing eps shows
    "tiny_rows": lambda rng, r, n, dt, inner: _up(rng, r, n, dt, inner, sx=7e-4, sr=7e-4),
    # res = -x: h is 0, so out is 0 exactly unless eps is ignored (0/0)
    "cancel": lambda rng, r, n, dt, inner: (lambda d: {**d, "res": (-d["x"].astype(np.float64)).astype(dt)})(
        _up(rng, r, n, dt, inner)),
}


def _silu(g):
    with np.errstate(over="ignore"):
        return g / (1 + np.exp(-g))


def up_reference(d):
    h = d["x"].astype(np.float64) + d["res"].astype(np.float64)
    hn = h / np.sqrt((h ** 2).mean(axis=-1, keepdims=True) + float(d["eps"][0])) * d["w"].astype(np.float64)
    g = hn @ dequantize(d["gq"], d["gs"], d["gb"]).T
    u = hn @ dequantize(d["uq"], d["us"], d["ub"]).T
    return _silu(g) * u


def up_baseline(mx, d):
    """What the stock model runs: add, rms_norm, two quantized matmuls, silu, multiply."""
    import mlx.nn as nn
    eps = d["eps"]
    eps = eps if isinstance(eps, float) else float(eps.item())
    hn = mx.fast.rms_norm(d["x"] + d["res"], d["w"], eps)
    g = mx.quantized_matmul(hn, d["gq"], d["gs"], d["gb"], transpose=True, group_size=GROUP, bits=BITS)
    u = mx.quantized_matmul(hn, d["uq"], d["us"], d["ub"], transpose=True, group_size=GROUP, bits=BITS)
    return nn.silu(g) * u


# ── mlp_down ───────────────────────────────────────────────────────────────────────
def _down(rng, rows, n, dtype, inner, sa=1.0, wscale=None, ramp=False, flat=False, cancel=False, big_res=False):
    x = rng.standard_normal((rows, n)) * (30 if big_res else 1)
    res = -x if cancel else rng.standard_normal((rows, n))
    q, s, b = _weights(rng, n, inner, dtype, wscale, ramp, flat)
    return {"act": _act(rng, (rows, inner), dtype, sa), "x": x.astype(dtype), "res": res.astype(dtype),
            "dq": q, "ds": s, "db": b}


DOWN_GENERATORS = {
    "normal": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner),
    "uniform01": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner, sa=0.5),
    "ramp": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner, ramp=True),
    "flat_groups": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner, flat=True),
    # res = -x: the residual cancels and only Wd @ act is left; dropping either residual term shows
    "cancel": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner, cancel=True),
    # a big residual stream next to a small MLP update, like deep layers of a real model
    "big_residual": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner, big_res=True),
    "big_act": lambda rng, r, n, dt, inner: _down(rng, r, n, dt, inner, sa=40.0),
}


def down_reference(d):
    a = d["act"].astype(np.float64)
    return d["x"].astype(np.float64) + d["res"].astype(np.float64) + a @ dequantize(d["dq"], d["ds"], d["db"]).T


def down_baseline(mx, d):
    """What the stock model runs: a quantized matmul, then h + mlp (h = x + res, already summed in the model)."""
    y = mx.quantized_matmul(d["act"], d["dq"], d["ds"], d["db"], transpose=True, group_size=GROUP, bits=BITS)
    return (d["x"] + d["res"]) + y


# ── shared ───────────────────────────────────────────────────────────────────────
def invariants(inputs, out, ref):
    return []


def launch(rows, n, config):
    tg = config["TG"]
    return (math.ceil(n / config["BN"]) * tg, rows, 1), (tg, 1, 1)


def _sizes(rng, purpose):
    """(rows, n, K): few rows (decode), any n, K a multiple of 64."""
    if purpose == "timing":
        return 1, off_grid_n(rng, 2000, 6000), GROUP * int(rng.integers(10, 40))
    hi_rows, hi_n, hi_k = (6, 3000, 48) if purpose == "fuzz" else (9, 5000, 80)
    return int(rng.integers(1, hi_rows)), off_grid_n(rng, 2, hi_n), GROUP * int(rng.integers(1, hi_k))


# Visible battery: tails (n not a multiple of anything), K from one group up, real model shapes.
SHAPES = [(1, 1, 64), (1, 7, 64), (1, 33, 128), (2, 100, 192), (1, 257, 512), (3, 513, 640), (1, 1000, 896)]


def _visible(model_shapes, kinds):
    return (
        [Case("normal", r, n, dt, 1500 + r + n + k, "shape", k) for r, n, k in SHAPES for dt in ("float32", "float16")]
        + [Case("normal", r, n, dt, 1600 + n, "model", k) for r, n, k in model_shapes for dt in ("float16", "bfloat16")]
        + [Case(kind, 2, 301, dt, 61, "values", 384) for kind in kinds for dt in ("float32", "float16")]
        + [Case(kind, 1, 129, "bfloat16", 67, "bf16", 256) for kind in kinds[:3]]
    )


def _bytes(weight_sets):
    def fn(rows, n, k, item):
        weights = n * k // 2 + 2 * n * (k // GROUP) * item  # 4-bit values + scales + biases
        return weight_sets * weights + rows * (2 * k + n) * item
    return fn


# Qwen2.5-0.5B (hidden 896, MLP 4864) at one token is in the visible battery; Llama-3.2-1B's
# (2048, 8192) is timed. Hidden draws cover other sizes.
UP_SPEC = OpSpec(
    name="mlp_up",
    summary="rmsnorm + 4-bit gate & up matmuls + silu, one kernel (decode)",
    input_names=("x", "res", "w", "eps", "gq", "gs", "gb", "uq", "us", "ub"),
    contract=UP_CONTRACT,
    generators=UP_GENERATORS,
    reference=up_reference,
    mlx_baseline=up_baseline,
    invariants=invariants,
    visible=_visible([(1, 4864, 896)],
                     ("big_act", "ramp", "flat_groups", "outlier", "tiny_rows", "cancel")),
    fuzz_kinds=("normal", "ramp", "big_act", "flat_groups"),
    hidden_kinds=("outlier", "normal", "cancel", "big_act", "ramp", "tiny_rows"),
    timing_shapes=((1, 4864, 896), (1, 8192, 2048)),
    naive=Case("uniform01", 1, 256, "float32", 42, "naive", 256),
    dims=3,
    launch=launch,
    required_params=("BN",),
    bytes_fn=_bytes(2),
    flops_fn=lambda m, n, k: 4 * m * n * k,
    sizes=_sizes,
    int_inputs=("gq", "uq"),
    keep_inner_on_shrink=True,
    about="The first half of a model's MLP block at decode, in one kernel: the residual add, the norm, "
          "the gate and up matmuls on 4-bit weights, and silu. MLX runs those as 6 kernels. At one word "
          "at a time, launching kernels is most of the cost, so this is about running fewer of them while "
          "still reading the weights as fast as MLX does.",
)

DOWN_SPEC = OpSpec(
    name="mlp_down",
    summary="4-bit down matmul + residual add, one kernel (decode)",
    input_names=("act", "x", "res", "dq", "ds", "db"),
    contract=DOWN_CONTRACT,
    generators=DOWN_GENERATORS,
    reference=down_reference,
    mlx_baseline=down_baseline,
    invariants=invariants,
    visible=_visible([(1, 896, 4864)], ("ramp", "flat_groups", "cancel", "big_residual", "big_act")),
    fuzz_kinds=("normal", "ramp", "cancel", "flat_groups"),
    hidden_kinds=("big_residual", "normal", "cancel", "ramp", "big_act"),
    timing_shapes=((1, 896, 4864), (1, 2048, 8192)),
    naive=Case("uniform01", 1, 256, "float32", 42, "naive", 256),
    dims=3,
    launch=launch,
    required_params=("BN",),
    bytes_fn=_bytes(1),
    flops_fn=lambda m, n, k: 2 * m * n * k,
    sizes=_sizes,
    int_inputs=("dq",),
    keep_inner_on_shrink=True,
    about="The second half of a model's MLP block at decode, in one kernel: the down matmul on 4-bit "
          "weights plus both residual adds. With mlp_up, the whole block after attention becomes 2 "
          "kernels instead of 8.",
)
