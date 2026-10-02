"""A numpy stand-in for a GPU, so the judge's logic can be tested anywhere.

`Kernel.source` names a Python function here instead of holding Metal code. This is
test scaffolding only: the real harness never executes Python from a kernel.
"""

from __future__ import annotations

import numpy as np

from carmen.backends import PAD
from carmen.ops.base import to_bf16


def _store(a, dtype):
    """What the GPU would hold in a buffer of `dtype` (bf16 kept as exact float32 values)."""
    return to_bf16(a) if dtype == "bfloat16" else np.asarray(a).astype(dtype)


def _softmax32(v):
    v = v.astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        m = v.max(axis=-1, keepdims=True)
        e = np.exp(v - m)
        return e / e.sum(axis=-1, keepdims=True)


def _prep(op_name, inputs):
    x = inputs["x"].astype(np.float32)
    if op_name == "masked_softmax":
        return x * inputs["scale"][0] + inputs["mask"].astype(np.float32)
    return x


def _norm32(op_name, inputs):
    """Float32 two-pass layernorm / rmsnorm, like the golden kernels."""
    x, w = inputs["x"].astype(np.float32), inputs["w"].astype(np.float32)
    if op_name == "rmsnorm":
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(1e-5)) * w
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + np.float32(1e-5)) * w + inputs["b"].astype(np.float32)


def _attention32(inputs, offset_aware=True):
    q, k, v = (inputs[x].astype(np.float32) for x in ("q", "k", "v"))
    lq, d = q.shape
    lk = k.shape[0]
    s = q @ k.T / np.float32(np.sqrt(d))
    shift = (lk - lq) if offset_aware else 0
    s = np.where(np.arange(lk)[None, :] <= np.arange(lq)[:, None] + shift, s, -np.inf)
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)) @ v


def good(op_name, inputs, cfg):
    if op_name == "residual_rmsnorm":
        h = inputs["x"].astype(np.float32) + inputs["res"].astype(np.float32)
        y = h / np.sqrt((h * h).mean(axis=-1, keepdims=True) + inputs["eps"][0]) * inputs["w"].astype(np.float32)
        return np.concatenate([h, y])
    if op_name == "attention":
        return _attention32(inputs)
    if op_name == "matmul":
        return inputs["a"].astype(np.float32) @ inputs["b"].astype(np.float32)
    if op_name == "add_rmsnorm":
        h = inputs["x"].astype(np.float32) + inputs["res"].astype(np.float32)
        return _norm32("rmsnorm", {"x": h, "w": inputs["w"]})
    if op_name in ("layernorm", "rmsnorm"):
        return _norm32(op_name, inputs)
    return _softmax32(_prep(op_name, inputs))


def one_pass_variance(op_name, inputs, cfg):
    """layernorm with var = E[x^2] - mean^2: right near zero, wrong for offset rows."""
    x = inputs["x"].astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = np.maximum((x * x).mean(axis=-1, keepdims=True) - mean * mean, 0)
    return (x - mean) / np.sqrt(var + np.float32(1e-5)) * inputs["w"].astype(np.float32) + inputs["b"]


def no_max_subtraction(op_name, inputs, cfg):
    v = _prep(op_name, inputs)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(v)
        return e / e.sum(axis=-1, keepdims=True)


def drops_last_element(op_name, inputs, cfg):
    y = _softmax32(_prep(op_name, inputs))
    y[:, -1] = np.nan  # never written: stays at the NaN poison value
    return y


def stores_through_fp16(op_name, inputs, cfg):
    return _softmax32(_prep(op_name, inputs)).astype(np.float16).astype(np.float32)


def racy(op_name, inputs, cfg, _state={"i": 0}):
    _state["i"] += 1
    y = _softmax32(_prep(op_name, inputs))
    if _state["i"] % 3 == 0 and y.shape[1] > 256:
        y[0, 0] *= 1.5
    return y


def zeros(op_name, inputs, cfg):
    return np.zeros(inputs["x"].shape, np.float32)


def bad_when_big_tg(op_name, inputs, cfg):
    y = _softmax32(_prep(op_name, inputs))
    if cfg["TG"] > 512:
        y *= 1.01
    return y


def writes_past_end(op_name, inputs, cfg):
    return _softmax32(_prep(op_name, inputs))


def wrong_leading_dim(op_name, inputs, cfg):
    """matmul reading row r of A at r * N instead of r * K: exactly right when the matrices are square."""
    a, b = inputs["a"].astype(np.float32), inputs["b"].astype(np.float32)
    (m, k), n = a.shape, b.shape[1]
    flat = np.concatenate([a.ravel(), np.zeros(m * n + k, np.float32)])
    rows = np.stack([flat[r * n: r * n + k] for r in range(m)])
    return rows @ b


def cache_offset_ignored(op_name, inputs, cfg):
    """attention with the causal mask aligned top-left: exactly right when Lq == Lk."""
    return _attention32(inputs, offset_aware=False)


def eps_hardcoded(op_name, inputs, cfg):
    """residual_rmsnorm with Llama's eps baked in: right only when the model's eps is 1e-5."""
    return good(op_name, {**inputs, "eps": np.array([1e-5], np.float32)}, cfg)


KERNELS = {f.__name__: f for f in (good, no_max_subtraction, drops_last_element, stores_through_fp16,
                                    racy, zeros, bad_when_big_tg, writes_past_end, one_pass_variance,
                                    wrong_leading_dim, cache_offset_ignored,
                                    eps_hardcoded)}


class FakeAdapter:
    name = "fake"

    def chip(self):
        return "numpy"

    def build(self, op, kernel):
        if kernel.source not in KERNELS:
            raise RuntimeError(f"error: use of undeclared identifier '{kernel.source}'")
        return (op.name, KERNELS[kernel.source], kernel.source)

    def upload(self, inputs, dtype="float32"):
        return inputs

    def run(self, built, op, config, dev, rows, n, dtype):
        op_name, fn, name = built
        size = op.outputs * rows * n
        out = np.full(size + PAD, np.nan, dtype=np.float32 if dtype == "bfloat16" else dtype)
        out[:size] = _store(fn(op_name, dev, config), dtype).ravel()
        if name == "writes_past_end":
            out[size] = 0
        return out

    def launch(self, built, op, config, dev, rows, n, dtype, inner=1):
        return lambda: [self.run(built, op, config, dev, rows, n, dtype) for _ in range(inner)]

    def baseline(self, op, dev):
        return lambda: good(op.name, dev, {}).astype(dev[op.input_names[0]].dtype)

    def baseline_launch(self, op, dev, inner=1, compiled=False):
        return lambda: [self.baseline(op, dev)() for _ in range(inner)]

    def measure_peak_gbps(self):
        return 100.0
