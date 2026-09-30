"""A numpy stand-in for a GPU, so the judge's logic can be tested anywhere.

`Kernel.source` names a Python function here instead of holding Metal code. This is
test scaffolding only: the real harness never executes Python from a kernel.
"""

from __future__ import annotations

import numpy as np

from carmen.backends import PAD


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


def good(op_name, inputs, cfg):
    return _softmax32(_prep(op_name, inputs))


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


KERNELS = {f.__name__: f for f in (good, no_max_subtraction, drops_last_element, stores_through_fp16,
                                    racy, zeros, bad_when_big_tg, writes_past_end)}


class FakeAdapter:
    name = "fake"

    def chip(self):
        return "numpy"

    def build(self, op, kernel):
        if kernel.source not in KERNELS:
            raise RuntimeError(f"error: use of undeclared identifier '{kernel.source}'")
        return (op.name, KERNELS[kernel.source], kernel.source)

    def upload(self, inputs):
        return inputs

    def run(self, built, op, config, dev, rows, n, dtype):
        op_name, fn, name = built
        out = np.full(rows * n + PAD, np.nan, dtype=dtype)
        out[: rows * n] = fn(op_name, dev, config).astype(dtype).ravel()
        if name == "writes_past_end":
            out[rows * n] = 0
        return out

    def launch(self, built, op, config, dev, rows, n, dtype):
        return lambda: self.run(built, op, config, dev, rows, n, dtype)

    def baseline(self, op, dev):
        return lambda: good(op.name, dev, {}).astype(dev["x"].dtype)

    def baseline_launch(self, op, dev):
        return lambda: self.baseline(op, dev)()

    def measure_peak_gbps(self):
        return 100.0
