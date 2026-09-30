"""Apple Metal via MLX's mx.fast.metal_kernel. The only file that knows about Metal."""

from __future__ import annotations

import time

import numpy as np

from .base import PAD, Kernel

try:
    import mlx.core as mx
except ImportError as e:  # pragma: no cover - only on machines without MLX
    raise ImportError("the metal backend needs MLX on Apple Silicon: pip install 'carmen[metal]'") from e

_DTYPES = {"float32": mx.float32, "float16": mx.float16}

COPY_KERNEL = """
    uint i = thread_position_in_grid.x;
    if (i < inp_shape[0]) { out[i] = inp[i]; }
"""


class MetalAdapter:
    name = "metal"

    def chip(self) -> str:
        info = mx.metal.device_info() if hasattr(mx, "metal") and hasattr(mx.metal, "device_info") else mx.device_info()
        return str(info.get("device_name", "apple-gpu"))

    def build(self, op, kernel: Kernel):
        return mx.fast.metal_kernel(
            name=f"carmen_{op.name}_{kernel.digest()}",
            input_names=list(op.input_names),
            output_names=["out"],
            source=kernel.source,
            header=kernel.header,
        )

    def upload(self, inputs):
        return {k: mx.array(v) for k, v in inputs.items()}

    def _call(self, built, op, config, dev_inputs, rows, n, dtype):
        tg = config["TG"]
        template = [("T", _DTYPES[dtype])] + [(k, v) for k, v in config.items()]
        (out,) = built(
            inputs=[dev_inputs[name] for name in op.input_names],
            template=template,
            # The harness sets the launch shape: one threadgroup per row. Carmy never picks
            # the grid, which rules out MLX's silent dispatch truncation (grid < threadgroup).
            grid=(rows * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(rows * n + PAD,)],
            output_dtypes=[_DTYPES[dtype]],
            init_value=float("nan"),
        )
        return out

    def launch(self, built, op, config, dev_inputs, rows, n, dtype, inner: int = 1):
        # `inner` launches share one eval, so per-call Python/dispatch overhead is amortized
        # and the timer sees GPU time, not host time.
        def go():
            mx.eval(*[self._call(built, op, config, dev_inputs, rows, n, dtype) for _ in range(inner)])
        return go

    def run(self, built, op, config, dev_inputs, rows, n, dtype):
        out = self._call(built, op, config, dev_inputs, rows, n, dtype)
        mx.eval(out)
        return np.array(out)

    def baseline(self, op, dev_inputs):
        def go():
            y = op.mlx_baseline(mx, dev_inputs)
            mx.eval(y)
            return np.array(y)
        return go

    def baseline_launch(self, op, dev_inputs, inner: int = 1):
        def go():
            mx.eval(*[op.mlx_baseline(mx, dev_inputs) for _ in range(inner)])
        return go

    def release(self) -> None:
        """Hand MLX's cached buffers back to the OS between timing runs."""
        (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()

    def measure_peak_gbps(self) -> float:
        n = 64 * 1024 * 1024  # 256 MB of float32
        kern = mx.fast.metal_kernel(name="carmen_copy", input_names=["inp"], output_names=["out"], source=COPY_KERNEL)
        inp = mx.random.uniform(shape=(n,))
        mx.eval(inp)

        def go():
            (o,) = kern(inputs=[inp], grid=(n, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(n,)], output_dtypes=[mx.float32])
            mx.eval(o)

        for _ in range(3):
            go()
        best = float("inf")
        for _ in range(20):
            t = time.perf_counter()
            go()
            best = min(best, time.perf_counter() - t)
        return 2 * n * 4 / best / 1e9
