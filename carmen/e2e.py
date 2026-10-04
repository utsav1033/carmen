"""`carmen e2e <model>`: does a real model get faster, with the same answers?

Runs one model four ways and compares each against stock MLX:

  stock     the model as mlx-lm loads it
  plumbing  gate and up projections merged into one matmul per layer (weights concatenated once)
  kernel    carmen's residual_rmsnorm swapped into every layer: the residual add and the norm
            after it become one kernel, twice per layer (mid-layer, and the end of a layer fused
            with the next layer's input norm), so 4 small kernels per layer become 2
  both      plumbing + kernel

For each: prefill and decode tokens/s, and whether the outputs still match stock (the same
greedy tokens, and how far the first logits moved). A faster model with different answers is
reported as broken, not faster.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import Kernel
from .profile import PROMPT_TEXT

MODES = ("stock", "plumbing", "kernel", "both")
OP = "residual_rmsnorm"


@dataclass
class Result:
    mode: str
    prefill_tps: float
    decode_tps: float
    tokens_match: int | None = None  # how many generated tokens equal stock's, from the start
    tokens_total: int = 0
    max_logit_diff: float | None = None  # first-step logits vs stock
    same_first_token: bool | None = None


# ── which kernel to use ─────────────────────────────────────────────────────────────
def latest_champion(runs_dir: Path, op: str = OP) -> tuple[Kernel, dict, str, dict] | None:
    """The champion of the newest run of `op` that has one: (kernel, best config, run id, verdict)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    for d in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        ev, sm = d / "events.jsonl", d / "summary.json"
        if not ev.exists() or not sm.exists():
            continue
        first = json.loads(ev.read_text().splitlines()[0])
        champ = json.loads(sm.read_text()).get("champion")
        if first.get("op") != op or not champ:
            continue
        a = d / "attempts" / champ
        kernel = Kernel.from_json(json.loads((a / "kernel.json").read_text()))
        verdict = json.loads((a / "verdict.json").read_text())
        return kernel, verdict.get("best_config") or kernel.configs[0], f"{d.name}:{champ}", verdict
    return None


def verdict_lines(results: list[Result]) -> list[str]:
    """Plain-English summary, stock first."""
    stock = next((r for r in results if r.mode == "stock"), None)
    lines = []
    for r in results:
        if r is stock or stock is None:
            continue
        gain = r.decode_tps / stock.decode_tps - 1
        same = r.tokens_match == r.tokens_total
        lines.append(f"{r.mode}: decode {gain:+.1%} vs stock, "
                     + ("same answers" if same else f"answers diverge at token {r.tokens_match + 1}"))
    return lines


# ── the part that needs MLX and mlx-lm ─────────────────────────────────────────────
def _imports():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as e:  # pragma: no cover - only on machines without MLX
        raise SystemExit("carmen e2e needs MLX and mlx-lm on Apple Silicon: pip install 'carmen[models]'") from e
    return mx, nn, load, make_prompt_cache


def _layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise SystemExit("unsupported architecture: expected model.model.layers (Llama, Qwen, Mistral style)")
    return layers


def _subclass(obj, name: str, call):
    """Give one instance its own __call__ without touching the shared class (other loads stay stock)."""
    cls = obj.__class__
    obj.__class__ = type(f"Carmen{name}{cls.__name__}", (cls,), {"__call__": call})


def apply_plumbing(mx, nn, model) -> int:
    """Merge each MLP's gate and up projections into one matmul. Returns how many layers changed."""
    import copy
    changed = 0
    for layer in _layers(model):
        mlp = layer.mlp
        g, u = getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None)
        if g is None or u is None or "bias" in g:
            continue
        merged = copy.copy(g)
        for key in ("weight", "scales", "biases"):  # quantized layers carry scales and biases too
            if key in g:
                merged[key] = mx.concatenate([g[key], u[key]], axis=0)
        mx.eval(merged.parameters())
        split = g["weight"].shape[0]

        def call(self, x, _m=merged, _s=split):
            gu = _m(x)
            return self.down_proj(nn.silu(gu[..., :_s]) * gu[..., _s:])
        object.__setattr__(mlp, "_carmen_gate_up", merged)
        _subclass(mlp, "GateUp", call)
        changed += 1
    return changed


def make_fused(mx, kernel: Kernel, config: dict):
    """carmen's residual_rmsnorm as a function (x, res, norm) -> (h, y)."""
    from . import ops
    op = ops.get(OP)
    k = mx.fast.metal_kernel(name=f"carmen_{OP}_{kernel.digest()}", input_names=list(op.input_names),
                             output_names=["out"], source=kernel.source, header=kernel.header)
    eps_cache: dict[float, object] = {}

    def fused(x, res, norm):
        shape, n = x.shape, x.shape[-1]
        rows = x.size // n
        eps = float(getattr(norm, "eps", 1e-5))
        if eps not in eps_cache:
            eps_cache[eps] = mx.array([eps], dtype=mx.float32)
        grid, tg = op.grid(rows, n, config)
        (out,) = k(inputs=[x.reshape(rows, n), res.reshape(rows, n), norm.weight, eps_cache[eps]],
                   template=[("T", x.dtype)] + list(config.items()), grid=grid, threadgroup=tg,
                   output_shapes=[(2, rows, n)], output_dtypes=[x.dtype])
        return out[0].reshape(shape), out[1].reshape(shape)
    return fused


def make_fused_prebuilt(mx, kernel: Kernel, config: dict):
    """Same kernel, but everything that only depends on the shape (template, grid, eps array,
    output shape) is built once per shape, so a call does the least Python possible."""
    from . import ops
    op = ops.get(OP)
    k = mx.fast.metal_kernel(name=f"carmen_{OP}_{kernel.digest()}", input_names=list(op.input_names),
                             output_names=["out"], source=kernel.source, header=kernel.header)
    plans: dict = {}

    def fused(x, res, norm):
        key = (x.shape, x.dtype, id(norm))
        p = plans.get(key)
        if p is None:
            n = x.shape[-1]
            rows = x.size // n
            grid, tg = op.grid(rows, n, config)
            eps = mx.array([float(getattr(norm, "eps", 1e-5))], dtype=mx.float32)
            p = plans[key] = (rows, n, x.shape, eps, [("T", x.dtype)] + list(config.items()), grid, tg,
                              [(2, rows, n)], [x.dtype])
        rows, n, shape, eps, template, grid, tg, out_shapes, out_dtypes = p
        (out,) = k(inputs=[x.reshape(rows, n), res.reshape(rows, n), norm.weight, eps], template=template,
                   grid=grid, threadgroup=tg, output_shapes=out_shapes, output_dtypes=out_dtypes)
        return out[0].reshape(shape), out[1].reshape(shape)
    return fused


# ── where does the time of one call go? ───────────────────────────────────────────────
CALL_CHAIN = 48  # residual+norm calls per word in Qwen 0.5B (24 layers x 2)
CALL_REPS = 30


@dataclass
class CallTiming:
    variant: str
    rows: int
    total_us: float  # per call, inside a chain of CALL_CHAIN dependent calls, Python + GPU
    python_us: float  # per call, just building the graph in Python (no GPU)
    error: str | None = None


def call_bench(name: str, kernel: Kernel, config: dict, rows_list=(1, 512), on_progress=print) -> list[CallTiming]:
    """Time one residual-add + rmsnorm call several ways at the model's real shape, to see whether
    the GPU work or the cost of calling the kernel decides the speed inside a model."""
    import numpy as np
    mx, nn, load, _ = _imports()
    on_progress(f"loading {name} for its norm weights")
    model, _ = load(name)
    norm = _layers(model)[0].post_attention_layernorm
    n, dt = norm.weight.shape[0], norm.weight.dtype
    eps = float(getattr(norm, "eps", 1e-5))
    slow, fast = make_fused(mx, kernel, config), make_fused_prebuilt(mx, kernel, config)

    def stock(x, r):
        h = x + r
        return h, mx.fast.rms_norm(h, norm.weight, eps)

    empty = mx.fast.metal_kernel(name="carmen_empty", input_names=["x"], output_names=["out"],
                                 source="if (thread_position_in_grid.x == 0) { out[0] = x[0]; }")
    variants = {
        "MLX: x + 1 (smallest op there is)": lambda x, r: (x + 1, x),
        "MLX: empty custom kernel": lambda x, r: (empty(inputs=[x], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                                                        output_shapes=[x.shape], output_dtypes=[x.dtype])[0], x),
        "stock: x + r, then rms_norm": stock,
        "stock, mx.compile": mx.compile(stock),
        "carmen, as e2e calls it": lambda x, r: slow(x, r, norm),
        "carmen, prebuilt args": lambda x, r: fast(x, r, norm),
        "carmen, prebuilt + mx.compile": mx.compile(lambda x, r: fast(x, r, norm)),
    }
    out = []
    for rows in rows_list:
        x0 = mx.random.normal((1, rows, n)).astype(dt)
        r = (mx.random.normal((1, rows, n)) * 0.1).astype(dt)
        mx.eval(x0, r)
        on_progress(f"timing {len(variants)} ways at {rows} x {n} {str(dt).split('.')[-1]}")
        for label, f in variants.items():
            def chain():
                x, h = x0, None
                for _ in range(CALL_CHAIN):  # each call feeds the next, like layers in a model
                    h, x = f(x, r)
                return [h, x]
            try:
                for _ in range(3):
                    mx.eval(chain())
                py, tot = [], []
                for _ in range(CALL_REPS):
                    t0 = time.perf_counter()
                    outs = chain()
                    t1 = time.perf_counter()
                    mx.eval(outs)
                    t2 = time.perf_counter()
                    py.append((t1 - t0) / CALL_CHAIN)
                    tot.append((t2 - t0) / CALL_CHAIN)
                out.append(CallTiming(label, rows, float(np.median(tot)) * 1e6, float(np.median(py)) * 1e6))
            except Exception as e:  # report it, don't hide it: a variant that can't run is a finding
                out.append(CallTiming(label, rows, float("nan"), float("nan"), f"{type(e).__name__}: {e}"[:160]))
    del model
    return out


def save_calls(name: str, timings: list[CallTiming], kernel_id: str | None, runs_dir: Path) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"calls-{name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"model": name, "kernel": kernel_id, "chain": CALL_CHAIN,
                                "timings": [asdict(t) for t in timings]}, indent=2))
    return path


def apply_kernel(mx, model, fused) -> int:
    """Swap the fused kernel into every layer. Each layer computes h = x + attn and its norm in one
    kernel, then the layer's output and the next layer's input norm in another."""
    layers = list(_layers(model))
    for i, layer in enumerate(layers):
        object.__setattr__(layer, "_carmen_next", layers[i + 1] if i + 1 < len(layers) else None)
        object.__setattr__(layer, "_carmen_pre", None)

        def call(self, x, mask=None, cache=None):
            pre = self._carmen_pre
            normed = pre[1] if pre is not None and pre[0] is x else self.input_layernorm(x)
            object.__setattr__(self, "_carmen_pre", None)
            r = self.self_attn(normed, mask, cache)
            h, n2 = fused(x, r, self.post_attention_layernorm)
            r = self.mlp(n2)
            nxt = self._carmen_next
            if nxt is None:
                return h + r
            out, n_next = fused(h, r, nxt.input_layernorm)
            object.__setattr__(nxt, "_carmen_pre", (out, n_next))
            return out
        _subclass(layer, "Fused", call)
    return len(layers)


def _generate(mx, model, make_prompt_cache, prompt, gen_tokens: int):
    """Greedy: returns (prefill seconds, decode seconds, generated ids, first-step logits)."""
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(prompt, cache=cache)[:, -1, :]
    y = mx.argmax(logits, axis=-1)
    mx.eval(y, logits)
    t1 = time.perf_counter()
    first = logits.astype(mx.float32)
    ids = [int(y.item())]
    for _ in range(gen_tokens - 1):
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
        ids.append(int(y.item()))
    return t1 - t0, time.perf_counter() - t1, ids, first


def run(name: str, modes=MODES, kernel: Kernel | None = None, config: dict | None = None,
        prompt_tokens: int = 512, gen_tokens: int = 128, repeats: int = 3, on_progress=print) -> list[Result]:
    mx, nn, load, make_prompt_cache = _imports()
    results, stock_ids, stock_logits = [], None, None
    for mode in modes:
        on_progress(f"{mode}: loading {name}")
        model, tokenizer = load(name)
        ids = tokenizer.encode(PROMPT_TEXT * (prompt_tokens // 20 + 1))[:prompt_tokens]
        prompt = mx.array(ids)[None]
        if mode in ("plumbing", "both"):
            on_progress(f"{mode}: merged gate+up in {apply_plumbing(mx, nn, model)} layers")
        if mode in ("kernel", "both"):
            if kernel is None:
                raise SystemExit("no residual_rmsnorm kernel: run `carmen run residual_rmsnorm` or pass --kernel golden")
            on_progress(f"{mode}: carmen kernel in {apply_kernel(mx, model, make_fused(mx, kernel, config))} layers")
        _generate(mx, model, make_prompt_cache, prompt, 8)  # warm up: compile kernels, fill caches
        best = None
        for _ in range(repeats):
            p, d, out_ids, logits = _generate(mx, model, make_prompt_cache, prompt, gen_tokens)
            if best is None or d < best[1]:
                best = (p, d, out_ids, logits)
        p, d, out_ids, logits = best
        r = Result(mode, prompt_tokens / p, (gen_tokens - 1) / d, tokens_total=gen_tokens)
        if mode == "stock":
            stock_ids, stock_logits = out_ids, logits
        elif stock_ids is not None:
            r.tokens_match = next((i for i, (a, b) in enumerate(zip(out_ids, stock_ids)) if a != b), len(out_ids))
            r.max_logit_diff = float(mx.abs(logits - stock_logits).max().item())
            r.same_first_token = out_ids[0] == stock_ids[0]
        results.append(r)
        on_progress(f"{mode}: prefill {r.prefill_tps:,.0f} tok/s, decode {r.decode_tps:,.1f} tok/s")
        del model
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return results


def save(name: str, results: list[Result], kernel_id: str | None, runs_dir: Path) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"e2e-{name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"model": name, "kernel": kernel_id, "results": [asdict(r) for r in results]}, indent=2))
    return path
