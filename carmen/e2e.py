"""`carmen e2e <model>`: does a real model get faster, with the same answers?

Runs one model several ways and compares each against stock MLX:

  stock     the model as mlx-lm loads it
  plumbing  gate and up projections merged into one matmul per layer (weights concatenated once)
  compile   the part of each layer after attention (residual add, norm, MLP, residual add) built
            once with mx.compile instead of op by op in Python every word; attention stays as is
            (its KV cache grows every word, which mx.compile can't follow)
  kernel    carmen's residual_rmsnorm swapped into every layer: the residual add and the norm
            after it become one kernel, twice per layer, so 4 small kernels per layer become 2
  all       plumbing + compile + kernel
Modes combine with +, e.g. plumbing+compile.

Every version is loaded up front and they take turns (stock, plumbing, ..., then again), so a
Mac that heats up or slows down mid-run hits every mode alike. Each mode reports its median and
its range over the turns; a gain is real only when the ranges don't overlap.

Also reports the decode speed limit: every word reads every weight once, so tokens/s can't beat
(measured memory bandwidth / weight bytes). The gap between that and stock is what's left to win.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import Kernel
from .profile import PROMPT_TEXT

MODES = ("stock", "plumbing", "compile", "kernel", "all")
PARTS = ("plumbing", "compile", "kernel")
OP = "residual_rmsnorm"


def parts(mode: str) -> set[str]:
    """Which changes a mode applies: 'all' is every one, 'both' is plumbing+kernel, else split on +."""
    if mode == "stock":
        return set()
    if mode == "all":
        return set(PARTS)
    if mode == "both":
        return {"plumbing", "kernel"}
    got = set(mode.split("+"))
    bad = got - set(PARTS)
    if bad:
        raise SystemExit(f"unknown e2e mode part(s) {sorted(bad)}: use stock, all, or {'+'.join(PARTS)} combined with +")
    return got


@dataclass
class Result:
    mode: str
    prefill_tps: float  # median over the turns
    decode_tps: float
    tokens_match: int | None = None  # how many generated tokens equal stock's, from the start
    tokens_total: int = 0
    max_logit_diff: float | None = None  # first-step logits vs stock
    same_first_token: bool | None = None
    decode_lo: float | None = None  # slowest and fastest turn
    decode_hi: float | None = None
    prefill_lo: float | None = None
    prefill_hi: float | None = None
    turns: int = 1
    weight_bytes: int | None = None  # stock only: bytes every decoded word must read


def beyond_noise(r: Result, stock: Result) -> str:
    """'faster' or 'slower' only when this mode's range clears stock's; else 'within noise'."""
    if None in (r.decode_lo, r.decode_hi, stock.decode_lo, stock.decode_hi):
        return "unknown"
    if r.decode_lo > stock.decode_hi:
        return "faster"
    if r.decode_hi < stock.decode_lo:
        return "slower"
    return "within noise"


def speed_limit_tps(weight_bytes: int, peak_gbps: float) -> float:
    return peak_gbps * 1e9 / weight_bytes


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


def make_raw(mx, kernel: Kernel, config: dict):
    """carmen's residual_rmsnorm as a plain function (x, res, w, eps) -> (h, y). Everything that
    depends only on the shape (template, grid, eps array) is built once per shape."""
    from . import ops
    op = ops.get(OP)
    k = mx.fast.metal_kernel(name=f"carmen_{OP}_{kernel.digest()}", input_names=list(op.input_names),
                             output_names=["out"], source=kernel.source, header=kernel.header)
    plans: dict = {}

    def raw(x, res, w, eps: float):
        key = (x.shape, x.dtype, eps)
        p = plans.get(key)
        if p is None:
            n = x.shape[-1]
            rows = x.size // n
            grid, tg = op.grid(rows, n, config)
            p = plans[key] = (rows, n, mx.array([eps], dtype=mx.float32), [("T", x.dtype)] + list(config.items()),
                              grid, tg, [(2, rows, n)], [x.dtype])
        rows, n, eps_arr, template, grid, tg, out_shapes, out_dtypes = p
        (out,) = k(inputs=[x.reshape(rows, n), res.reshape(rows, n), w, eps_arr], template=template,
                   grid=grid, threadgroup=tg, output_shapes=out_shapes, output_dtypes=out_dtypes)
        return out[0].reshape(x.shape), out[1].reshape(x.shape)
    return raw


def make_fused(mx, kernel: Kernel, config: dict, compiled: bool = True):
    """(x, res, norm) -> (h, y). Compiled by default: `carmen e2e --calls` measured the plain call
    at 28.5 us vs 17.3 us compiled at decode (Qwen 0.5B, M4), from the reshapes and slices around it."""
    raw = make_raw(mx, kernel, config)
    per_eps: dict[float, object] = {}

    def fused(x, res, norm):
        eps = float(getattr(norm, "eps", 1e-5))
        if not compiled:
            return raw(x, res, norm.weight, eps)
        f = per_eps.get(eps)
        if f is None:
            f = per_eps[eps] = mx.compile(lambda x_, r_, w_: raw(x_, r_, w_, eps))
        return f(x, res, norm.weight)
    return fused


def make_regime_fused(mx, champions: dict[str, tuple[Kernel, dict]]):
    """One fused function that calls the decode champion for 1-row inputs and the prefill
    champion otherwise: each use case runs the kernel that won it."""
    fns = {name: make_fused(mx, k, cfg) for name, (k, cfg) in champions.items()}
    one, many = fns.get("decode") or fns["prefill"], fns.get("prefill") or fns["decode"]

    def fused(x, res, norm):
        return (one if x.size == x.shape[-1] else many)(x, res, norm)
    return fused


def latest_regime_champions(runs_dir: Path, op: str = OP) -> tuple[dict[str, tuple[Kernel, dict]], str, dict] | None:
    """Per-regime champions of the newest targeted run of `op`: ({regime: (kernel, config)}, run id, summary)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    for d in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        ev, sm = d / "events.jsonl", d / "summary.json"
        if not ev.exists() or not sm.exists():
            continue
        summary = json.loads(sm.read_text())
        if json.loads(ev.read_text().splitlines()[0]).get("op") != op or not summary.get("regime_champions"):
            continue
        champs = {name: (Kernel.from_json(json.loads((d / "attempts" / c["attempt"] / "kernel.json").read_text())),
                         c["config"]) for name, c in summary["regime_champions"].items()}
        return champs, d.name, summary
    return None


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


WARMUP_S = 0.5  # Apple GPUs clock up under sustained load; short bursts run at whatever clock they find


def _chain_once(mx, f, x0, r) -> tuple[float, float]:
    """(total us, python us) per call, over CALL_CHAIN dependent calls like one word of decode."""
    x, h = x0, None
    t0 = time.perf_counter()
    for _ in range(CALL_CHAIN):  # each call feeds the next, like layers in a model
        h, x = f(x, r)
    t1 = time.perf_counter()
    mx.eval([h, x])
    t2 = time.perf_counter()
    return (t2 - t0) / CALL_CHAIN * 1e6, (t1 - t0) / CALL_CHAIN * 1e6


def _warm(mx, fns, x0, r, seconds: float = WARMUP_S) -> None:
    """Run every variant until the GPU has been busy for `seconds`: compiles kernels and lifts the clock."""
    end = time.perf_counter() + seconds
    while True:
        for f in fns:
            _chain_once(mx, f, x0, r)
        if time.perf_counter() > end:
            return


def _chain_timer(mx):
    import numpy as np

    def measure(f, x0, r) -> tuple[float, float]:
        _warm(mx, [f], x0, r, seconds=0.1)
        tot, py = zip(*[_chain_once(mx, f, x0, r) for _ in range(CALL_REPS)])
        return float(np.median(tot)), float(np.median(py))
    return measure


def _model_norm(name: str, on_progress):
    mx, nn, load, _ = _imports()
    on_progress(f"loading {name} for its norm weights")
    model, _ = load(name)
    norm = _layers(model)[0].post_attention_layernorm
    del model
    return mx, norm


def _inputs(mx, rows: int, norm):
    n, dt = norm.weight.shape[0], norm.weight.dtype
    x0 = mx.random.normal((1, rows, n)).astype(dt)
    r = (mx.random.normal((1, rows, n)) * 0.1).astype(dt)
    mx.eval(x0, r)
    return x0, r


def _stock(mx, norm):
    eps = float(getattr(norm, "eps", 1e-5))

    def stock(x, r):
        h = x + r
        return h, mx.fast.rms_norm(h, norm.weight, eps)
    return stock


def call_bench(name: str, kernel: Kernel, config: dict, rows_list=(1, 512), on_progress=print) -> list[CallTiming]:
    """Time one residual-add + rmsnorm call several ways at the model's real shape, to see whether
    the GPU work or the cost of calling the kernel decides the speed inside a model."""
    mx, norm = _model_norm(name, on_progress)
    measure = _chain_timer(mx)
    stock = _stock(mx, norm)
    warmed = False
    plain, comp = make_fused(mx, kernel, config, compiled=False), make_fused(mx, kernel, config)
    empty = mx.fast.metal_kernel(name="carmen_empty", input_names=["x"], output_names=["out"],
                                 source="if (thread_position_in_grid.x == 0) { out[0] = x[0]; }")
    variants = {
        "MLX: x + 1 (smallest op there is)": lambda x, r: (x + 1, x),
        "MLX: empty custom kernel": lambda x, r: (empty(inputs=[x], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                                                        output_shapes=[x.shape], output_dtypes=[x.dtype])[0], x),
        "stock: x + r, then rms_norm": stock,
        "stock, mx.compile": mx.compile(stock),
        "carmen, plain call": lambda x, r: plain(x, r, norm),
        "carmen, compiled call (what e2e uses)": lambda x, r: comp(x, r, norm),
    }
    out = []
    for rows in rows_list:
        x0, r = _inputs(mx, rows, norm)
        on_progress(f"timing {len(variants)} ways at {rows} x {x0.shape[-1]} {str(x0.dtype).split('.')[-1]}")
        if not warmed:
            _warm(mx, [stock], x0, r)
            warmed = True
        for label, f in variants.items():
            try:
                out.append(CallTiming(label, rows, *measure(f, x0, r)))
            except Exception as e:  # report it, don't hide it: a variant that can't run is a finding
                out.append(CallTiming(label, rows, float("nan"), float("nan"), f"{type(e).__name__}: {e}"[:160]))
    return out


HOLDOUT_TURNS = 15


@dataclass
class Holdout:
    regime: str
    rows: int
    stock_us: float  # medians over the turns
    compiled_us: float
    carmen_us: float
    judge_vs_compiled: float  # what the judge measured for this regime's champion
    ratio_lo: float | None = None  # middle half of the per-turn (compiled / carmen) ratios
    ratio_hi: float | None = None
    stock_spread: float = 1.0  # stock's own p75 / p25 across turns: how steady the ruler was

    @property
    def vs_compiled(self) -> float:
        return self.compiled_us / self.carmen_us

    @property
    def vs_stock(self) -> float:
        return self.stock_us / self.carmen_us

    @property
    def verdict(self) -> str:
        """unsteady: stock alone moved >15%, so nothing can be concluded. tie: in the model it's within
        5% of compiled stock, or its range spans 1.0. holds: same side of 1.0 as the judge and within 25%.
        Otherwise the judge was wrong."""
        if self.stock_spread > 1.15:
            return "unsteady"
        m, j = self.vs_compiled, self.judge_vs_compiled
        if abs(m - 1) <= 0.05 or (self.ratio_lo is not None and self.ratio_lo <= 1 <= self.ratio_hi):
            return "tie"
        return "holds" if (j >= 1) == (m >= 1) and abs(m / j - 1) <= 0.25 else "disagrees"

    @property
    def agrees(self) -> bool:
        return self.verdict == "holds"


def holdout(name: str, champions: dict[str, tuple[Kernel, dict, int, float]], on_progress=print) -> list[Holdout]:
    """The frozen final check: each regime's champion called the way the model calls it (compiled,
    48 calls chained, real norm weights and dtype), vs stock and mx.compile(stock). Carmy never sees
    this number, so it can't be optimized against; it only says whether the judge's score holds.
    The GPU is warmed first and the three take turns, so clock changes hit all of them alike."""
    import numpy as np
    mx, norm = _model_norm(name, on_progress)
    stock = _stock(mx, norm)
    stock_c = mx.compile(stock)
    out = []
    for regime, (kernel, config, rows, judged) in champions.items():
        x0, r = _inputs(mx, rows, norm)
        f = make_fused(mx, kernel, config)
        fns = {"stock": stock, "compiled": stock_c, "carmen": lambda x, r_, f=f: f(x, r_, norm)}
        on_progress(f"in-model check: {regime} champion at {rows} x {x0.shape[-1]}, {HOLDOUT_TURNS} turns")
        _warm(mx, list(fns.values()), x0, r)
        t = {k: [] for k in fns}
        names = list(fns)
        for turn in range(HOLDOUT_TURNS):
            for k in names[turn % 3:] + names[:turn % 3]:
                t[k].append(_chain_once(mx, fns[k], x0, r)[0])
        ratios = np.array(t["compiled"]) / np.array(t["carmen"])
        p25, p75 = np.percentile(t["stock"], [25, 75])
        out.append(Holdout(regime, rows, float(np.median(t["stock"])), float(np.median(t["compiled"])),
                           float(np.median(t["carmen"])), judged, float(np.percentile(ratios, 25)),
                           float(np.percentile(ratios, 75)), float(p75 / p25)))
    return out


def save_calls(name: str, timings: list[CallTiming], kernel_id: str | None, runs_dir: Path) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"calls-{name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"model": name, "kernel": kernel_id, "chain": CALL_CHAIN,
                                "timings": [asdict(t) for t in timings]}, indent=2))
    return path


def apply_layers(mx, model, fused=None, compile_tail: bool = False) -> int:
    """Rewrite what each layer does after attention. With `fused`, the residual add + norm become
    carmen's kernel (h = x + attn and its norm in one call, then the layer's output and the next
    layer's input norm in another). With `compile_tail`, that whole tail (adds, norms, MLP) is built
    once by mx.compile. Attention itself is untouched."""
    layers = list(_layers(model))
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        if fused is None:
            def tail(x, r, _l=layer):
                h = x + r
                return [h + _l.mlp(_l.post_attention_layernorm(h))]
        else:
            def tail(x, r, _l=layer, _n=nxt):
                h, n2 = fused(x, r, _l.post_attention_layernorm)
                r2 = _l.mlp(n2)
                return [h + r2] if _n is None else list(fused(h, r2, _n.input_layernorm))
        object.__setattr__(layer, "_carmen_tail", mx.compile(tail) if compile_tail else tail)
        object.__setattr__(layer, "_carmen_next", nxt if fused is not None else None)
        object.__setattr__(layer, "_carmen_pre", None)

        def call(self, x, mask=None, cache=None):
            pre = self._carmen_pre
            normed = pre[1] if pre is not None and pre[0] is x else self.input_layernorm(x)
            object.__setattr__(self, "_carmen_pre", None)
            out = self._carmen_tail(x, self.self_attn(normed, mask, cache))
            if len(out) == 2:
                object.__setattr__(self._carmen_next, "_carmen_pre", (out[0], out[1]))
            return out[0]
        _subclass(layer, "Tail", call)
    return len(layers)


def apply_kernel(mx, model, fused) -> int:
    return apply_layers(mx, model, fused)


def weight_bytes(model) -> int:
    """Bytes one decoded word must read: every weight once. An untied input embedding is a lookup
    of one row, not a full read, so it doesn't count; a tied one is read in full as the output head."""
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    untied = getattr(model, "lm_head", None) is not None
    return sum(v.nbytes for k, v in params if not (untied and "embed_tokens" in k))


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
        prompt_tokens: int = 512, gen_tokens: int = 128, repeats: int = 5, on_progress=print,
        champions: dict[str, tuple[Kernel, dict]] | None = None) -> list[Result]:
    """`champions` ({regime: (kernel, config)}) runs each regime's own champion; else `kernel` everywhere.
    All modes are loaded at once and take `repeats` turns each, in rotating order."""
    import numpy as np
    mx, nn, load, make_prompt_cache = _imports()
    modes = list(modes)
    for m in modes:
        parts(m)  # fail on a typo before loading anything
    if any("kernel" in parts(m) for m in modes) and kernel is None and not champions:
        raise SystemExit("no residual_rmsnorm kernel: run `carmen run residual_rmsnorm` or pass --kernel golden")
    loaded, prompt, wbytes = {}, None, None
    for mode in modes:
        on_progress(f"{mode}: loading {name}")
        model, tokenizer = load(name)
        if prompt is None:
            ids = tokenizer.encode(PROMPT_TEXT * (prompt_tokens // 20 + 1))[:prompt_tokens]
            prompt = mx.array(ids)[None]
        if mode == "stock":
            wbytes = weight_bytes(model)
        p = parts(mode)
        if "plumbing" in p:
            on_progress(f"{mode}: merged gate+up in {apply_plumbing(mx, nn, model)} layers")
        if "kernel" in p or "compile" in p:
            fused = None
            if "kernel" in p:
                fused = make_regime_fused(mx, champions) if champions else make_fused(mx, kernel, config)
            n = apply_layers(mx, model, fused, compile_tail="compile" in p)
            on_progress(f"{mode}: " + " + ".join(x for x in (("carmen kernel" if fused else ""),
                                                             ("compiled layer tails" if "compile" in p else "")) if x)
                        + f" in {n} layers")
        _generate(mx, model, make_prompt_cache, prompt, 8)  # warm up: compile kernels, fill caches
        loaded[mode] = model

    on_progress(f"taking turns: {repeats} x {len(modes)} runs of {prompt_tokens} prompt + {gen_tokens} new tokens")
    samples = {m: [] for m in modes}
    for turn in range(repeats):
        for mode in modes[turn % len(modes):] + modes[:turn % len(modes)]:
            samples[mode].append(_generate(mx, loaded[mode], make_prompt_cache, prompt, gen_tokens))

    results, stock = [], None
    for mode in modes:
        pre = [prompt_tokens / s[0] for s in samples[mode]]
        dec = [(gen_tokens - 1) / s[1] for s in samples[mode]]
        r = Result(mode, float(np.median(pre)), float(np.median(dec)), tokens_total=gen_tokens,
                   decode_lo=min(dec), decode_hi=max(dec), prefill_lo=min(pre), prefill_hi=max(pre),
                   turns=repeats, weight_bytes=wbytes if mode == "stock" else None)
        out_ids, logits = samples[mode][0][2], samples[mode][0][3]
        if mode == "stock":
            stock = (out_ids, logits)
        elif stock is not None:
            r.tokens_match = next((i for i, (a, b) in enumerate(zip(out_ids, stock[0])) if a != b), len(out_ids))
            r.max_logit_diff = float(mx.abs(logits - stock[1]).max().item())
            r.same_first_token = out_ids[0] == stock[0][0]
        results.append(r)
        on_progress(f"{mode}: prefill {r.prefill_tps:,.0f} tok/s, decode {r.decode_tps:,.1f} tok/s "
                    f"({r.decode_lo:,.1f}-{r.decode_hi:,.1f})")
    del loaded
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return results


def save(name: str, results: list[Result], kernel_id: str | None, runs_dir: Path, extra: dict | None = None) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"e2e-{name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"model": name, "kernel": kernel_id, "results": [asdict(r) for r in results],
                                **(extra or {})}, indent=2))
    return path
