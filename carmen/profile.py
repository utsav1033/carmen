"""`carmen profile <model>`: where does a real model spend its time on this Mac?

Loads a model with mlx-lm, measures prefill and decode speed end to end, then times
every step of one transformer layer in isolation at the model's real shapes:
1 token for decode (writing a word), the prompt length for prefill (reading your
prompt). The breakdown says which steps are worth a carmen kernel.

Steps are timed `INNER` at a time inside one eval, so the number is the cost of the
step inside a running model (encoding + GPU time), not a sync per op.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROMPT_TEXT = ("The history of computing is a story of people finding ways to make machines do more "
               "with less: smaller transistors, cleverer algorithms, and better tools for thinking. ")

INNER = 20
REPS = 30

# Which steps carmen already has a kernel family for (op name in carmen.ops).
CARMEN_OPS = {"rmsnorm": "rmsnorm", "add_rmsnorm": "add_rmsnorm", "attention": "attention"}


@dataclass
class Spec:
    model: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    mlp: int
    vocab: int
    norm_eps: float | None
    dtype: str
    bits: int | None
    group_size: int | None


@dataclass
class Step:
    name: str
    kind: str
    us: float  # microseconds per occurrence (one layer, or once per token for the head)
    per_token: int = 1  # how many times it runs per token: layers for per-layer steps, 1 for the head


@dataclass
class Profile:
    spec: Spec
    prompt_tokens: int
    gen_tokens: int
    prefill_tps: float
    decode_tps: float
    decode: list[Step] = field(default_factory=list)
    prefill: list[Step] = field(default_factory=list)
    kernel_floor_us: float = 0.0


def table(steps: list[Step], measured_ms: float) -> list[dict]:
    """Share of the measured time per step. Steps are timed in isolation, so the total can
    differ from the end-to-end time; `coverage` says how much of it the steps explain."""
    total_us = sum(s.us * s.per_token for s in steps)
    rows = []
    for s in sorted(steps, key=lambda s: -s.us * s.per_token):
        t = s.us * s.per_token
        rows.append({"step": s.name, "kind": s.kind, "us_each": s.us, "times": s.per_token,
                     "ms_total": t / 1e3, "share": t / total_us if total_us else 0.0,
                     "carmen": CARMEN_OPS.get(s.kind)})
    coverage = (total_us / 1e3) / measured_ms if measured_ms else None
    return rows + [{"step": "(sum of steps)", "ms_total": total_us / 1e3, "coverage": coverage}]


def by_kind(steps: list[Step]) -> dict[str, float]:
    """Share of time per kind of step (matmul, norm, attention, ...)."""
    total = sum(s.us * s.per_token for s in steps) or 1.0
    out: dict[str, float] = {}
    for s in steps:
        out[s.kind] = out.get(s.kind, 0.0) + s.us * s.per_token / total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def save(p: Profile, runs_dir: Path) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    name = p.spec.model.split("/")[-1]
    path = runs_dir / f"profile-{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(asdict(p), indent=2))
    return path


# ── the part that needs MLX and mlx-lm ─────────────────────────────────────────────
def _imports():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as e:  # pragma: no cover - only on machines without MLX
        raise SystemExit("carmen profile needs MLX and mlx-lm on Apple Silicon: pip install 'carmen[models]'") from e
    return mx, nn, load, make_prompt_cache


def _layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise SystemExit("unsupported architecture: expected model.model.layers (Llama, Qwen, Mistral style)")
    return inner, layers


def read_spec(model, name: str) -> Spec:
    args = vars(getattr(model, "args", object())) if hasattr(model, "args") else {}
    _, layers = _layers(model)
    layer = layers[0]
    attn, norm = layer.self_attn, layer.input_layernorm
    heads = args.get("num_attention_heads") or getattr(attn, "n_heads", 0)
    hidden = args.get("hidden_size") or norm.weight.shape[0]
    return Spec(
        model=name, hidden=hidden, layers=len(layers), heads=heads,
        kv_heads=args.get("num_key_value_heads") or getattr(attn, "n_kv_heads", heads),
        head_dim=args.get("head_dim") or hidden // max(heads, 1),
        mlp=args.get("intermediate_size") or 0, vocab=args.get("vocab_size") or 0,
        norm_eps=args.get("rms_norm_eps"), dtype=str(norm.weight.dtype).replace("mlx.core.", ""),
        bits=getattr(attn.q_proj, "bits", None), group_size=getattr(attn.q_proj, "group_size", None))


def _timer(mx):
    import numpy as np

    def measure(fn) -> float:
        for _ in range(3):
            mx.eval(fn())
        ts = []
        for _ in range(REPS):
            t = time.perf_counter()
            mx.eval([fn() for _ in range(INNER)])
            ts.append((time.perf_counter() - t) / INNER)
        return float(np.median(ts)) * 1e6
    return measure


def _layer_steps(mx, nn, model, spec: Spec, tokens: int, context: int) -> list[Step]:
    """Time each step of layer 0 for `tokens` new tokens attending over `context` cached ones."""
    measure = _timer(mx)
    inner, layers = _layers(model)
    L = layers[0]
    A, M = L.self_attn, L.mlp
    dt = L.input_layernorm.weight.dtype
    x = mx.random.normal((1, tokens, spec.hidden)).astype(dt)
    x2 = mx.random.normal((1, tokens, spec.hidden)).astype(dt)
    qh = mx.random.normal((1, spec.heads, tokens, spec.head_dim)).astype(dt)
    kh = mx.random.normal((1, spec.kv_heads, tokens, spec.head_dim)).astype(dt)
    kc = mx.random.normal((1, spec.kv_heads, context, spec.head_dim)).astype(dt)
    vc = mx.random.normal((1, spec.kv_heads, context, spec.head_dim)).astype(dt)
    ao = mx.random.normal((1, tokens, spec.heads * spec.head_dim)).astype(dt)
    g = M.gate_proj(x)
    u = M.up_proj(x)
    h = (nn.silu(g) * u).astype(dt)
    mx.eval(x, x2, qh, kh, kc, vc, ao, g, u, h)
    scale = getattr(A, "scale", spec.head_dim ** -0.5)
    mask = "causal" if tokens == context else None
    n = spec.layers

    steps = [
        Step("rmsnorm before attention", "rmsnorm", measure(lambda: L.input_layernorm(x)), n),
        Step("q, k, v projections", "matmul", measure(lambda: [A.q_proj(x), A.k_proj(x), A.v_proj(x)]), n),
        Step("rope on q and k", "rope", measure(lambda: [A.rope(qh, offset=context - tokens),
                                                        A.rope(kh, offset=context - tokens)]), n),
        Step(f"attention over {context} tokens", "attention",
             measure(lambda: mx.fast.scaled_dot_product_attention(qh, kc, vc, scale=scale, mask=mask)), n),
        Step("output projection", "matmul", measure(lambda: A.o_proj(ao)), n),
        Step("residual add + rmsnorm", "add_rmsnorm", measure(lambda: L.post_attention_layernorm(x + x2)), n),
        Step("mlp gate and up projections", "matmul", measure(lambda: [M.gate_proj(x), M.up_proj(x)]), n),
        Step("silu(gate) x up", "elementwise", measure(lambda: nn.silu(g) * u), n),
        Step("mlp down projection", "matmul", measure(lambda: M.down_proj(h)), n),
        Step("residual add", "elementwise", measure(lambda: x + x2), n),
    ]
    head = getattr(model, "lm_head", None)
    head_fn = head if head is not None else getattr(inner.embed_tokens, "as_linear", None)
    if head_fn is not None:
        last = x[:, -1:, :]
        steps.append(Step("final norm + vocabulary projection", "matmul",
                          measure(lambda: head_fn(inner.norm(last))), 1))
    return steps


def run(name: str, prompt_tokens: int = 512, gen_tokens: int = 128, on_progress=print) -> Profile:
    mx, nn, load, make_prompt_cache = _imports()
    on_progress(f"loading {name} (downloads on first use)")
    model, tokenizer = load(name)
    spec = read_spec(model, name)

    ids = tokenizer.encode(PROMPT_TEXT * (prompt_tokens // 20 + 1))[:prompt_tokens]
    prompt = mx.array(ids)[None]

    def generate_once():
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = model(prompt, cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        t1 = time.perf_counter()
        for _ in range(gen_tokens):
            logits = model(y[:, None], cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    on_progress("warming up, then timing prefill and decode")
    generate_once()
    prefill_s, decode_s = generate_once()

    on_progress("timing every step of one layer at decode and prefill shapes")
    context = prompt_tokens + gen_tokens // 2
    decode = _layer_steps(mx, nn, model, spec, 1, context)
    prefill = _layer_steps(mx, nn, model, spec, prompt_tokens, prompt_tokens)

    tiny = mx.zeros((1,))
    mx.eval(tiny)
    floor = _timer(mx)(lambda: tiny + 1)

    return Profile(spec, prompt_tokens, gen_tokens, prompt_tokens / prefill_s, gen_tokens / decode_s,
                   decode, prefill, floor)
