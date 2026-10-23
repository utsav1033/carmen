"""`carmen levers <model>`: what makes a model faster once kernels can't.

On a 3B model stock MLX already decodes at ~91% of the bandwidth speed limit (every word reads every
weight once), so faster kernels have ~9% left to win. What's left is reading fewer bytes per word:

  stock   the model as mlx-lm loads it
  spec    speculative decoding: a small draft model of the same family guesses a few words, the model
          checks them all in one pass, so its weights are read once for several words. Greedy, so the
          answers must match stock (fp16 rounding in the batched check can flip a late near-tie).
  kv8     the KV cache stored in 8 bits from the first word. Lossy.
  kv4     the KV cache stored in 4 bits. Lossy.
  w3      the weights requantized to 3 bits. No 3-bit release is published, so they come from the
          8-bit one (dequantize, then quantize). Lossy.

Lossy levers are scored for drift, not just speed: stock's own answer is fed back through each one
(teacher forcing), and we report how often its top word agrees with stock's and the mean KL divergence
of its next-word distribution from stock's.

Context length is the third lever's axis: at long prompts the KV cache, not the weights, is what each
word reads most, which is where kv8/kv4 pay. `--prompt 512,8192` runs both and compares.

The prompt is Python's own documentation (pydoc_data, in every Python install), not a repeated
sentence: a draft model can guess repeated text perfectly, which would flatter spec.

Same method as `carmen e2e`: every lever is loaded up front, they take turns in rotating order, and a
gain counts only when the 95% range of the per-turn ratio to stock excludes 1.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import e2e

LEVERS = ("stock", "spec", "kv8", "kv4", "w3")
KV_BITS = {"kv8": 8, "kv4": 4}
PREFILL_STEP = 2048  # long prompts go in chunks, like mlx-lm does, so 32k tokens fit in memory
DRAFTS = {"Qwen2.5-": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
          "Llama-3.2-": "mlx-community/Llama-3.2-1B-Instruct-4bit"}


@dataclass
class LeverResult(e2e.Result):
    context: int = 0  # prompt tokens
    accept_rate: float | None = None  # spec: share of words that came from the draft
    top1: float | None = None  # lossy levers: share of positions whose top word matches stock's
    kl: float | None = None  # lossy levers: mean KL(stock || lever) of the next-word distribution, nats
    kv_bytes: int | None = None  # KV cache each decoded word reads at this context


def default_draft(name: str) -> str | None:
    """The smallest model of the same family, if we know one and it isn't the model itself."""
    for family, draft in DRAFTS.items():
        if family in name and draft != name:
            return draft
    return None


def eight_bit(name: str) -> str | None:
    return name[:-len("-4bit")] + "-8bit" if name.endswith("-4bit") else None


def prompt_text() -> str:
    from pydoc_data.topics import topics
    return "\n\n".join(topics[k] for k in sorted(topics))


def prompt_ids(tokenizer, n: int) -> list[int]:
    text = prompt_text()
    ids = tokenizer.encode(text[:n * 8])  # ~4 characters a token: plenty, without encoding all of it
    if len(ids) < n:
        raise SystemExit(f"--prompt {n} is longer than the bundled text ({len(ids):,} tokens)")
    return ids[:n]


def requantize(mx, nn, model, bits: int = 3, group_size: int = 64) -> int:
    """Rewrite every quantized layer at `bits`: dequantize what's there, quantize again. Returns how many."""
    n = 0
    for _, m in model.named_modules():
        if not isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
        m.weight, m.scales, m.biases = mx.quantize(w, group_size=group_size, bits=bits)
        m.bits, m.group_size = bits, group_size
        n += 1
    mx.eval(model.parameters())
    return n


def _prefill(mx, model, cache, prompt):
    """All prompt tokens through the model in chunks; returns the last position's logits."""
    for start in range(0, prompt.shape[1], PREFILL_STEP):
        logits = model(prompt[:, start:start + PREFILL_STEP], cache=cache)
        mx.eval([c.state for c in cache])
    return logits[:, -1, :]


def _quantize_cache(cache, bits: int | None) -> None:
    if bits is None:
        return
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized"):
            cache[i] = c.to_quantized(group_size=64, bits=bits)


def _plain(mx, model, make_prompt_cache, prompt, gen: int, kv_bits: int | None = None):
    """Greedy, one word at a time: (prefill s, decode s, ids, None)."""
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = _prefill(mx, model, cache, prompt)
    _quantize_cache(cache, kv_bits)
    y = mx.argmax(logits, axis=-1)
    mx.eval(y, [c.state for c in cache])
    t1 = time.perf_counter()
    ids = [int(y.item())]
    for _ in range(gen - 1):
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
        ids.append(int(y.item()))
    return t1 - t0, time.perf_counter() - t1, ids, None


def _spec(mx, model, draft, prompt, gen: int, draft_tokens: int):
    """Greedy speculative decoding via mlx-lm: (prefill s, decode s, ids, words from the draft)."""
    from mlx_lm.generate import speculative_generate_step
    t0 = time.perf_counter()
    steps = speculative_generate_step(prompt[0], model, draft, num_draft_tokens=draft_tokens,
                                      max_tokens=gen, prefill_step_size=PREFILL_STEP)
    tok, _, _ = next(steps)  # prefill of both models + the first word
    t1 = time.perf_counter()
    ids, drafted = [int(tok)], 0
    for tok, _, from_draft in steps:
        ids.append(int(tok))
        drafted += bool(from_draft)
    return t1 - t0, time.perf_counter() - t1, ids, drafted


def _teacher_forced(mx, model, make_prompt_cache, prompt, answer: list[int], kv_bits: int | None = None):
    """Log-probabilities the model gives each word of `answer`, with stock's answer fed in."""
    cache = make_prompt_cache(model)
    if prompt.shape[1] > 1:
        _prefill(mx, model, cache, prompt[:, :-1])
    _quantize_cache(cache, kv_bits)
    x = mx.array([int(prompt[0, -1].item())] + answer[:-1])[None]
    logits = model(x, cache=cache)[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def drift(mx, ref, got) -> tuple[float, float]:
    """(top-1 agreement, mean KL(ref || got)) over positions, from two log-probability arrays."""
    top1 = float((mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1)).astype(mx.float32).mean().item())
    kl = float((mx.exp(ref) * (ref - got)).sum(axis=-1).mean().item())
    return top1, kl


def run(name: str, levers=LEVERS, contexts=(512,), gen_tokens: int = 128, repeats: int = 10,
        draft: str | None = None, draft_tokens: int = 3, on_progress=print) -> list[LeverResult]:
    """Every lever at every context length, paired turns, plus drift for the lossy ones."""
    import numpy as np
    mx, nn, load, make_prompt_cache = e2e._imports()
    levers = list(levers)
    bad = set(levers) - set(LEVERS)
    if bad:
        raise SystemExit(f"unknown lever(s) {sorted(bad)}: use {', '.join(LEVERS)}")
    if "stock" not in levers:
        levers.insert(0, "stock")
    draft = draft or default_draft(name)
    if "spec" in levers and not draft:
        raise SystemExit(f"no draft model known for {name}: pass --draft <smaller model of the same family>")
    source = eight_bit(name)
    if "w3" in levers and not source:
        raise SystemExit("w3 needs the 8-bit release of the model to requantize from; pass a ...-4bit model name")

    on_progress(f"loading {name}")
    model, tokenizer = load(name)
    wbytes = e2e.weight_bytes(model)
    draft_model = w3_model = None
    if "spec" in levers:
        on_progress(f"spec: loading draft {draft}")
        draft_model, _ = load(draft)
    if "w3" in levers:
        on_progress(f"w3: loading {source} and requantizing to 3 bits")
        w3_model, _ = load(source)
        on_progress(f"w3: {requantize(mx, nn, w3_model)} layers now 3-bit, "
                    f"{e2e.weight_bytes(w3_model) / 1e9:.2f} GB per word vs {wbytes / 1e9:.2f}")

    def once(lever, prompt):
        if lever == "spec":
            return _spec(mx, model, draft_model, prompt, gen_tokens, draft_tokens)
        return _plain(mx, w3_model if lever == "w3" else model, make_prompt_cache, prompt, gen_tokens,
                      KV_BITS.get(lever))

    results = []
    for ctx in contexts:
        prompt = mx.array(prompt_ids(tokenizer, ctx))[None]
        kvb = {bits: cache_bytes(mx, model, make_prompt_cache, prompt, bits)
               for bits in {KV_BITS.get(lv) for lv in levers}}
        for lever in levers:  # warm up: compile kernels, fill caches
            once(lever, prompt[:, :64])
        on_progress(f"context {ctx:,}: {repeats} turns x {len(levers)} levers, {gen_tokens} new tokens each")
        samples = {lv: [] for lv in levers}
        for turn in range(repeats):
            for lever in levers[turn % len(levers):] + levers[:turn % len(levers)]:
                samples[lever].append(once(lever, prompt))

        stock_ids = samples["stock"][0][2]
        ref = _teacher_forced(mx, model, make_prompt_cache, prompt, stock_ids)
        for lever in levers:
            s = samples[lever]
            pre = [ctx / x[0] for x in s]
            dec = [(gen_tokens - 1) / x[1] for x in s]
            r = LeverResult(lever, float(np.median(pre)), float(np.median(dec)), tokens_total=gen_tokens,
                            decode_lo=min(dec), decode_hi=max(dec), prefill_lo=min(pre), prefill_hi=max(pre),
                            turns=repeats, decode_turns=dec, prefill_turns=pre, context=ctx,
                            weight_bytes=e2e.weight_bytes(w3_model) if lever == "w3" else wbytes,
                            kv_bytes=kvb[KV_BITS.get(lever)])
            ids = s[0][2]
            if lever != "stock":
                r.tokens_match = next((i for i, (a, b) in enumerate(zip(ids, stock_ids)) if a != b), len(ids))
            if lever == "spec":
                r.accept_rate = s[0][3] / max(gen_tokens - 1, 1)
            if lever in KV_BITS or lever == "w3":
                got = _teacher_forced(mx, w3_model if lever == "w3" else model, make_prompt_cache, prompt,
                                      stock_ids, KV_BITS.get(lever))
                r.top1, r.kl = drift(mx, ref, got)
            results.append(r)
            on_progress(f"{lever} @ {ctx:,}: decode {r.decode_tps:,.1f} tok/s ({r.decode_lo:,.1f}-{r.decode_hi:,.1f})")
    del model, draft_model, w3_model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return results


def cache_bytes(mx, model, make_prompt_cache, prompt, kv_bits: int | None = None) -> int:
    """Bytes of KV cache one decoded word reads after `prompt` (keys and values, every layer)."""
    cache = make_prompt_cache(model)
    _prefill(mx, model, cache, prompt)
    _quantize_cache(cache, kv_bits)
    from mlx.utils import tree_flatten
    return sum(a.nbytes for c in cache for _, a in tree_flatten(c.state) if hasattr(a, "nbytes"))


def speed_limit_tps(weight_bytes: int, kv: int, peak_gbps: float) -> float:
    """Every word reads every weight and the whole KV cache once."""
    return peak_gbps * 1e9 / (weight_bytes + kv)
