"""Carmy: the model that writes kernels. Smart, trusted to write, never trusted to grade.

Every call is stateless: Carmy gets a structured handoff (task, chip facts, playbook,
the best verified kernel so far and the judge's located feedback), never a chat history.
Carmy has no tools, so it cannot touch the judge; it can only return kernel text.
"""

from __future__ import annotations

import json
import os

from .backends import Kernel

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = """\
You are Carmy, a GPU kernel engineer. You write Apple Metal compute kernels that run through MLX's \
mx.fast.metal_kernel, and you care about two things in this order: exactly correct, then as fast as the hardware allows.

How your kernel runs:
- You write only the kernel BODY. MLX generates the signature: each input is a pointer named after it \
(element type T; `scale` is float), the output is `device T* out`, and each input `name` also gets \
`name_shape` (`const constant int*`, e.g. `x_shape`) with its shape. MLX puts small inputs in the `constant` address space and large ones in `device`, so never \
spell out an input pointer's address space: write `auto xr = x + row * n;`.
- T is float, half or bfloat (bfloat16: float32's range with 8 bits of precision). Unless the task says otherwise, the harness launches one threadgroup \
per row: grid = (rows * TG, 1, 1), threadgroup = (TG, 1, 1). TG is a template constant you choose per config \
(a multiple of 32, at most 1024). When the task specifies its own launch (e.g. output tiles), follow it exactly.
- You may declare extra integer template constants (for example N_READS) and give up to 6 configs; \
the harness checks every config for correctness, drops failing ones, and keeps the fastest.
- Thread attributes are available by name: threadgroup_position_in_grid (uint3), \
thread_position_in_threadgroup (uint3), thread_position_in_grid (uint3), threads_per_threadgroup (uint3), \
simdgroup_index_in_threadgroup (uint), thread_index_in_simdgroup (uint). SIMD width is 32. \
simd_sum / simd_max, threadgroup memory (32 KB) and threadgroup_barrier are available.
- `header` is optional text placed before the kernel (helper functions). Do not use #include; the only pragmas allowed are `#pragma unroll` and `#pragma clang loop`.
- The output buffer is pre-filled with NaN before your kernel runs.

How you are judged (you cannot see the judge):
- Every output is compared with a float64 reference on many inputs, including sizes and data you never see. \
Tolerance is a few ulps plus float32 accumulation error, relative to each row's scale.
- Kernels that special-case sizes or data patterns fail on the unseen inputs. Write general code.
- Speed is measured against the stock MLX op on the same GPU, and reported as a fraction of measured peak memory \
bandwidth (and GFLOP/s for compute-bound ops like matmul).

Return JSON only, matching the schema. `plan` is 1-3 sentences: the structure and why it should be fast. \
`lessons_used` lists the ids of playbook lessons you actually applied (empty if none).
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "source": {"type": "string"},
        "header": {"type": "string"},
        "configs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "TG": {"type": "integer"},
                    "params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
                            "required": ["name", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["TG", "params"],
                "additionalProperties": False,
            },
        },
        "lessons_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["plan", "source", "header", "configs", "lessons_used"],
    "additionalProperties": False,
}

REFLECT_SYSTEM = """\
You review one round of GPU kernel attempts and write down what was learned, for a future engineer who \
will start from scratch. Only write a lesson when the judge's results show it: a failure that was fixed, \
or a change that made a verified kernel measurably faster. Each lesson states a mechanism, the condition \
where it applies, and the action. You may name size regimes with powers of two ("rows longer than 4096"), \
never exact test sizes. Mark a lesson `chip` if it is about the \
hardware (memory access, SIMD, threadgroups) and would help any op, or `op` if it is about this operation's \
math. Cite the attempt ids that prove it, exactly as written (for example "0-1"). Keep each lesson under 600 characters. Return at most 3 lessons; return none if nothing was proven.
"""

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["chip", "op"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "kind", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lessons"],
    "additionalProperties": False,
}

VARIANTS = [
    "Take the most robust, straightforward high-performance approach.",
    "Optimize memory traffic aggressively: vectorized loads, fewer passes over the row, registers over threadgroup memory.",
    "Try a structurally different algorithm from the obvious one (for example an online, single-pass reduction).",
]


class CarmyError(RuntimeError):
    pass


class CarmyAuthError(CarmyError):
    """The API rejected the credentials. Retrying can't help, so the run stops."""


def via_proxy() -> bool:
    """True when ANTHROPIC_BASE_URL points at an Anthropic-compatible proxy. Proxies get plain
    Messages API requests, without Anthropic-only betas. The URL is never printed."""
    base = os.environ.get("ANTHROPIC_BASE_URL", "")
    return bool(base) and "api.anthropic.com" not in base


def _client():
    """Reads ANTHROPIC_API_KEY and, for a proxy, ANTHROPIC_BASE_URL from the environment (or .env)."""
    import anthropic
    return anthropic.Anthropic()


def _request(system: str, prompt: str, schema: dict, model: str, effort: str) -> dict:
    req = {
        "model": model,
        "max_tokens": 16000,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": effort},
    }
    if os.environ.get("CARMEN_STRUCTURED", "1") != "0":
        req["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    return req


def _call(system: str, prompt: str, schema: dict, model: str, effort: str) -> tuple[dict, dict]:
    import anthropic
    client = _client()
    req = _request(system, prompt, schema, model, effort)
    try:
        if via_proxy():
            # Server-side refusal fallbacks are an Anthropic API feature; a proxy
            # handles fallbacks with its own router config instead.
            resp = client.messages.create(**req)
        else:
            resp = client.beta.messages.create(**req, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise CarmyAuthError(f"the API rejected the key ({e.status_code}): {e.message}") from e
    except anthropic.BadRequestError as e:
        hint = (" If your proxy doesn't pass structured outputs through, set CARMEN_STRUCTURED=0."
                if via_proxy() else "")
        raise CarmyError(f"API rejected the request: {e.message}.{hint}") from e
    if resp.stop_reason == "refusal":
        raise CarmyError(f"model declined ({getattr(resp.stop_details, 'category', None)})")
    if resp.stop_reason == "max_tokens":
        raise CarmyError("response truncated at max_tokens")
    text = next(b.text for b in resp.content if b.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
             "model": resp.model}
    return _parse_json(text), usage


def _parse_json(text: str) -> dict:
    """Structured outputs guarantee bare JSON. Without them (CARMEN_STRUCTURED=0), the model
    may wrap it in a ```json fence; strip that, and fail loudly on anything else."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise CarmyError(f"model did not return valid JSON: {text[:300]!r}") from e


def parse(data: dict) -> tuple[Kernel, list[str]]:
    configs = []
    for c in data["configs"] or [{"TG": 256, "params": []}]:
        cfg = {"TG": int(c["TG"])}
        cfg.update({p["name"]: int(p["value"]) for p in c.get("params", [])})
        configs.append(cfg)
    return Kernel(data["source"], data.get("header", ""), configs, data.get("plan", "")), list(data["lessons_used"])


def prompt(op, *, chip: str, peak: float | None, playbook: str, champion: dict | None,
           last: dict | None, variant: int, k: int, history: str = "") -> str:
    parts = [f"TASK: write a Metal kernel for `{op.name}` ({op.summary}).\n\n{op.contract}"]
    parts.append(f"CHIP: {chip}. Measured peak memory bandwidth: {peak:.0f} GB/s." if peak else f"CHIP: {chip}.")
    if playbook:
        parts.append("PLAYBOOK (lessons verified by the judge in earlier rounds; apply the ones that fit):\n" + playbook)
    if champion:
        parts.append("CURRENT BEST VERIFIED KERNEL:\n```metal\n" + champion["kernel"]["source"] + "\n```\n"
                     f"header:\n```metal\n{champion['kernel']['header']}\n```\n"
                     "Judge feedback on it:\n" + champion["feedback"])
        parts.append("Your score is the geomean speedup over EVERY shape and dtype in the table above. Speeding up "
                     "one shape while slowing the others lowers the score, and the kernel is thrown away. Make ONE "
                     "structural change you expect to raise the geomean without making any shape slower, keep it "
                     "correct, and say in `plan` which bottleneck you are attacking and which shapes should gain.")
    elif last:
        parts.append("PREVIOUS ATTEMPT (rejected):\n```metal\n" + last["kernel"]["source"] + "\n```\n"
                     "Judge feedback:\n" + last["feedback"] + "\n\nFix the problem the judge located.")
    else:
        parts.append("Write a correct, fast kernel.")
    if history:
        parts.append("ALREADY TRIED THIS RUN (the judge's results; do not repeat an idea that lost, "
                     "build on what won):\n" + history)
    if k > 1:
        parts.append(f"You are attempt {variant + 1} of {k} running in parallel. {VARIANTS[variant % len(VARIANTS)]}")
    return "\n\n".join(parts)


def write(op, *, model: str = DEFAULT_MODEL, effort: str = "high", **prompt_kw):
    """One Carmy call. Returns (kernel, lessons_used, usage, prompt_text)."""
    text = prompt(op, **prompt_kw)
    data, usage = _call(SYSTEM, text, SCHEMA, model, effort)
    kernel, used = parse(data)
    return kernel, used, usage, text


def reflect(op, summary: str, *, model: str = DEFAULT_MODEL) -> list[dict]:
    data, _ = _call(REFLECT_SYSTEM, f"Operation: {op.name} ({op.summary}).\n\n{summary}",
                    REFLECT_SCHEMA, model, "medium")
    return data["lessons"]
