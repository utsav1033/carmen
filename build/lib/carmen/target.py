"""`--for <model>`: judge a kernel where it will actually run.

A kernel is fast *somewhere*: at some shape, in some number format, against some baseline. The
SoL-Pi paper (NVIDIA, 2026) argues the score must be declared before the search starts and must
match the deployment; otherwise the search optimizes a number nobody runs. carmen learned this the
hard way: a residual_rmsnorm champion scored 1.15x in the judge (float32/float16, test shapes) and
0.69x inside Qwen (bfloat16, 512 x 896).

A Target fixes, before Carmy writes anything:
  - the model's number format (read from the model, not assumed)
  - one shape per regime: `decode` (1 row: writing a word) and `prefill` (the prompt length)
  - the metric: speedup vs mx.compile(stock op), the strongest baseline a user gets for free
and the judge keeps a champion per regime instead of one averaged number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ALIASES = {
    "qwen0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "llama1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "qwen3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}

# Ops whose rows are tokens and whose row length is the model's hidden size.
TARGETABLE = ("rmsnorm", "add_rmsnorm", "residual_rmsnorm", "layernorm", "mlp_up", "mlp_down")
DECODE_ONLY = ("mlp_up", "mlp_down")  # decode kernels: prefill stays stock

METRIC = "speedup vs mx.compile(stock MLX op), geomean over regimes"


@dataclass
class Target:
    model: str
    dtype: str
    regimes: dict[str, list[int]]  # regime -> [rows, n]
    metric: str = METRIC

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Target":
        return Target(d["model"], d["dtype"], {k: list(v) for k, v in d["regimes"].items()}, d.get("metric", METRIC))

    def describe(self) -> str:
        shapes = ", ".join(f"{name} {'x'.join(map(str, s))}" for name, s in self.regimes.items())
        return (f"This kernel is for {self.model}. It runs in {self.dtype} at exactly these shapes: {shapes} "
                "(decode = writing one word, 1 row; prefill = reading the prompt). Score: "
                f"{self.metric}. Shapes outside these do not count, but every input must still be correct.")


def make(model: str, hidden: int, dtype: str, prompt_tokens: int = 512, op: str = "", mlp: int = 0) -> Target:
    if op == "mlp_up":  # (rows, n, K): output is the MLP width, K the hidden size
        return Target(model, dtype, {"decode": [1, mlp, hidden]})
    if op == "mlp_down":
        return Target(model, dtype, {"decode": [1, hidden, mlp]})
    return Target(model, dtype, {"decode": [1, hidden], "prefill": [prompt_tokens, hidden]})


def resolve(op_name: str, model: str, prompt_tokens: int = 512) -> Target:
    """Read the model's hidden size and number format (needs MLX and mlx-lm)."""
    if op_name not in TARGETABLE:
        raise SystemExit(f"--for works with {', '.join(TARGETABLE)} so far: their shapes come straight from "
                         f"the model's sizes. {op_name} isn't wired to a model shape yet.")
    from . import profile
    mx, nn, load, _ = profile._imports()
    name = ALIASES.get(model, model)
    m, _ = load(name)
    spec = profile.read_spec(m, name)
    del m
    return make(name, spec.hidden, spec.dtype, prompt_tokens, op_name, spec.mlp)
