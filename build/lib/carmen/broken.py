"""Seeded broken kernels: how we measure the judge.

Each mutant is the golden kernel with one realistic bug, grouped into the fault
families from "Measuring the Checker" (Du et al., arXiv 2609.22220): boundary,
precision, sync, indexing, semantic. A strong judge kills all of them and never
rejects the golden kernel. Every mutant here is designed to be *actually* wrong
on some legal input; `carmen broken` reports which ones get through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends import Kernel

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


@dataclass(frozen=True)
class Mutant:
    name: str
    family: str
    old: str
    new: str
    why: str
    every: bool = False  # replace every occurrence, not just the first


MUTANTS = [
    Mutant("no_max_subtraction", "precision", "metal::exp(float(xr[i]) - m)", "metal::exp(float(xr[i]))",
           "exp overflows for large inputs"),
    Mutant("fp16_accumulate", "precision", "float s = 0.0f;", "half s = 0.0h;",
           "sum loses precision on long rows"),
    Mutant("store_through_half", "precision", "o[i] = T(", "o[i] = T((half)",
           "fp32 outputs rounded to fp16 (escapes allclose 1e-2)"),
    Mutant("tail_write_dropped", "boundary", "for (int i = tid; i < n; i += TG) { o[i]",
           "for (int i = tid; i < n - 1; i += TG) { o[i]", "last element never written"),
    Mutant("tail_sum_dropped", "boundary", "for (int i = tid; i < n; i += TG) { s +=",
           "for (int i = tid; i < n - 1; i += TG) { s +=", "last element missing from the sum"),
    Mutant("max_start_at_one", "boundary", "for (int i = tid; i < n; i += TG) { m =",
           "for (int i = tid + 1; i < n; i += TG) { m =", "first element of each thread skipped in max"),
    Mutant("barrier_removed", "sync",
           "if (lane == 0) { shared[0] = v; }\n}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nm = shared[0];",
           "if (lane == 0) { shared[0] = v; }\n}\nm = shared[0];", "race: threads read the max before it is written"),
    Mutant("partial_simd_reduce", "indexing", "(lane < n_sg) ? shared[lane] : 0.0f",
           "(lane < n_sg / 2) ? shared[lane] : 0.0f", "half the simdgroups dropped from the sum"),
    Mutant("wrong_row_stride", "indexing", "device T* o = out + row * n;", "device T* o = out + row * (n - 1);",
           "rows overlap in the output"),
    Mutant("max_sentinel_zero", "semantic", "float m = -INFINITY;", "float m = 0.0f;",
           "all-negative rows underflow to 0/0"),
    Mutant("sum_uses_local_max", "semantic", "m = shared[0];\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n\n// 2.",
           "threadgroup_barrier(mem_flags::mem_threadgroup);\n\n// 2.", "each simdgroup normalizes with its own max"),
    Mutant("divide_by_n", "semantic", "/ s)", "/ float(n))", "normalizes by n instead of the sum"),
]


# The norms share one golden-kernel skeleton (sum, reduce, rsqrt, write), so one list covers both.
NORM_MUTANTS = [
    Mutant("fp16_accumulate", "precision", "float s = 0.0f;", "half s = 0.0h;",
           "sum kept in half: overflows or drifts on big or offset rows"),
    Mutant("store_through_half", "precision", "o[i] = T(", "o[i] = T((half)",
           "fp32 outputs rounded to fp16 (escapes allclose 1e-2)"),
    Mutant("tail_write_dropped", "boundary", "for (int i = tid; i < n; i += TG) { o[i]",
           "for (int i = tid; i < n - 1; i += TG) { o[i]", "last element never written"),
    Mutant("tail_sum_dropped", "boundary", "for (int i = tid; i < n; i += TG) { s +=",
           "for (int i = tid; i < n - 1; i += TG) { s +=", "last element missing from the sum"),
    Mutant("barrier_removed", "sync",
           "if (lane == 0) { shared[0] = v; }\n}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n",
           "if (lane == 0) { shared[0] = v; }\n}\n", "race: threads read the sum before it is written"),
    Mutant("partial_simd_reduce", "indexing", "(lane < n_sg) ? shared[lane] : 0.0f",
           "(lane < n_sg / 2) ? shared[lane] : 0.0f", "half the simdgroups dropped from the sum"),
    Mutant("wrong_row_stride", "indexing", "device T* o = out + row * n;", "device T* o = out + row * (n - 1);",
           "rows overlap in the output"),
    Mutant("eps_dropped", "semantic", "+ eps)", ")", "no eps: zero rows divide by zero, tiny rows blow up"),
    Mutant("divide_by_n_minus_1", "semantic", "shared[0] / float(n)", "shared[0] / float(n - 1)",
           "averages over n - 1 instead of n"),
    Mutant("weight_ignored", "semantic", " * float(w[i])", "", "learned weight never applied"),
]

# matmul has its own skeleton (tiles, K loop), so its own bugs.
MATMUL_MUTANTS = [
    Mutant("fp16_accumulate", "precision", "float acc = 0.0f;", "half acc = 0.0h;",
           "dot products summed in half: drifts on long K"),
    Mutant("store_through_half", "precision", "out[r * N + c] = T(acc);", "out[r * N + c] = T((half)acc);",
           "fp32 outputs rounded to fp16"),
    Mutant("k_tail_dropped", "boundary", "k < K; ++k", "k < K - 1; ++k", "last term of every dot product missing"),
    Mutant("last_column_unwritten", "boundary", "if (r < M && c < N)", "if (r < M && c < N - 1)",
           "the right edge of the output is never written"),
    Mutant("k_starts_at_one", "boundary", "int k = 0; k < K", "int k = 1; k < K", "first term of every dot product missing"),
    Mutant("wrong_leading_dim", "indexing", "a[r * K + k]", "a[r * N + k]",
           "A indexed with N instead of K: right only when the matrices are square"),
    Mutant("b_transposed", "indexing", "b[k * N + c]", "b[c * K + k]", "B read as if transposed"),
    Mutant("tile_axes_swapped", "indexing", "int r = ty * BM + e / BN;\n    int c = tx * BN + e % BN;",
           "int r = tx * BM + e / BN;\n    int c = ty * BN + e % BN;",
           "grid x and y swapped: right only when the tile grid is square"),
]

# attention: the causal boundary, the cache offset, the softmax and the scale.
ATTENTION_MUTANTS = [
    Mutant("no_max_subtraction", "precision", "metal::exp(SCORE(i, j) - m)", "metal::exp(SCORE(i, j))",
           "exp overflows on large scores", every=True),
    Mutant("fp16_accumulate", "precision", "float acc = 0.0f;", "half acc = 0.0h;",
           "the weighted sum of v kept in half: drifts on long sequences"),
    Mutant("causal_excludes_self", "boundary", "i + (Lk - Lq));", "i + (Lk - Lq) - 1);",
           "each query can't see its own key"),
    Mutant("causal_peeks_ahead", "boundary", "i + (Lk - Lq));", "i + (Lk - Lq) + 1);",
           "each query sees one future key"),
    Mutant("head_dim_tail_dropped", "boundary", "d < D; d += TG", "d < D - 1; d += TG",
           "the last output column is never written"),
    Mutant("cache_offset_ignored", "indexing", "metal::min(Lk - 1, i + (Lk - Lq))", "metal::min(Lk - 1, i)",
           "causal mask aligned top-left: right only when Lq == Lk"),
    Mutant("v_read_transposed", "indexing", "v[j * D + d]", "v[d * Lk + j]", "v read as if transposed"),
    Mutant("barrier_removed", "sync",
           "if (lane == 0) { shared[0] = x; }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    m = shared[0];",
           "if (lane == 0) { shared[0] = x; }\n    }\n    m = shared[0];", "race: threads read the max before it is written"),
    Mutant("scale_missing", "semantic", "* scl)", ")", "scores not divided by sqrt(D)"),
    Mutant("not_normalized", "semantic", "T(acc / l)", "T(acc)", "weighted sum never divided by the softmax total"),
]

_SHARED_NORM = [m for m in NORM_MUTANTS if m.name not in ("eps_dropped", "wrong_row_stride")]

MUTANTS_BY_OP = {
    "residual_rmsnorm": _SHARED_NORM + [
        Mutant("eps_dropped", "semantic", "+ eps[0])", ")", "no eps: zero rows divide by zero"),
        Mutant("eps_hardcoded", "semantic", "+ eps[0])", "+ 1e-5f)", "eps fixed at Llama's 1e-5: wrong for Qwen's 1e-6"),
        Mutant("residual_tail_unwritten", "boundary", "for (int i = tid; i < n; i += TG) { hsum[i]",
               "for (int i = tid; i < n - 1; i += TG) { hsum[i]", "last element of h never written"),
        Mutant("y_overwrites_h", "indexing", "device T* o = out + (rows + row) * n;", "device T* o = out + row * n;",
               "y written over h: the residual is lost"),
        Mutant("residual_not_added", "semantic", "hsum[i] = T(float(xr[i]) + float(rr[i]));", "hsum[i] = T(float(xr[i]));",
               "h is x alone: the block's output never joins the residual"),
    ],
    "attention": ATTENTION_MUTANTS,
    "mlp_up": [
        Mutant("fp16_accumulate", "precision", "float g = 0.0f, u = 0.0f;", "half g = 0.0h, u = 0.0h;",
               "dot products summed in half: long rows lose precision"),
        Mutant("silu_overflows", "precision", "T(g / (1.0f + metal::exp(-g)) * u)",
               "T(metal::exp(g) / (1.0f + metal::exp(g)) * g * u)", "silu as e^g/(1+e^g): inf/inf = NaN for big g"),
        Mutant("bias_dropped", "semantic", "+ bgv) * hn;", ") * hn;", "gate weights lose their per-group bias"),
        Mutant("nibble_order_reversed", "indexing", "float((qg >> (4 * t)) & 0xFu)", "float((qg >> (4 * (7 - t))) & 0xFu)",
               "reads the 8 values of a word highest-first"),
        Mutant("wrong_group", "indexing", "const int grp = j * groups + k0 / 64;", "const int grp = j * groups + k0 / 128;",
               "scale and bias from the wrong group once K > 64"),
        Mutant("last_column_dropped", "boundary", "if (j >= n) { break; }", "if (j >= n - 1) { break; }",
               "the last output column is never written"),
        Mutant("last_word_dropped", "boundary", "for (int wd = lane; wd < words; wd += 32)",
               "for (int wd = lane; wd < words - 1; wd += 32)", "the last 8 values of K are left out of the dot product"),
        Mutant("eps_dropped", "semantic", "+ eps[0])", ")", "no eps: a zero row divides by zero"),
        Mutant("residual_not_added", "semantic", "const float hn = (float(xr[k]) + float(rr[k]))",
               "const float hn = (float(xr[k]))", "the matmuls see x, not x + res"),
        Mutant("barrier_removed", "sync", "threadgroup_barrier(mem_flags::mem_threadgroup);\nconst float inv",
               "const float inv", "race: threads read the sum before it is written"),
        Mutant("partial_simd_reduce", "indexing", "(lane < n_sg) ? shared[lane]", "(lane < n_sg / 2) ? shared[lane]",
               "half the simdgroups dropped from the norm's sum"),
        Mutant("wrong_row_stride", "indexing", "out[row * n + j]", "out[row * (n - 1) + j]", "rows overlap in the output"),
        Mutant("gate_up_swapped", "semantic", "T(g / (1.0f + metal::exp(-g)) * u)", "T(u / (1.0f + metal::exp(-u)) * g)",
               "silu applied to up instead of gate"),
    ],
    "mlp_down": [
        Mutant("fp16_accumulate", "precision", "float acc = 0.0f;", "half acc = 0.0h;",
               "dot product summed in half: long rows lose precision"),
        Mutant("bias_dropped", "semantic", "+ bv) *", ") *", "weights lose their per-group bias"),
        Mutant("nibble_order_reversed", "indexing", "(q >> (4 * t))", "(q >> (4 * (7 - t)))",
               "reads the 8 values of a word highest-first"),
        Mutant("wrong_group", "indexing", "const int grp = j * groups + k0 / 64;", "const int grp = j * groups + k0 / 128;",
               "scale and bias from the wrong group once K > 64"),
        Mutant("last_column_dropped", "boundary", "if (j >= n) { break; }", "if (j >= n - 1) { break; }",
               "the last output column is never written"),
        Mutant("last_word_dropped", "boundary", "for (int wd = lane; wd < words; wd += 32)",
               "for (int wd = lane; wd < words - 1; wd += 32)", "the last 8 values of K are left out of the dot product"),
        Mutant("residual_dropped", "semantic", "float(x[row * n + j]) + float(res[row * n + j]) + acc",
               "float(x[row * n + j]) + acc", "the attention output never joins the residual stream"),
        Mutant("wrong_act_row", "indexing", "auto ar = act + row * K;", "auto ar = act + row * (K - 8);",
               "rows after the first read the wrong activations"),
    ],
    "matmul": MATMUL_MUTANTS,
    "layernorm": NORM_MUTANTS,
    "rmsnorm": NORM_MUTANTS,
    "add_rmsnorm": NORM_MUTANTS + [
        Mutant("residual_dropped_on_write", "semantic", "o[i] = T((float(xr[i]) + float(rr[i])) * inv",
               "o[i] = T(float(xr[i]) * inv", "normalizes x + res but writes x, not the sum"),
    ],
}


def has_golden(op: str) -> bool:
    return (GOLDEN_DIR / f"{op}.metal").exists()


GOLDEN_CONFIGS = {"matmul": [{"TG": 256, "BM": 32, "BN": 32}], "attention": [{"TG": 64, "BQ": 1}],
                  "mlp_up": [{"TG": 256, "BN": 8}], "mlp_down": [{"TG": 256, "BN": 8}]}


def golden(op: str) -> Kernel:
    path = GOLDEN_DIR / f"{op}.metal"
    header = GOLDEN_DIR / f"{op}.header.metal"
    configs = GOLDEN_CONFIGS.get(op, [{"TG": 256}])
    return Kernel(path.read_text(), header.read_text() if header.exists() else "",
                  configs=[dict(c) for c in configs], plan="golden reference kernel")


def mutants(op: str) -> list[tuple[Mutant, Kernel]]:
    base = golden(op).source
    out = []
    for m in MUTANTS_BY_OP.get(op, MUTANTS):
        old = m.old.replace("float(xr[i])", "(float(xr[i]) * scl + float(mr[i]))") if op == "masked_softmax" else m.old
        new = m.new.replace("float(xr[i])", "(float(xr[i]) * scl + float(mr[i]))") if op == "masked_softmax" else m.new
        if old not in base:
            raise ValueError(f"mutant {m.name} does not apply to the {op} golden kernel")
        g = golden(op)
        out.append((m, Kernel(base.replace(old, new, -1 if m.every else 1), g.header, configs=g.configs, plan=m.why)))
    return out
