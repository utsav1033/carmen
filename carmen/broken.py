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

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "kernels" / "golden"


@dataclass(frozen=True)
class Mutant:
    name: str
    family: str
    old: str
    new: str
    why: str


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


def golden(op: str) -> Kernel:
    path = GOLDEN_DIR / f"{op}.metal"
    return Kernel(path.read_text(), configs=[{"TG": 256}], plan="golden reference kernel")


def mutants(op: str) -> list[tuple[Mutant, Kernel]]:
    base = golden(op).source
    out = []
    for m in MUTANTS:
        old = m.old.replace("float(xr[i])", "(float(xr[i]) * scl + float(mr[i]))") if op == "masked_softmax" else m.old
        new = m.new.replace("float(xr[i])", "(float(xr[i]) * scl + float(mr[i]))") if op == "masked_softmax" else m.new
        if old not in base:
            raise ValueError(f"mutant {m.name} does not apply to the {op} golden kernel")
        out.append((m, Kernel(base.replace(old, new, 1), configs=[{"TG": 256}], plan=m.why)))
    return out
