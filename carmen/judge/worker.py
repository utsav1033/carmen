"""The part of the judge that touches the GPU. Runs in its own process.

A bad kernel can hang the GPU or abort the process, so the parent launches this
with a timeout and treats a crash as a verdict, not an accident.
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np

from .. import ops
from ..backends import PAD, Kernel, static_check
from ..ops.base import Case
from . import check, stats

DETERMINISM_RUNS = 5
INNER = 20  # max launches per timed sample, so fixed per-call overhead doesn't dominate small kernels
TIMING_MEMORY = 256 * 1024 * 1024  # cap on buffers one timed sample may allocate (GPU memory is your RAM)
SHRINK_LADDER = [(1, 1), (1, 2), (1, 3), (1, 7), (1, 31), (1, 32), (1, 33), (1, 64), (1, 65),
                 (1, 255), (1, 256), (1, 257), (1, 1025), (3, 33), (2, 4097)]


class Prepared:
    """One test case, with its answer key and the stock library's result."""

    def __init__(self, op, adapter, case: Case):
        self.case = case
        self.inputs = op.materialize(case)
        self.ref = op.reference(self.inputs)
        self.dev = adapter.upload(self.inputs)
        base = adapter.baseline(op, self.dev)()
        base_err = check.row_relative_error(np.asarray(base, dtype=np.float64), self.ref)
        self.tol = check.tolerance(case.dtype, case.n, base_err)
        cap = check.CAP * check.floor_tol(case.dtype, case.n)
        # Input validity gate: if the stock library can't pass an input under the cap, the
        # input is unfair (or the reference semantics disagree), so drop the input, not the kernel.
        self.valid = check.compare(np.asarray(base).ravel(), self.ref, case.dtype, cap).ok


def _run_case(adapter, op, built, cfg, p: Prepared):
    out = adapter.run(built, op, cfg, p.dev, p.case.rows, p.case.n, p.case.dtype)
    cmp = check.compare(out, p.ref, p.case.dtype, p.tol, pad=PAD)
    if cmp.ok:
        rows, n = p.ref.shape
        inv = op.invariants(p.inputs, out[: rows * n].reshape(rows, n).astype(np.float64), p.ref)
        if inv:
            cmp.ok, cmp.patterns, cmp.where = False, inv, "whole output"
    return out, cmp


def _shrink(adapter, op, built, cfg, case: Case) -> str | None:
    """Find the smallest input of the same kind that still fails."""
    for rows, n in SHRINK_LADDER:
        if rows * n >= case.rows * case.n:
            break
        small = case.with_shape(rows, n)
        try:
            p = Prepared(op, adapter, small)
            if p.valid and not _run_case(adapter, op, built, cfg, p)[1].ok:
                return f"rows={rows}, n={n}"
        except Exception:
            return f"rows={rows}, n={n} (crashes)"
    return None


def _correctness(adapter, op, built, cfg, prepared: list[Prepared], shrink: bool):
    failures = []
    for p in prepared:
        if not p.valid:
            continue
        try:
            _, cmp = _run_case(adapter, op, built, cfg, p)
        except Exception as e:
            failures.append({"case": p.case.label(), "family": p.case.family, "patterns": ["launch error"],
                             "detail": _trim(str(e))})
            break
        if not cmp.ok:
            failures.append({"case": p.case.label(), "family": p.case.family, **cmp.to_json()})
    if not failures:
        big = [p for p in prepared if p.valid and p.case.n >= 257][:2]
        for p in big:
            outs = [adapter.run(built, op, cfg, p.dev, p.case.rows, p.case.n, p.case.dtype).tobytes()
                    for _ in range(DETERMINISM_RUNS)]
            if len(set(outs)) > 1:
                failures.append({"case": p.case.label(), "family": p.case.family,
                                 "patterns": ["output changes between identical runs (race condition)"],
                                 "where": "nondeterministic"})
                break
    # Lead with the largest failing input (most informative location), then shrink it
    # to the smallest one that still fails.
    failures.sort(key=lambda f: -_size(f["case"]))
    if failures and shrink and "launch error" not in failures[0]["patterns"]:
        first = next(p for p in prepared if p.case.label() == failures[0]["case"])
        failures[0]["smallest_failing"] = _shrink(adapter, op, built, cfg, first.case)
    return failures


def _size(label: str) -> int:
    rows, n = label.split("[", 1)[1].split(",", 1)[0].split("x")
    return int(rows) * int(n)


def _inner(op, rows: int, n: int, dt: str) -> int:
    """Launches per sample: as many as fit the memory cap (the stock op allocates a few
    intermediates per launch), at least 1, at most INNER."""
    per_launch = op.bytes_moved(rows, n, dt) * 2
    return max(1, min(INNER, TIMING_MEMORY // per_launch))


def _timing(adapter, op, built, cfg, shapes, dtypes, peak_gbps):
    rows_out = []
    for rows, n in shapes:
        for dt in dtypes:
            p = Prepared(op, adapter, Case("normal", rows, n, dt, rows * 31 + n, "timing"))
            k = _inner(op, rows, n, dt)
            tc, tb = stats.time_pair(adapter.launch(built, op, cfg, p.dev, rows, n, dt, inner=k),
                                     adapter.baseline_launch(op, p.dev, inner=k))
            tc, tb = tc / k, tb / k
            if hasattr(adapter, "release"):
                adapter.release()
            s, lo, hi = stats.speedup(tc, tb)
            gbps = op.bytes_moved(rows, n, dt) / float(np.median(tc)) / 1e9
            rows_out.append({"shape": [rows, n], "dtype": dt, "ms": float(np.median(tc)) * 1e3,
                             "baseline_ms": float(np.median(tb)) * 1e3, "speedup": s, "ci": [lo, hi],
                             "gbps": gbps, "pct_peak": gbps / peak_gbps if peak_gbps else None})
    return rows_out


def evaluate(req: dict, adapter) -> dict:
    op = ops.get(req["op"])
    kernel = Kernel.from_json(req["kernel"])
    res = {"op": op.name, "digest": kernel.digest(), "stage": "static", "correct": False}

    err = static_check(kernel)
    if err:
        res["error"] = err
        return res

    visible = [Prepared(op, adapter, Case.from_json(c)) for c in req["visible"]]
    res["invalid_cases"] = [p.case.label() for p in visible if not p.valid]

    res["stage"] = "compile"
    try:
        built = adapter.build(op, kernel)
        # MLX compiles lazily: the first launch is where compile errors surface.
        probe = next(p for p in visible if p.valid)
        adapter.run(built, op, kernel.configs[0], probe.dev, probe.case.rows, probe.case.n, probe.case.dtype)
    except Exception as e:
        res["error"] = _trim(str(e))
        return res

    res["stage"] = "correctness"
    configs = []
    for cfg in kernel.configs:
        fails = _correctness(adapter, op, built, cfg, visible, shrink=True)
        configs.append({"config": cfg, "passed": not fails, "n_failures": len(fails), "failures": fails[:5]})
    res["configs"] = configs

    naive = Prepared(op, adapter, op.naive)
    naive_out = adapter.run(built, op, kernel.configs[0], naive.dev, naive.case.rows, naive.case.n, naive.case.dtype)
    res["naive_pass"] = check.naive_check(naive_out, naive.ref, check.NAIVE_LOOSE)
    res["naive_pass_strict"] = check.naive_check(naive_out, naive.ref, check.NAIVE_STRICT)
    res["full_pass_default_config"] = configs[0]["passed"]

    passing = [c["config"] for c in configs if c["passed"]]
    if not passing:
        return res
    res["correct"] = True
    if req.get("skip_timing"):
        res["stage"] = "done"
        return res

    res["stage"] = "timing"
    peak = req.get("peak_gbps")
    shapes = [tuple(s) for s in req.get("timing_shapes") or op.timing_shapes]
    dtypes = req.get("timing_dtypes", ["float32", "float16"])
    best = None
    # Pick the config on one representative shape, then time only the winner everywhere.
    if len(passing) > 1:
        probe = [shapes[len(shapes) // 2]]
        pick = max(passing, key=lambda cfg: _timing(adapter, op, built, cfg, probe, ["float32"], peak)[0]["speedup"])
    else:
        pick = passing[0]
    t = _timing(adapter, op, built, pick, shapes, dtypes, peak)
    best = (pick, stats.geomean([r["speedup"] for r in t]), t)
    res["best_config"], res["speedup_geomean"], res["timing"] = best
    default = kernel.configs[0]
    res["default_config_speedup"] = best[1] if default == best[0] or default not in passing else \
        stats.geomean([r["speedup"] for r in _timing(adapter, op, built, default, shapes[:1], ["float32"], peak)])
    pcts = [r["pct_peak"] for r in best[2] if r["pct_peak"] is not None]
    res["pct_peak_min"] = min(pcts) if pcts else None
    # Faster than the memory system can move the bytes means the measurement is wrong
    # (or the kernel skipped work): never report it as a speedup without a flag.
    res["suspect_timing"] = any(p > 1.05 for p in pcts)

    if req.get("hidden"):
        res["stage"] = "hidden"
        hidden = [Prepared(op, adapter, Case.from_json(c)) for c in req["hidden"]]
        hf = _correctness(adapter, op, built, best[0], hidden, shrink=False)
        n_valid = sum(p.valid for p in hidden)
        res["hidden"] = {"total": n_valid, "failed": len({f["case"] for f in hf}), "failures": hf}
        ht = _timing(adapter, op, built, best[0], [tuple(s) for s in req["hidden_timing_shapes"]], ["float32"], peak)
        res["hidden"]["timing"] = ht
        res["hidden"]["speedup_geomean"] = stats.geomean([r["speedup"] for r in ht])
    res["stage"] = "done"
    return res


def _trim(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[:lines])


def main() -> None:
    req = json.loads(sys.stdin.read())
    from .. import backends
    try:
        if req.get("task") == "peak":
            adapter = backends.get(req["backend"])
            result = {"chip": adapter.chip(), "peak_gbps": adapter.measure_peak_gbps()}
        else:
            result = evaluate(req, backends.get(req["backend"]))
    except Exception:
        result = {"stage": "crash", "correct": False, "error": _trim(traceback.format_exc(), 40)}
    with open(req["out_path"], "w") as f:
        json.dump(result, f, default=float)


if __name__ == "__main__":
    main()
