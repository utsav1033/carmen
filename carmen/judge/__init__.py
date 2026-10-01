"""The judge: trusted, deterministic, and simple. It never asks a model anything.

`judge()` assembles the test cases (fixed visible battery + fresh per-round fuzz +
regression corpus + secret hidden draw), runs the GPU part in a separate process,
and returns a verdict. `feedback()` turns a verdict into what Carmy is allowed to
see: located errors and a speed profile, never the hidden tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import ops
from ..backends import Kernel
from ..ops.base import Case, fuzz_cases, hidden_cases, hidden_timing_shapes
from . import worker

CACHE = Path(os.environ.get("CARMEN_CACHE", Path.home() / ".cache" / "carmen"))


def _spawn(req: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as d:
        req = {**req, "out_path": str(Path(d) / "result.json")}
        try:
            root = str(Path(__file__).resolve().parents[2])
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))}
            proc = subprocess.run([sys.executable, "-m", "carmen.judge.worker"], input=json.dumps(req),
                                  capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"stage": "timeout", "correct": False,
                    "error": f"kernel did not finish within {timeout:.0f}s (hang or infinite loop)"}
        out = Path(req["out_path"])
        if out.exists():
            return json.loads(out.read_text())
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        return {"stage": "crash", "correct": False,
                "error": f"judge process died (exit {proc.returncode}); the kernel likely faulted the GPU.\n{tail}"}


def peak_gbps(backend: str = "metal", refresh: bool = False, timeout: float = 300) -> dict:
    """Measured copy-kernel bandwidth, cached per backend."""
    path = CACHE / f"peak-{backend}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    result = _spawn({"task": "peak", "backend": backend}, timeout)
    if "peak_gbps" not in result:
        raise RuntimeError(f"could not measure peak bandwidth: {result.get('error')}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def build_request(op_name: str, kernel: Kernel, *, round_seed: int | None = None,
                  hidden_seed: int | None = None, corpus: list[Case] = (), backend: str = "metal",
                  peak: float | None = None, skip_timing: bool = False) -> dict:
    op = ops.get(op_name)
    visible = list(op.visible) + list(corpus)
    if round_seed is not None:
        visible += fuzz_cases(op, round_seed)
    req = {"op": op_name, "backend": backend, "kernel": kernel.to_json(),
           "visible": [c.to_json() for c in visible], "peak_gbps": peak, "skip_timing": skip_timing}
    if hidden_seed is not None:
        req["hidden"] = [c.to_json() for c in hidden_cases(op, hidden_seed)]
        req["hidden_timing_shapes"] = hidden_timing_shapes(hidden_seed)
    return req


def judge(op_name: str, kernel: Kernel, *, adapter=None, timeout: float = 900, **kw) -> dict:
    """Judge one kernel. Pass `adapter` to run in-process (tests, or trusted debugging)."""
    req = build_request(op_name, kernel, **kw)
    if adapter is not None:
        return worker.evaluate(req, adapter)
    return _spawn(req, timeout)


def feedback(v: dict) -> str:
    """What Carmy sees: located errors and the speed profile. Hidden results are never included."""
    if v["stage"] in ("static", "compile", "timeout", "crash"):
        return f"REJECTED at {v['stage']}:\n{v.get('error', '')}"
    lines = []
    for c in v.get("configs", []):
        cfg = " ".join(f"{k}={val}" for k, val in c["config"].items())
        if c["passed"]:
            lines.append(f"[{cfg}] correct on every visible test.")
            continue
        f = c["failures"][0]
        msg = f"[{cfg}] FAILED {c['n_failures']} test(s). First: {f['case']} -> {'; '.join(f['patterns'])}"
        if f.get("where"):
            msg += f", in {f['where']}"
        if isinstance(f.get("max_err"), float) and f["max_err"] > 0:
            msg += f". Max row-relative error {_fmt(f['max_err'])}, allowed {_fmt(f['tol'])}"
        if f.get("first_bad"):
            r, col, got, want = f["first_bad"]
            msg += f". At row {r}, column {col}: got {_fmt(got)}, expected {_fmt(want)}"
        if f.get("smallest_failing"):
            msg += f". Smallest failing input of this kind: {f['smallest_failing']}"
        if f.get("detail"):
            msg += f"\n{f['detail']}"
        lines.append(msg + ".")
    if v.get("correct"):
        best = " ".join(f"{k}={val}" for k, val in v["best_config"].items())
        lines.append(f"\nSpeed (best config {best}) vs the stock MLX op, geomean {v['speedup_geomean']:.2f}x:")
        for r in v["timing"]:
            pct = f"{r['pct_peak']:.0%} of peak bandwidth" if r.get("pct_peak") else f"{r['gbps']:.0f} GB/s"
            comp = f", {r['speedup_compiled']:.2f}x vs mx.compile" if "speedup_compiled" in r else ""
            lines.append(f"  {r['shape'][0]}x{r['shape'][1]} {r['dtype']}: {r['ms']:.3f} ms vs {r['baseline_ms']:.3f} ms "
                         f"-> {r['speedup']:.2f}x [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]{comp}, {pct}")
    return "\n".join(lines)


def _fmt(x) -> str:
    return x if isinstance(x, str) else f"{x:.6g}"
