"""`carmen bench`: does the feedback loop beat blind sampling?

For each op, run the loop and the best-of-N control arm the same number of times with the
same budget (rounds x k Carmy calls, no early stop), each from a fresh memory, and compare:

  - final champion speed (vs MLX, and vs mx.compile(MLX))
  - round-1 best vs final best: did later rounds add anything?
  - how many kernels the judge verified

Writes bench.json and a markdown table (bench.md) ready to paste into a README.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable

from .events import read_events

MODES = ("loop", "bon")


def _curve(run_dir: Path) -> list[float]:
    """Best verified speedup found in each round (0 when a round verified nothing)."""
    best: dict[int, float] = {}
    for e in read_events(run_dir):
        if e["type"] == "judge_result":
            best.setdefault(e["round"], 0.0)
            if e.get("correct") and e.get("speedup"):
                best[e["round"]] = max(best[e["round"]], e["speedup"])
    return [best[r] for r in sorted(best)]


def run(ops: list[str], *, repeats: int, rounds: int, k: int, root: Path, run_fn: Callable,
        on_progress: Callable[[str], None] = print, **run_kw) -> dict:
    root = Path(root)
    results = []
    for op in ops:
        for i in range(repeats):
            for mode in MODES:
                tag = f"{op}-{mode}-{i}"
                on_progress(f"{tag}: cooking ({rounds} rounds x {k} drafts)")
                sm = run_fn(op, rounds=rounds, k=k, mode=mode, patience=rounds + 1,
                            runs_dir=root / tag, memory_dir=root / tag / "memory", **run_kw)
                curve = _curve(Path(sm["run_dir"]))
                results.append({"op": op, "mode": mode, "repeat": i, "speedup": sm.get("speedup"),
                                "speedup_compiled": sm.get("speedup_compiled"),
                                "hidden_speedup": sm.get("hidden_speedup"), "verified": sm["verified"],
                                "attempts": sm["attempts"], "curve": curve, "run_dir": sm["run_dir"]})
                on_progress(f"{tag}: best {_x(sm.get('speedup'))}, per round {' '.join(_x(c) for c in curve)}")
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "rounds": rounds, "k": k, "repeats": repeats,
           "results": results, "table": table(results)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "bench.json").write_text(json.dumps(out, indent=2, default=float))
    (root / "bench.md").write_text(markdown(out["table"], rounds, k, repeats))
    return out


def _mean(xs):
    xs = [x for x in xs if x]
    return statistics.mean(xs) if xs else None


def table(results: list[dict]) -> list[dict]:
    rows = []
    for op in dict.fromkeys(r["op"] for r in results):
        for mode in MODES:
            rs = [r for r in results if r["op"] == op and r["mode"] == mode]
            if not rs:
                continue
            first = [r["curve"][0] for r in rs if r["curve"]]
            rows.append({
                "op": op, "mode": mode, "runs": len(rs),
                "speedup": _mean(r["speedup"] for r in rs),
                "speedup_min": min((r["speedup"] or 0 for r in rs), default=0),
                "speedup_max": max((r["speedup"] or 0 for r in rs), default=0),
                "speedup_compiled": _mean(r["speedup_compiled"] for r in rs),
                "hidden_speedup": _mean(r["hidden_speedup"] for r in rs),
                "round1": _mean(first),
                "verified": sum(r["verified"] for r in rs), "attempts": sum(r["attempts"] for r in rs),
            })
    return rows


def _x(v) -> str:
    return f"{v:.2f}x" if v else "-"


def markdown(rows: list[dict], rounds: int, k: int, repeats: int) -> str:
    lines = [f"Each row: {repeats} run(s) of {rounds} rounds x {k} drafts, fresh memory, no early stop.",
             "`loop` feeds the judge's results back; `best-of-N` makes the same number of calls blind.", "",
             "| kernel | mode | speedup vs MLX (mean, range) | vs mx.compile | unseen sizes | round 1 → final | verified |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        mode = "loop" if r["mode"] == "loop" else "best-of-N"
        lines.append(f"| {r['op']} | {mode} | {_x(r['speedup'])} ({_x(r['speedup_min'])}–{_x(r['speedup_max'])}) "
                     f"| {_x(r['speedup_compiled'])} | {_x(r['hidden_speedup'])} | {_x(r['round1'])} → {_x(r['speedup'])} "
                     f"| {r['verified']}/{r['attempts']} |")
    return "\n".join(lines) + "\n"
