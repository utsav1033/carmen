"""carmen: the command line.

    carmen ops                         list the ops the judge knows
    carmen peak                        measure this GPU's memory bandwidth (cached)
    carmen judge softmax my.metal      judge a kernel you wrote
    carmen broken softmax              measure the judge against seeded broken kernels
    carmen run masked_softmax          let Carmy cook
    carmen report runs/<id>            summarize a run
    carmen playbook                    show what Carmy has learned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import broken, dotenv, judge, ops, ui
from .backends import Kernel
from .carmy import DEFAULT_MODEL


def _configs(args) -> list[dict]:
    params = dict(p.split("=", 1) for p in args.param or [])
    return [{"TG": tg, **{k: int(v) for k, v in params.items()}} for tg in (args.tg or [256])]


def _verdict_line(v: dict) -> str:
    if v["stage"] in ("static", "compile", "timeout", "crash"):
        return f"{ui.BAD} rejected at {v['stage']}"
    if not v["correct"]:
        n = sum(c["n_failures"] for c in v["configs"])
        return f"{ui.BAD} wrong ({n} failing tests)"
    return f"{ui.OK} verified"


def cmd_ops(args) -> int:
    ui.banner("ops")
    ui.table(["op", "what", "visible tests"], [[ui.s(o.name, "bold"), o.summary, len(o.visible)] for o in ops.OPS.values()])
    return 0


def cmd_peak(args) -> int:
    ui.banner("peak bandwidth")
    p = judge.peak_gbps(args.backend, refresh=args.refresh)
    gbps = f"{p['peak_gbps']:.0f} GB/s"
    print(f"{ui.OK} {p['chip']}: {ui.s(gbps, 'bold')} (copy kernel, best of 20)")
    return 0


def cmd_judge(args) -> int:
    ui.banner(f"judge · {args.op}")
    kernel = Kernel(Path(args.kernel).read_text(), Path(args.header).read_text() if args.header else "", _configs(args))
    peak = judge.peak_gbps(args.backend)["peak_gbps"]
    v = judge.judge(args.op, kernel, backend=args.backend, peak=peak, round_seed=0,
                    hidden_seed=args.hidden_seed if args.hidden else None)
    if args.json:
        print(json.dumps(v, indent=2, default=float))
        return 0 if v["correct"] else 1
    print(_verdict_line(v))
    if v.get("naive_pass") is not None:
        mark = ui.OK if v["naive_pass"] else ui.BAD
        print(f"{mark} KernelBench-style check (one shape, allclose 1e-2): {'pass' if v['naive_pass'] else 'fail'}")
    ui.rule("feedback Carmy would see")
    print(judge.feedback(v))
    if v.get("hidden"):
        h = v["hidden"]
        ui.rule("hidden tests (never shown to Carmy)")
        print(f"{ui.OK if not h['failed'] else ui.BAD} {h['total'] - h['failed']}/{h['total']} hidden cases pass; "
              f"hidden-size speed {ui.speed(h.get('speedup_geomean'))}")
    return 0 if v["correct"] else 1


def cmd_broken(args) -> int:
    ui.banner(f"measuring the judge · {args.op}")
    rows, killed, fooled = [], 0, 0
    g = judge.judge(args.op, broken.golden(args.op), backend=args.backend, skip_timing=True)
    print(f"golden kernel: {_verdict_line(g)}")
    if not g["correct"]:
        print(f"{ui.WARN} the golden kernel fails, so the tests or the reference need fixing first:\n{judge.feedback(g)}")
        return 1
    ms = broken.mutants(args.op)
    for m, kernel in ms:
        v = judge.judge(args.op, kernel, backend=args.backend, skip_timing=True)
        dead = not v["correct"]
        killed += dead
        naive = v.get("naive_pass")
        fooled += bool(naive) and dead
        where = ""
        if v.get("configs") and v["configs"][0]["failures"]:
            where = v["configs"][0]["failures"][0].get("where", "")
        elif v.get("error"):
            where = v["error"].splitlines()[0][:50]
        rows.append([m.name, ui.s(m.family, "grey"), ui.OK + " killed" if dead else ui.BAD + " SURVIVED",
                     (ui.s("passes", "yellow") if naive else ui.s("fails", "grey")) if naive is not None else "—",
                     ui.s(where[:60], "grey")])
    ui.table(["mutant", "family", "carmen judge", "naive check", "where carmen located it"], rows)
    ui.rule()
    print(f"carmen killed {ui.s(f'{killed}/{len(ms)}', 'bold')} seeded bugs. "
          f"The KernelBench-style check would have accepted {ui.s(f'{fooled}/{len(ms)}', 'bold', 'yellow')} of them.")
    return 0 if killed == len(ms) else 1


def cmd_run(args) -> int:
    from . import loop
    ui.banner(f"cooking · {args.op} · {args.mode}")
    peak = judge.peak_gbps(args.backend)
    print(ui.s(f"{peak['chip']} · {peak['peak_gbps']:.0f} GB/s peak · {args.model} · effort {args.effort}", "grey"))

    def on_event(t, d):
        if t == "round_started":
            ui.rule(f"round {d['round']}")
        elif t == "attempt_submitted":
            print(f"  {ui.s(d['attempt'], 'bold')} {ui.s((d['plan'] or '')[:90], 'grey')}")
        elif t == "judge_result":
            speed = f"  {ui.speed(d['speedup'])}  {ui.pct(d['pct_peak_min'])} of peak" if d["correct"] else ""
            h = d.get("hidden")
            hid = f"  hidden {h['total'] - h['failed']}/{h['total']}" if h else ""
            mark = ui.OK if d["correct"] else ui.BAD
            print(f"     {mark} {d['stage']}{speed}{ui.s(hid, 'grey')}")
        elif t == "champion":
            print(f"  {ui.s('★ new champion', 'magenta', 'bold')} {d['attempt']}  {ui.speed(d['speedup'])}")
        elif t == "corpus_grew":
            print(ui.s(f"  + {len(d['added'])} failing input(s) saved to the regression corpus", "grey"))
        elif t == "playbook_updated" and d["proposed"]:
            kept = d["proposed"] - len(d["rejected"])
            print(ui.s(f"  playbook: {kept} lesson(s) kept, {len(d['rejected'])} rejected", "grey"))
        elif t in ("carmy_error", "reflect_error"):
            print(f"  {ui.WARN} {t}: {d['error'][:200]}")
        elif t == "stopped":
            print(ui.s(f"  stopped: {d['reason']}", "grey"))

    summary = loop.run(args.op, rounds=args.rounds, k=args.k, mode=args.mode, model=args.model, effort=args.effort,
                       backend=args.backend, runs_dir=Path(args.runs), memory_dir=Path(args.memory),
                       patience=args.patience, peak=peak, on_event=on_event)
    ui.rule("result")
    _print_summary(summary)
    return 0


def _print_summary(sm: dict) -> None:
    ui.table(["", ""], [
        ["attempts", f"{sm['attempts']} ({sm['compiled']} compiled, {sm['verified']} verified)"],
        ["first-round verified", sm["first_round_verified"]],
        ["naive check fooled", f"{sm['naive_pass_but_wrong']} of {sm['naive_pass']} naive passes were wrong"],
        ["hidden tests clean", sm["hidden_clean"] + " verified kernels"],
        ["champion", sm.get("champion") or "none"],
        ["speedup vs MLX", ui.speed(sm.get("speedup"))],
        ["on hidden sizes", ui.speed(sm.get("hidden_speedup"))],
        ["worst % of peak", ui.pct(sm.get("pct_peak_min"))],
    ])
    print(ui.s(f"\nrun saved to {sm['run_dir']}", "grey"))


def cmd_report(args) -> int:
    sm = json.loads((Path(args.run) / "summary.json").read_text())
    sm.setdefault("run_dir", args.run)
    ui.banner(f"report · {Path(args.run).name}")
    _print_summary(sm)
    return 0


def cmd_playbook(args) -> int:
    from .memory import Playbook
    ui.banner("playbook")
    pb = Playbook(Path(args.memory) / "playbook.json")
    if not pb.lessons:
        print(ui.s("empty: Carmy hasn't proven anything yet.", "grey"))
        return 0
    ui.table(["id", "kind", "status", "u", "used", "lesson"],
             [[l.id, l.kind, l.status, f"{l.u:+.2f}", l.n_ado, l.text[:90]] for l in pb.lessons.values()])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="carmen", description="Let an LLM cook GPU kernels. Trust nothing it can't prove.")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--env", default=".env", help="file of KEY=value settings to load (default: ./.env)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ops", help="list ops").set_defaults(fn=cmd_ops)
    p = sub.add_parser("peak", help="measure memory bandwidth")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_peak)

    p = sub.add_parser("judge", help="judge a kernel you wrote")
    p.add_argument("op", choices=list(ops.OPS))
    p.add_argument("kernel", help="path to the kernel body (.metal)")
    p.add_argument("--header", help="optional header file")
    p.add_argument("--tg", type=int, action="append", help="threads per threadgroup (repeat to sweep)")
    p.add_argument("--param", action="append", help="extra template constant NAME=VALUE")
    p.add_argument("--hidden", action="store_true", help="also run a fresh hidden draw")
    p.add_argument("--hidden-seed", type=int, default=1234)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("broken", help="measure the judge against seeded broken kernels")
    p.add_argument("op", choices=list(ops.OPS))
    p.set_defaults(fn=cmd_broken)

    p = sub.add_parser("run", help="let Carmy cook")
    p.add_argument("op", choices=list(ops.OPS))
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--k", type=int, default=3, help="parallel Carmy calls per round")
    p.add_argument("--mode", choices=["loop", "bon"], default="loop", help="bon = best-of-N control arm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--runs", default="runs")
    p.add_argument("--memory", default="memory")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="summarize a run")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("playbook", help="show learned lessons")
    p.add_argument("--memory", default="memory")
    p.set_defaults(fn=cmd_playbook)

    args = ap.parse_args(argv)
    dotenv.load(args.env)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
