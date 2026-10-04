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
import time
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


def cmd_ui(args) -> int:
    from .tui import main as tui_main
    tui_main(args.backend, args.runs, args.memory, args.model)
    return 0


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
    rows, killed, fooled, fooled_strict = [], 0, 0, 0
    g = judge.judge(args.op, broken.golden(args.op), backend=args.backend, skip_timing=True)
    print(f"golden kernel: {_verdict_line(g)}")
    if g.get("invalid_cases"):
        # Inputs where stock MLX itself disagrees with the answer key are dropped, not graded.
        print(ui.s(f"  {len(g['invalid_cases'])} input(s) dropped because stock MLX disagrees with the answer key: "
                   f"{', '.join(g['invalid_cases'][:4])}", "yellow"))
    if not g["correct"]:
        print(f"{ui.WARN} the golden kernel fails, so the tests or the reference need fixing first:\n{judge.feedback(g)}")
        return 1
    ms = broken.mutants(args.op)
    for m, kernel in ms:
        v = judge.judge(args.op, kernel, backend=args.backend, skip_timing=True)
        dead = not v["correct"]
        killed += dead
        naive, strict = v.get("naive_pass"), v.get("naive_pass_strict")
        fooled += bool(naive) and dead
        fooled_strict += bool(strict) and dead
        where = ""
        if v.get("configs") and v["configs"][0]["failures"]:
            where = v["configs"][0]["failures"][0].get("where", "")
        elif v.get("error"):
            where = v["error"].splitlines()[0][:50]
        rows.append([m.name, ui.s(m.family, "grey"), ui.OK + " killed" if dead else ui.BAD + " SURVIVED",
                     _naive(naive), _naive(strict),
                     ui.s(where[:70], "grey")])
    ui.table(["mutant", "family", "carmen judge", "naive 1e-2", "naive 1e-4", "where carmen located it"], rows)
    ui.rule()
    print(f"carmen killed {ui.s(f'{killed}/{len(ms)}', 'bold')} seeded bugs. A KernelBench-style check "
          f"(one shape, allclose) would have accepted {ui.s(f'{fooled}/{len(ms)}', 'bold', 'yellow')} at "
          f"tolerance 1e-2 and {ui.s(f'{fooled_strict}/{len(ms)}', 'bold', 'yellow')} at 1e-4.")
    return 0 if killed == len(ms) else 1


def _naive(passed) -> str:
    if passed is None:
        return "—"
    return ui.s("passes", "yellow") if passed else ui.s("fails", "grey")


def cmd_run(args) -> int:
    from . import loop
    ui.banner(f"cooking · {args.op} · {args.mode}")
    peak = judge.peak_gbps(args.backend)
    from .carmy import CarmyAuthError
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
            print(f"  {ui.s('★ new champion', 'green', 'bold')} {d['attempt']}  {ui.speed(d['speedup'])}")
        elif t == "corpus_grew":
            print(ui.s(f"  + {len(d['added'])} failing input(s) saved to the regression corpus", "grey"))
        elif t == "playbook_updated" and d["proposed"]:
            kept = d["proposed"] - len(d["rejected"])
            print(ui.s(f"  playbook: {kept} lesson(s) kept, {len(d['rejected'])} rejected", "grey"))
            for x in d["rejected"]:
                why = x["reason"] if isinstance(x, dict) else x
                print(ui.s(f"    rejected: {why}", "grey"))
        elif t in ("carmy_error", "reflect_error"):
            print(f"  {ui.WARN} {t}: {d['error'][:200]}")
        elif t == "stopped":
            print(ui.s(f"  stopped: {d['reason']}", "grey"))

    tgt = None
    if args.for_model:
        from . import target
        print(ui.s(f"reading {args.for_model} for its shapes and number format", "grey"))
        tgt = target.resolve(args.op, args.for_model, args.prompt)
        print(f"target: {ui.s(tgt.model, 'bold')} · {tgt.dtype} · "
              + ", ".join(f"{k} {'x'.join(map(str, v))}" for k, v in tgt.regimes.items()))
        print(ui.s(f"score, declared before the run: {tgt.metric}", "grey"))
    try:
        summary = loop.run(args.op, rounds=args.rounds, k=args.k, mode=args.mode, model=args.model,
                           effort=args.effort, backend=args.backend, runs_dir=Path(args.runs),
                           memory_dir=Path(args.memory), patience=args.patience, peak=peak, on_event=on_event,
                           target=tgt.to_json() if tgt else None)
    except CarmyAuthError as e:
        print(f"\n{ui.BAD} {e}")
        print(ui.s("Check ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if you use a proxy) in .env, and that no old key is exported in your shell.", "grey"))
        return 2
    ui.rule("result")
    _print_summary(summary)
    if tgt and summary.get("regime_champions"):
        _holdout(args, summary)
    return 0


def _holdout(args, summary: dict) -> None:
    """The frozen final check, inside the real model. Carmy never saw it."""
    from . import e2e
    ui.rule("held-out check: inside the model")
    if args.op != e2e.OP:
        print(ui.s(f"only {e2e.OP} is wired into a model so far; {args.op}'s champions are judged, not model-checked.",
                   "grey"))
        return
    run_dir = Path(summary["run_dir"])
    champs = {}
    for regime, c in summary["regime_champions"].items():
        k = Kernel.from_json(json.loads((run_dir / "attempts" / c["attempt"] / "kernel.json").read_text()))
        champs[regime] = (k, c["config"], c["shape"][0], c["speedup_compiled"])
    results = e2e.holdout(summary["target"]["model"], champs, on_progress=lambda m: print(ui.s("  " + m, "grey")))
    rows = [[h.regime, summary["regime_champions"][h.regime]["attempt"], ui.speed(h.judge_vs_compiled),
             ui.speed(h.vs_compiled), ui.speed(h.vs_stock), f"{h.carmen_us:.1f} / {h.compiled_us:.1f} us",
             ui.s(f"{ui.OK} holds", "green") if h.agrees else ui.s("judge was wrong", "yellow")] for h in results]
    ui.table(["regime", "champion", "judge vs compile", "in model vs compile", "in model vs stock",
              "carmen / compiled", ""], rows, align="llrrrrl")
    summary["holdout"] = [{**h.__dict__, "vs_compiled": h.vs_compiled, "vs_stock": h.vs_stock, "agrees": h.agrees}
                          for h in results]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(ui.s("in model = called the way `carmen e2e` calls it (compiled, 48 calls chained, the model's norm "
               "weights). Agrees = same side of 1.0 and within 25% of the judge.", "grey"))


def _print_summary(sm: dict) -> None:
    ui.table(["", ""], [
        ["attempts", f"{sm['attempts']} ({sm['compiled']} compiled, {sm['verified']} verified)"],
        ["first-round verified", sm["first_round_verified"]],
        ["naive check fooled", f"{sm['naive_pass_but_wrong']} of {sm['naive_pass']} naive passes were wrong"],
        ["hidden tests clean", sm["hidden_clean"] + " verified kernels"],
        ["champion", sm.get("champion") or "none"],
        ["speedup vs MLX", ui.speed(sm.get("speedup"))],
        ["vs mx.compile(MLX)", ui.speed(sm.get("speedup_compiled"))],
        ["on hidden sizes", ui.speed(sm.get("hidden_speedup"))],
        ["worst % of peak", ui.pct(sm.get("pct_peak_min"))],
    ])
    if sm.get("regime_champions"):
        t = sm["target"]
        print(f"\ntarget {ui.s(t['model'], 'bold')} · {t['dtype']} · champion per regime (score vs mx.compile):")
        ui.table(["regime", "shape", "champion", "config", "vs stock", "vs mx.compile"], [
            [name, "x".join(map(str, c["shape"])), c["attempt"], " ".join(f"{k}={v}" for k, v in c["config"].items()),
             ui.speed(c["speedup"]), ui.speed(c["speedup_compiled"])] for name, c in sm["regime_champions"].items()],
            align="lllrrr")
    print(ui.s(f"\nrun saved to {sm['run_dir']}", "grey"))


def cmd_bench(args) -> int:
    from . import bench, loop
    from .carmy import CarmyAuthError
    args.ops = args.ops or ["masked_softmax", "add_rmsnorm"]
    unknown = [o for o in args.ops if o not in ops.OPS]
    if unknown:
        print(f"{ui.BAD} unknown kernel(s): {', '.join(unknown)}. Choose from: {', '.join(ops.OPS)}")
        return 2
    calls = len(args.ops) * len(bench.MODES) * args.repeats * args.rounds * args.k
    ui.banner("bench · loop vs best-of-N")
    print(f"{len(args.ops)} kernel(s) x 2 modes x {args.repeats} repeat(s) x {args.rounds} rounds x {args.k} drafts "
          f"= {ui.s(str(calls), 'bold')} model calls (effort {args.effort}). Leave it running; it takes a while.")
    if not args.yes and input("start? [y/N] ").strip().lower() != "y":
        return 1
    peak = judge.peak_gbps(args.backend)
    root = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    try:
        out = bench.run(args.ops, repeats=args.repeats, rounds=args.rounds, k=args.k, root=root, run_fn=loop.run,
                        on_progress=lambda m: print(ui.s("  " + m, "grey")), model=args.model, effort=args.effort,
                        backend=args.backend, peak=peak)
    except CarmyAuthError as e:
        print(f"\n{ui.BAD} {e}")
        return 2
    ui.rule("result")
    ui.table(["kernel", "mode", "vs MLX", "vs mx.compile", "unseen sizes", "round 1 → final", "verified"],
             [[r["op"], "loop" if r["mode"] == "loop" else "best-of-N", ui.speed(r["speedup"]),
               ui.speed(r["speedup_compiled"]), ui.speed(r["hidden_speedup"]),
               f"{bench._x(r['round1'])} → {bench._x(r['speedup'])}", f"{r['verified']}/{r['attempts']}"]
              for r in out["table"]])
    print(ui.s(f"\nsaved {root / 'bench.md'} (paste it into the README) and {root / 'bench.json'}", "grey"))
    return 0


def cmd_profile(args) -> int:
    from . import profile as prof
    ui.banner(f"profile · {args.model}")
    p = prof.run(args.model, args.prompt, args.gen, on_progress=lambda m: print(ui.s("  " + m, "grey")))
    sp = p.spec
    quant = f"{sp.bits}-bit weights (groups of {sp.group_size})" if sp.bits else "unquantized weights"
    print(f"\n{ui.s(sp.model, 'bold')}: {sp.layers} layers · hidden {sp.hidden} · {sp.heads} heads / "
          f"{sp.kv_heads} kv heads · head {sp.head_dim} · mlp {sp.mlp} · {quant} · activations {sp.dtype} · "
          f"norm eps {sp.norm_eps}")
    print(f"prefill {ui.s(f'{p.prefill_tps:,.0f} tok/s', 'bold')} ({p.prompt_tokens}-token prompt) · "
          f"decode {ui.s(f'{p.decode_tps:,.1f} tok/s', 'bold')} ({p.gen_tokens} tokens)")
    for title, steps, measured_ms in (("writing one word (decode)", p.decode, 1e3 / p.decode_tps),
                                      (f"reading the prompt (prefill, {p.prompt_tokens} tokens)", p.prefill,
                                       1e3 * p.prompt_tokens / p.prefill_tps)):
        ui.rule(title)
        rows = prof.table(steps, measured_ms)
        body, total = rows[:-1], rows[-1]
        ui.table(["step", "each", "x", "total", "share", "carmen"],
                 [[r["step"], f"{r['us_each']:7.1f} us", r["times"], f"{r['ms_total']:7.3f} ms",
                   ui.pct(r["share"]), ui.s(r["carmen"] or "", "green")] for r in body], align="lrrrrl")
        cov = total.get("coverage")
        print(ui.s(f"  steps add up to {total['ms_total']:.2f} ms vs {measured_ms:.2f} ms measured"
                   + (f" ({cov:.0%})" if cov else ""), "grey"))
        print("  by kind: " + "  ".join(f"{k} {v:.0%}" for k, v in prof.by_kind(steps).items()))
    launches = sum(s.per_token for s in p.decode)  # each step is at least one kernel
    print(ui.s(f"\nsmallest possible kernel on this GPU: {p.kernel_floor_us:.1f} us. ~{launches} step launches per "
               f"word (at least) -> {launches * p.kernel_floor_us / 1e3:.2f} ms of every word is launch overhead.", "grey"))
    print(ui.s(f"saved {prof.save(p, Path(args.runs))}", "grey"))
    return 0


def cmd_e2e(args) -> int:
    from . import e2e
    ui.banner(f"e2e · {args.model}")
    kernel = config = kernel_id = champions = None
    modes = [m.strip() for m in args.modes.split(",")]
    if args.calls or any(m in ("kernel", "both") for m in modes):
        if args.kernel == "golden":
            kernel = broken.golden(e2e.OP)
            config, kernel_id = kernel.configs[0], "golden reference kernel"
        elif args.kernel == "latest" and (rc := e2e.latest_regime_champions(Path(args.runs))):
            champions, run_id, sm = rc
            kernel, config = next(iter(champions.values()))
            kernel_id = f"{run_id}:" + ",".join(f"{k}={c['attempt']}" for k, c in sm["regime_champions"].items())
            print(f"kernels: a champion per regime from {ui.s(run_id, 'bold')} (targeted at {sm['target']['model']})")
            for name, c in sm["regime_champions"].items():
                print(ui.s(f"  {name}: {c['attempt']} {c['config']}, judge {c['speedup_compiled']:.2f}x vs mx.compile",
                           "grey"))
        else:
            found = e2e.latest_champion(Path(args.runs))
            if not found:
                print(f"{ui.BAD} no {e2e.OP} champion in {args.runs}: run `carmen run {e2e.OP}` or pass --kernel golden")
                return 2
            kernel, config, kernel_id, verdict = found
            print(f"kernel: {ui.s(kernel_id, 'bold')} (champion of your latest {e2e.OP} run), config {config}")
            for r in verdict.get("timing", []):
                print(ui.s(f"  on its own at {'x'.join(map(str, r['shape']))} {r['dtype']}: "
                           f"{r['speedup']:.2f}x vs MLX", "grey"))
    if args.calls:
        return _print_calls(e2e, args, kernel, config, kernel_id)
    if "stock" not in modes:
        modes = ["stock"] + modes  # every comparison is against stock
    results = e2e.run(args.model, modes, kernel, config, args.prompt, args.gen, args.repeats,
                      on_progress=lambda m: print(ui.s("  " + m, "grey")), champions=champions)
    stock = results[0]
    ui.rule("result")
    rows = []
    for r in results:
        if r.mode == "stock":
            rows.append([r.mode, f"{r.prefill_tps:,.0f}", f"{r.decode_tps:,.1f}", "", "reference"])
            continue
        same = r.tokens_match == r.tokens_total
        answers = (ui.s(f"{ui.OK} same {r.tokens_total} tokens", "green") if same else
                   ui.s(f"diverge at token {r.tokens_match + 1}", "yellow"))
        rows.append([r.mode, f"{r.prefill_tps:,.0f}", f"{r.decode_tps:,.1f}",
                     ui.speed(r.decode_tps / stock.decode_tps),
                     answers + ui.s(f"  (first logits within {r.max_logit_diff:.3g})", "grey")])
    ui.table(["mode", "prefill tok/s", "decode tok/s", "decode vs stock", "outputs vs stock"], rows, align="lrrrl")
    print(ui.s("\nfp16/bf16 rounding can flip a near-tie between two words, so a late divergence with a tiny "
               "logit difference is rounding, not a bug; an early one with a large difference is a bug.", "grey"))
    print(ui.s(f"saved {e2e.save(args.model, results, kernel_id, Path(args.runs))}", "grey"))
    return 0


def _print_calls(e2e, args, kernel, config, kernel_id) -> int:
    timings = e2e.call_bench(args.model, kernel, config, on_progress=lambda m: print(ui.s("  " + m, "grey")))
    for rows in sorted({t.rows for t in timings}):
        group = [t for t in timings if t.rows == rows]
        stock = next(t for t in group if t.variant.startswith("stock: "))
        ui.rule(f"one residual add + rmsnorm, {rows} row{'s' if rows > 1 else ''}, us per call")
        tbl = []
        for t in group:
            if t.error:
                tbl.append([t.variant, "", "", "", ui.s(t.error, "yellow")])
                continue
            tbl.append([t.variant, f"{t.total_us:.1f}", f"{t.python_us:.1f}",
                        ui.speed(stock.total_us / t.total_us) if t is not stock else "reference", ""])
        ui.table(["how", "total", "python", "vs stock", ""], tbl, align="lrrrl")
    one = {t.variant: t for t in timings if t.rows == 1}
    s, c = one["stock: x + r, then rms_norm"], one["carmen, compiled call (what e2e uses)"]
    extra = (c.total_us - s.total_us) * e2e.CALL_CHAIN / 1e3
    print(ui.s(f"\ntotal = Python + GPU per call, {e2e.CALL_CHAIN} calls in a row like one word of decode. "
               f"python = just building the call. At 1 row, carmen as e2e calls it costs {extra:+.2f} ms per word "
               f"vs stock ({e2e.CALL_CHAIN} calls).", "grey"))
    print(ui.s(f"saved {e2e.save_calls(args.model, timings, kernel_id, Path(args.runs))}", "grey"))
    return 0


def cmd_report(args) -> int:
    sm = json.loads((Path(args.run) / "summary.json").read_text())
    sm.setdefault("run_dir", args.run)
    ui.banner(f"report · {Path(args.run).name}")
    _print_summary(sm)
    return 0


def cmd_timings(args) -> int:
    """Where did the time go? Model calls vs judging vs reflection, from the run's own event log."""
    from .events import read_events
    runs = sorted(Path(args.runs).iterdir()) if not args.run else [Path(args.run)]
    run = [r for r in runs if (r / "events.jsonl").exists()][-1]
    ev = read_events(run)
    ui.banner(f"timings · {run.name}")
    rows, totals = [], {"model (Carmy)": 0.0, "judge": 0.0, "reflect": 0.0}
    t_round = t_judge = t_reflect = None
    for e in ev:
        if e["type"] == "round_started":
            t_round = e["t"]
        elif e["type"] == "attempt_submitted":
            if t_round is not None:
                dt = e["t"] - t_round
                totals["model (Carmy)"] += dt
                rows.append([f"round {e['round']}", "Carmy writing 3 drafts (parallel)", f"{dt:6.1f} s"])
                t_round = None
            t_judge = e["t"]
        elif e["type"] == "judge_result" and t_judge is not None:
            dt = e["t"] - t_judge
            totals["judge"] += dt
            rows.append(["", f"judge {e['attempt']} ({e['stage']})", f"{dt:6.1f} s"])
            t_reflect = e["t"]
        elif e["type"] in ("playbook_updated", "reflect_error") and t_reflect is not None:
            dt = e["t"] - t_reflect
            totals["reflect"] += dt
            rows.append(["", "reflector (lessons)", f"{dt:6.1f} s"])
            t_reflect = None
    ui.table(["", "step", "time"], rows, align="llr")
    ui.rule()
    total = ev[-1]["t"] - ev[0]["t"]
    print("  ".join(f"{k}: {ui.s(f'{v:.0f} s', 'bold')}" for k, v in totals.items()) + f"   total: {total:.0f} s")
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
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("ui", help="the terminal app (default when no command is given)")
    p.add_argument("--runs", default="runs")
    p.add_argument("--memory", default="memory")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(fn=cmd_ui)

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
    p.add_argument("--for", dest="for_model", metavar="MODEL",
                   help="judge at this model's shapes and number format, scored vs mx.compile, a champion per "
                        "regime (decode, prefill); e.g. qwen0.5b or an mlx-community id")
    p.add_argument("--prompt", type=int, default=512, help="prefill regime: prompt length in tokens (with --for)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("bench", help="loop vs best-of-N on the same budget, repeated, one summary table")
    p.add_argument("ops", nargs="*", metavar="op", help="kernels to bench (default: masked_softmax add_rmsnorm)")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--out", default="runs/bench")
    p.add_argument("--yes", action="store_true", help="don't ask before starting")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("profile", help="where a real model spends its time on this Mac (needs mlx-lm)")
    p.add_argument("model", nargs="?", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    p.add_argument("--prompt", type=int, default=512, help="prompt length in tokens")
    p.add_argument("--gen", type=int, default=128, help="tokens to generate")
    p.add_argument("--runs", default="runs")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("e2e", help="run a real model stock vs with carmen's changes: tok/s and same answers?")
    p.add_argument("model", nargs="?", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    p.add_argument("--modes", default="stock,plumbing,kernel,both")
    p.add_argument("--kernel", default="latest", help="latest = champions of your newest `run residual_rmsnorm --for` run (one per regime), else of your newest residual_rmsnorm run; or golden")
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--runs", default="runs")
    p.add_argument("--calls", action="store_true", help="time one kernel call several ways instead (where the time goes)")
    p.set_defaults(fn=cmd_e2e)

    p = sub.add_parser("report", help="summarize a run")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("timings", help="where a run spent its time (model vs judge)")
    p.add_argument("run", nargs="?", help="run directory (default: the latest)")
    p.add_argument("--runs", default="runs")
    p.set_defaults(fn=cmd_timings)

    p = sub.add_parser("playbook", help="show learned lessons")
    p.add_argument("--memory", default="memory")
    p.set_defaults(fn=cmd_playbook)

    args = ap.parse_args(argv)
    dotenv.load(args.env)
    if args.cmd is None:
        args = ap.parse_args(["--backend", args.backend, "--env", args.env, "ui"])
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
