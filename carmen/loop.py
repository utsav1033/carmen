"""The self-correcting loop.

    each round:
      Carmy x k, in parallel (same handoff, different angles)
      -> the judge runs every candidate (visible + fresh fuzz + corpus + secret hidden draw)
      -> keep the fastest verified kernel as champion
      -> failing fuzz inputs join the regression corpus; the playbook is credited by verdicts
      -> next round Carmy gets the champion + its per-shape profile (or the best failure + its
         diagnosis), plus a ledger of everything already tried this run and how it scored
    stop at ~90% of measured peak bandwidth, after `patience` rounds without improvement, or at `rounds`.

`mode="bon"` is the control arm: the same number of Carmy calls with no feedback,
no champion and no playbook (plain best-of-N).
"""

from __future__ import annotations

import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import carmy, judge, ops
from .backends import Kernel
from .events import Run
from .memory import Playbook
from .ops.base import Case, fuzz_cases

PEAK_TARGET = 0.90
MIN_GAIN = 1.02


def _corpus_path(memory_dir: Path, op: str) -> Path:
    return Path(memory_dir) / f"corpus-{op}.json"


def load_corpus(memory_dir: Path, op: str) -> list[Case]:
    p = _corpus_path(memory_dir, op)
    return [Case.from_json(d) for d in json.loads(p.read_text())] if p.exists() else []


def save_corpus(memory_dir: Path, op: str, cases: list[Case]) -> None:
    p = _corpus_path(memory_dir, op)
    p.parent.mkdir(parents=True, exist_ok=True)
    uniq = {c.label(): c for c in cases}
    p.write_text(json.dumps([c.to_json() for c in uniq.values()], indent=2))


def run(op_name: str, *, rounds: int = 6, k: int = 3, mode: str = "loop", model: str = carmy.DEFAULT_MODEL,
        effort: str = "high", backend: str = "metal", runs_dir: Path = Path("runs"),
        memory_dir: Path = Path("memory"), patience: int = 2, adapter=None,
        write_fn: Callable | None = None, reflect_fn: Callable | None = None,
        peak: dict | None = None, on_event: Callable[[str, dict], None] | None = None,
        start: dict | None = None, should_stop: Callable[[], bool] | None = None) -> dict:
    """`start` seeds the loop with an existing kernel ({id, kernel, verdict, feedback}): a verified
    one becomes the champion to improve, a failing one becomes the attempt to repair."""
    op = ops.get(op_name)
    write_fn = write_fn or carmy.write
    reflect_fn = reflect_fn or carmy.reflect
    learn = mode == "loop"
    peak = peak or judge.peak_gbps(backend)
    chip, peak_gbps = peak["chip"], peak["peak_gbps"]
    playbook = Playbook(Path(memory_dir) / "playbook.json")
    corpus = load_corpus(memory_dir, op.name) if learn else []
    rr = Run(runs_dir)
    emit = on_event or (lambda t, d: None)

    def log(t, **d):
        rr.log(t, **d)
        emit(t, d)

    log("run_started", op=op.name, mode=mode, model=model, effort=effort, rounds=rounds, k=k,
        chip=chip, peak_gbps=peak_gbps, backend=backend, run_dir=str(rr.dir))
    champion, last_fail, stale, attempts = None, None, 0, []
    if start is not None:
        if start["verdict"].get("correct"):
            champion = start
        else:
            last_fail = start
        log("started_from", source=start["id"], correct=bool(start["verdict"].get("correct")))

    for r in range(rounds):
        if should_stop and should_stop():
            log("stopped", reason="stopped by you")
            break
        round_seed, hidden_seed = secrets.randbits(31), secrets.randbits(31)
        rr.save(f"secrets/round-{r}.json", {"hidden_seed": hidden_seed})  # never shown to Carmy
        shown = playbook.shown(op.name) if learn else []
        pb_text = playbook.render(op.name) if learn else ""
        log("round_started", round=r, playbook=[l.id for l in shown])

        def one(i):
            return write_fn(op, model=model, effort=effort, chip=chip, peak=peak_gbps, playbook=pb_text,
                            champion=champion if learn else None, last=last_fail if learn else None,
                            variant=i, k=k, history=_ledger(attempts, champion) if learn else "")

        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = [pool.submit(one, i) for i in range(k)]
            results = []
            for i, f in enumerate(futures):
                try:
                    results.append(f.result())
                except carmy.CarmyAuthError as e:
                    log("carmy_error", round=r, attempt=i, error=str(e))
                    log("stopped", reason="the API rejected the key; fix it and rerun")
                    raise
                except Exception as e:  # a failed API call is logged, not hidden
                    log("carmy_error", round=r, attempt=i, error=str(e))
                    results.append(None)

        round_attempts = []
        for i, res in enumerate(results):
            if res is None:
                continue
            kernel, used, usage, prompt_text = res
            aid = f"{r}-{i}"
            rr.save(f"attempts/{aid}/kernel.metal", kernel.source)
            rr.save(f"attempts/{aid}/kernel.json", kernel.to_json())
            rr.save(f"attempts/{aid}/prompt.txt", prompt_text)
            log("attempt_submitted", round=r, attempt=aid, digest=kernel.digest(), plan=kernel.plan,
                configs=kernel.configs, lessons_used=used, usage=usage)
            v = judge.judge(op.name, kernel, adapter=adapter, backend=backend, round_seed=round_seed,
                            hidden_seed=hidden_seed, corpus=corpus, peak=peak_gbps)
            fb = judge.feedback(v)
            rr.save(f"attempts/{aid}/verdict.json", v)
            rr.save(f"attempts/{aid}/feedback.txt", fb)
            log("judge_result", round=r, attempt=aid, stage=v["stage"], correct=v["correct"],
                naive_pass=v.get("naive_pass"), speedup=v.get("speedup_geomean"),
                speedup_compiled=v.get("speedup_compiled_geomean"),
                pct_peak_min=v.get("pct_peak_min"), hidden=_hidden_summary(v))
            if learn:
                playbook.credit([l.id for l in shown], used, bool(v["correct"]), op.name)
            entry = {"id": aid, "kernel": kernel.to_json(), "verdict": v, "feedback": fb}
            round_attempts.append(entry)
            attempts.append(entry)

        if learn:
            fuzz = {c.label(): c for c in fuzz_cases(op, round_seed)}
            new = [fuzz[f["case"]] for a in round_attempts for c in a["verdict"].get("configs", [])
                   for f in c["failures"] if f["case"] in fuzz]
            if new:
                corpus = list({c.label(): c for c in corpus + new}.values())
                save_corpus(memory_dir, op.name, corpus)
                log("corpus_grew", round=r, added=[c.label() for c in new])

        improved = False
        for a in sorted((a for a in round_attempts if a["verdict"]["correct"]),
                        key=lambda a: -a["verdict"]["speedup_geomean"]):
            if champion is None or a["verdict"]["speedup_geomean"] > champion["verdict"]["speedup_geomean"] * MIN_GAIN:
                champion, improved = a, True
                log("champion", round=r, attempt=a["id"], speedup=a["verdict"]["speedup_geomean"],
                    pct_peak_min=a["verdict"].get("pct_peak_min"))
            break
        if champion is None and round_attempts:
            # Repair from the attempt that got furthest: correctness failures beat compile errors,
            # and fewer failing tests beat more.
            last_fail = min(round_attempts, key=lambda a: (
                a["verdict"]["stage"] != "correctness",
                sum(c["n_failures"] for c in a["verdict"].get("configs", [])) or 10**6))

        # Lessons come only from rounds that proved something: a new champion (the first verified
        # kernel, or a faster one). A round that changed nothing has nothing new to teach.
        if learn and improved:
            try:
                proposals = reflect_fn(op, _round_summary(round_attempts, champion))
            except Exception as e:
                log("reflect_error", round=r, error=str(e))
                proposals = []
            verified = {a["id"] for a in round_attempts if a["verdict"]["correct"]}
            rejected = playbook.add(proposals, op.name, verified)
            playbook.save()
            log("playbook_updated", round=r, proposed=len(proposals),
                rejected=[{"text": p.get("text", ""), "reason": why} for p, why in rejected],
                lessons=[l.__dict__ for l in playbook.lessons.values()])

        stale = 0 if improved else stale + 1
        pct = champion["verdict"].get("pct_peak_min") if champion else None
        log("round_finished", round=r, improved=improved, champion=champion["id"] if champion else None,
            pct_peak_min=pct)
        if pct is not None and pct >= PEAK_TARGET:
            log("stopped", reason=f"champion reached {pct:.0%} of peak bandwidth")
            break
        if champion is not None and stale >= patience:
            log("stopped", reason=f"no improvement for {patience} rounds")
            break

    summary = summarize(attempts, champion)
    summary["run_dir"] = str(rr.dir)
    rr.save("summary.json", summary)
    log("run_finished", **{k_: v_ for k_, v_ in summary.items() if k_ != "run_dir"})
    return summary


LEDGER_MAX = 12


def _scores(v: dict) -> str:
    return ", ".join(f"{r['shape'][0]}x{r['shape'][1]} {r['dtype'].replace('float', 'f')} {r['speedup']:.2f}x"
                     for r in v.get("timing", []))


def _ledger(attempts: list[dict], champion: dict | None) -> str:
    """One line per earlier attempt this run: what it tried and what the judge measured."""
    lines = []
    for a in attempts[-LEDGER_MAX:]:
        v = a["verdict"]
        plan = " ".join((a["kernel"].get("plan") or "").split())[:240]
        if v["correct"]:
            tag = " (current champion)" if champion and a["id"] == champion["id"] else ""
            res = f"verified, geomean {v['speedup_geomean']:.2f}x{tag}: {_scores(v)}"
        else:
            first = next((c["failures"][0] for c in v.get("configs", []) if c["failures"]), None)
            why = "; ".join(first["patterns"]) if first else ((v.get("error") or "").splitlines() or [""])[0]
            res = f"rejected at {v['stage']}: {why}"[:200]
        lines.append(f"- {a['id']}: {plan}\n  -> {res}")
    return "\n".join(lines)


def _hidden_summary(v: dict) -> dict | None:
    h = v.get("hidden")
    if not h:
        return None
    return {"total": h["total"], "failed": h["failed"], "speedup": h.get("speedup_geomean")}


def _round_summary(attempts: list[dict], champion: dict | None) -> str:
    lines = []
    for a in attempts:
        v = a["verdict"]
        status = "VERIFIED" if v["correct"] else f"REJECTED ({v['stage']})"
        speed = f", {v['speedup_geomean']:.2f}x vs stock MLX" if v["correct"] else ""
        lines.append(f"attempt {a['id']}: {status}{speed}\nplan: {a['kernel']['plan']}\n"
                     f"source:\n{a['kernel']['source']}\njudge feedback:\n{a['feedback']}\n")
    if champion:
        lines.append(f"Current champion: attempt {champion['id']}.")
    return "\n".join(lines)


def summarize(attempts: list[dict], champion: dict | None) -> dict:
    """The numbers carmen reports. Hidden results appear here, never in Carmy's prompts."""
    compiled = [a for a in attempts if a["verdict"]["stage"] not in ("static", "compile", "crash", "timeout")]
    correct = [a for a in attempts if a["verdict"]["correct"]]
    naive_pass = [a for a in compiled if a["verdict"].get("naive_pass")]
    fooled = [a for a in naive_pass if not a["verdict"].get("full_pass_default_config")]
    first = [a for a in attempts if a["id"].startswith("0-")]
    hidden = [a["verdict"]["hidden"] for a in correct if a["verdict"].get("hidden")]
    out = {
        "attempts": len(attempts),
        "compiled": len(compiled),
        "verified": len(correct),
        "first_round_verified": f"{sum(a['verdict']['correct'] for a in first)}/{len(first)}",
        "naive_pass": len(naive_pass),
        "naive_pass_but_wrong": len(fooled),
        "hidden_clean": f"{sum(h['failed'] == 0 for h in hidden)}/{len(hidden)}",
        "champion": champion["id"] if champion else None,
    }
    if champion:
        v = champion["verdict"]
        out.update(speedup=v["speedup_geomean"], speedup_compiled=v.get("speedup_compiled_geomean"),
                   pct_peak_min=v.get("pct_peak_min"),
                   best_config=v["best_config"], default_config_speedup=v.get("default_config_speedup"))
        if v.get("hidden"):
            out.update(hidden_speedup=v["hidden"].get("speedup_geomean"),
                       hidden_failed=v["hidden"]["failed"], hidden_total=v["hidden"]["total"])
    return out
