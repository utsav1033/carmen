"""carmen's terminal app: browse kernels and runs, judge, improve, and cook, in one window.

    ┌ logo · chip · route · stats ────────────────────────────────────┐
    │ navigator │ kernel source             │ details + actions       │
    │           ├ activity log ─────────────┤                         │
    └ keys ───────────────────────────────────────────────────────────┘

The app only reads what the harness writes (runs/, memory/) and calls the same
functions the CLI does. Anything that touches the GPU or the model runs in a worker
thread, one at a time.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, RichLog, Static, Tree

from . import broken, carmy, judge, loop, ops
from .backends import Kernel

# ── palette: dark glass, white type, green and purple used sparingly ─────────────
BG, PANEL, EDGE, EDGE_HI = "#0c0d11", "#111218", "#262235", "#8f76d6"
INK, DIM, FAINT = "#e8e6ef", "#8b8a9b", "#4d4b5c"
GREEN, PURPLE, RED, AMBER = "#7ee2a8", "#b39afc", "#ff7b8a", "#f2c46d"
WHITE, SHADE = "#ffffff", "#5c5c66"

LOGO = [
    " ██████╗ █████╗ ██████╗ ███╗   ███╗███████╗███╗   ██╗",
    "██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝████╗  ██║",
    "██║     ███████║██████╔╝██╔████╔██║█████╗  ██╔██╗ ██║",
    "██║     ██╔══██║██╔══██╗██║╚██╔╝██║██╔══╝  ██║╚██╗██║",
    "╚██████╗██║  ██║██║  ██║██║ ╚═╝ ██║███████╗██║ ╚████║",
    " ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝",
]


def logo() -> Text:
    """Solid white faces with a soft grey edge underneath: reads as extruded 3D type."""
    t = Text()
    for i, line in enumerate(LOGO):
        for ch in line:
            t.append(ch, style=f"bold {WHITE}" if ch == "█" else SHADE if ch != " " else "")
        if i < len(LOGO) - 1:
            t.append("\n")
    return t


# ── data ─────────────────────────────────────────────────────────────────────────
@dataclass
class Item:
    op: str
    title: str
    kernel: Kernel
    verdict: dict | None = None
    feedback: str = ""
    source: str = "golden"  # golden | attempt
    run: str | None = None
    attempt: str | None = None


@dataclass
class RunInfo:
    dir: Path
    op: str
    mode: str
    summary: dict
    attempts: list[Item] = field(default_factory=list)


def load_runs(runs_dir: Path) -> list[RunInfo]:
    out = []
    if not runs_dir.is_dir():
        return out
    for d in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        ev = d / "events.jsonl"
        if not ev.exists():
            continue
        first = json.loads(ev.read_text().splitlines()[0])
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        info = RunInfo(d, first.get("op", "?"), first.get("mode", "?"), summary)
        adir = d / "attempts"
        for a in sorted(adir.iterdir(), key=lambda p: [int(x) for x in p.name.split("-")]) if adir.is_dir() else []:
            if not (a / "kernel.json").exists():
                continue
            v = json.loads((a / "verdict.json").read_text()) if (a / "verdict.json").exists() else None
            fb = (a / "feedback.txt").read_text() if (a / "feedback.txt").exists() else ""
            info.attempts.append(Item(info.op, f"{d.name} · attempt {a.name}",
                                      Kernel.from_json(json.loads((a / "kernel.json").read_text())),
                                      v, fb, "attempt", d.name, a.name))
        out.append(info)
    return out


def cached_peak(backend: str) -> dict | None:
    p = judge.CACHE / f"peak-{backend}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── rendering helpers ────────────────────────────────────────────────────────────
def chip(v: dict | None) -> Text:
    if v is None:
        return Text(" ○ not judged ", style=f"{DIM} on #1a1a24")
    if v.get("correct"):
        return Text(" ✓ verified ", style=f"bold #0c0d11 on {GREEN}")
    return Text(f" ✗ {v.get('stage', 'wrong')} ", style=f"bold #0c0d11 on {RED}")


def speed(x) -> Text:
    if x is None:
        return Text("—", style=DIM)
    return Text(f"{x:.2f}×", style=f"bold {GREEN}" if x >= 1.05 else AMBER if x >= 0.95 else RED)


def bar(frac, width: int = 8) -> Text:
    if frac is None:
        return Text("—", style=DIM)
    n = max(0, min(width, round(frac * width)))
    return Text("█" * n, style=PURPLE) + Text("░" * (width - n), style=FAINT) + Text(f" {frac:.0%}", style=INK)


def tile(label: str, value: Text) -> Panel:
    return Panel(Align.center(value), subtitle=Text(label, style=DIM), border_style=EDGE, box=box.ROUNDED,
                 padding=(0, 1))


def details(item: Item) -> Group:
    v = item.verdict
    head = Text.assemble((item.op, f"bold {INK}"), ("  ·  ", FAINT), (item.title, DIM))
    plan = Text(item.kernel.plan or "no plan recorded", style=INK if item.kernel.plan else DIM)
    cfgs = Text("configs  " + "   ".join(" ".join(f"{k}={x}" for k, x in c.items()) for c in item.kernel.configs),
                style=DIM)
    parts = [head, chip(v), Text(""), plan, cfgs, Text("")]

    h = (v or {}).get("hidden") or {}
    hidden = Text(f"{h['total'] - h['failed']}/{h['total']}", style=f"bold {GREEN if not h['failed'] else RED}") \
        if h else Text("—", style=DIM)
    naive = (v or {}).get("naive_pass")
    naive_t = Text("—", style=DIM) if naive is None else Text("passes" if naive else "fails",
                                                            style=AMBER if naive and not v.get("correct") else INK)
    tiles = Table.grid(expand=True)
    for _ in range(4):
        tiles.add_column(ratio=1)
    pct = (v or {}).get("pct_peak_min")
    tiles.add_row(tile("vs mlx", speed((v or {}).get("speedup_geomean"))),
                  tile("peak", Text("—" if pct is None else f"{pct:.0%}", style=f"bold {PURPLE}")),
                  tile("hidden", hidden), tile("naive", naive_t))
    parts.append(tiles)

    if v and v.get("timing"):
        t = Table(box=box.SIMPLE_HEAD, expand=True, header_style=DIM, border_style=EDGE, pad_edge=False)
        for col, j in (("shape", "left"), ("", "left"), ("ms", "right"), ("mlx", "right"),
                       ("speed", "right"), ("of peak", "left")):
            t.add_column(col, justify=j, no_wrap=True)
        for r in v["timing"]:
            t.add_row(f"{r['shape'][0]}×{r['shape'][1]}", r["dtype"].replace("float", "f"), f"{r['ms']:.3f}",
                      f"{r['baseline_ms']:.3f}", speed(r["speedup"]), bar(r.get("pct_peak")))
        parts += [Text(""), t]
    if item.feedback:
        parts += [Text(""), Text("what Carmy sees", style=f"bold {PURPLE}"), Text(item.feedback, style=INK)]
    elif v is None:
        parts += [Text(""), Text("press  j  to judge this kernel on your GPU", style=DIM)]
    return Group(*parts)


# ── app ──────────────────────────────────────────────────────────────────────────
class Carmen(App):
    TITLE = "carmen"
    CSS = f"""
    Screen {{ background: {BG}; color: {INK}; }}
    #top {{ height: 8; padding: 1 2 0 2; background: {BG}; }}
    #logo {{ width: 58; }}
    #meta {{ padding: 1 0 0 2; color: {DIM}; }}
    #body {{ height: 1fr; padding: 0 1; }}
    .pane {{ background: {PANEL}; border: round {EDGE}; border-title-color: {PURPLE}; border-title-style: bold;
             border-subtitle-color: {DIM}; }}
    .pane:focus-within {{ border: round {EDGE_HI}; }}
    #nav {{ width: 40; overflow-x: hidden; }}
    #center {{ width: 1fr; }}
    #code {{ height: 1fr; }}
    #log {{ height: 11; }}
    #right {{ width: 72; }}
    #details {{ height: 1fr; padding: 0 1; }}
    #actions {{ height: 3; padding: 0 1; }}
    Button {{ min-width: 10; height: 3; margin-right: 1; background: #1a1a24; color: {INK}; border: tall {EDGE}; }}
    Button:hover {{ background: #221f31; border: tall {EDGE_HI}; }}
    Button.primary {{ color: {GREEN}; }}
    Button.accent {{ color: {PURPLE}; }}
    Tree {{ background: {PANEL}; padding: 0 1; }}
    Tree > .tree--cursor {{ background: #231f35; color: {INK}; text-style: bold; }}
    Tree > .tree--guides {{ color: {FAINT}; }}
    RichLog {{ background: {PANEL}; padding: 0 1; }}
    Footer {{ background: {BG}; }}
    Footer > .footer--key {{ color: {PURPLE}; background: {BG}; }}
    """
    BINDINGS = [
        Binding("j", "judge", "judge"),
        Binding("i", "improve", "improve with Carmy"),
        Binding("r", "run", "cook (3 rounds)"),
        Binding("b", "broken", "test the judge"),
        Binding("f5", "refresh", "refresh"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, backend: str = "metal", runs_dir: Path = Path("runs"), memory_dir: Path = Path("memory"),
                 model: str = carmy.DEFAULT_MODEL):
        super().__init__()
        self.backend, self.runs_dir, self.memory_dir, self.model = backend, Path(runs_dir), Path(memory_dir), model
        self.golden = {name: Item(name, "golden kernel", broken.golden(name)) for name in ops.OPS}
        self.current: Item = self.golden["masked_softmax"]
        self.busy = False

    # layout
    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static(logo(), id="logo")
            yield Static(id="meta")
        with Horizontal(id="body"):
            nav = Tree("carmen", id="nav", classes="pane")
            nav.show_root = False
            yield nav
            with Vertical(id="center"):
                with VerticalScroll(id="code", classes="pane"):
                    yield Static(id="source")
                yield RichLog(id="log", classes="pane", markup=False, wrap=True)
            with Vertical(id="right", classes="pane"):
                with VerticalScroll(id="details"):
                    yield Static(id="info")
                with Horizontal(id="actions"):
                    yield Button("judge", id="judge", classes="primary")
                    yield Button("improve", id="improve", classes="accent")
                    yield Button("cook", id="run", classes="accent")
                    yield Button("test judge", id="broken")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#nav").border_title = "kernels & runs"
        self.query_one("#code").border_title = "kernel"
        self.query_one("#log").border_title = "activity"
        self.query_one("#right").border_title = "details"
        self.refresh_nav()
        self.show(self.current)
        self.log_line("ready. pick a kernel on the left; j judges it, i asks Carmy to improve it, r cooks from scratch.",
                      DIM)

    # rendering
    def refresh_meta(self, runs: list[RunInfo]) -> None:
        peak = cached_peak(self.backend)
        gpu = f"{peak['chip']} · {peak['peak_gbps']:.0f} GB/s peak" if peak else "GPU not measured yet (carmen peak)"
        best = {}
        for r in runs:
            s = r.summary.get("speedup")
            if s and s > best.get(r.op, 0):
                best[r.op] = s
        lines = Text()
        lines.append("let the model cook. trust nothing it can't prove.\n\n", style=f"italic {INK}")
        lines.append("gpu    ", style=FAINT).append(gpu + "\n", style=INK)
        lines.append("model  ", style=FAINT).append(f"{self.model}\n", style=INK)
        lines.append("runs   ", style=FAINT).append(f"{len(runs)}", style=f"bold {PURPLE}")
        for op_name, s in best.items():
            lines.append("   best ", style=FAINT).append(f"{op_name} ", style=INK).append(f"{s:.2f}×", style=f"bold {GREEN}")
        self.query_one("#meta", Static).update(lines)

    def refresh_nav(self) -> None:
        runs = load_runs(self.runs_dir)
        self.refresh_meta(runs)
        tree: Tree = self.query_one("#nav", Tree)
        tree.clear()
        g = tree.root.add(Text("golden kernels", style=f"bold {PURPLE}"), expand=True)
        for item in self.golden.values():
            g.add_leaf(Text.assemble(("◆ ", PURPLE), (item.op, INK)), data=item)
        rn = tree.root.add(Text(f"runs ({len(runs)})", style=f"bold {PURPLE}"), expand=True)
        if not runs:
            rn.add_leaf(Text("none yet: press r", style=DIM))
        for r in runs:
            s = r.summary.get("speedup")
            label = Text.assemble((f"{r.dir.name[9:11]}:{r.dir.name[11:13]} ", DIM),
                                  (r.op, INK), (" bon" if r.mode == "bon" else "", FAINT))
            if s:
                label.append(f"  {s:.2f}×", style=GREEN)
            node = rn.add(label)
            for a in r.attempts:
                ok = a.verdict and a.verdict.get("correct")
                mark = ("✓ ", GREEN) if ok else ("✗ ", RED) if a.verdict else ("○ ", DIM)
                sp = f"  {a.verdict['speedup_geomean']:.2f}×" if ok else ""
                node.add_leaf(Text.assemble(mark, (a.attempt, INK), (sp, DIM)), data=a)

    def show(self, item: Item) -> None:
        self.current = item
        src = item.kernel.source.rstrip() + ("\n\n// header\n" + item.kernel.header if item.kernel.header else "")
        self.query_one("#source", Static).update(
            Syntax(src, "cpp", theme="one-dark", line_numbers=True, background_color=PANEL, word_wrap=False))
        self.query_one("#code").border_subtitle = f"{item.op} · {item.title}"
        self.query_one("#info", Static).update(details(item))

    def log_line(self, msg: str, style: str = INK) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    # events
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if isinstance(event.node.data, Item):
            self.show(event.node.data)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        getattr(self, f"action_{event.button.id}")()

    def action_refresh(self) -> None:
        self.refresh_nav()

    def _start(self, what: str) -> bool:
        if self.busy:
            self.notify("already working; one GPU job at a time", severity="warning")
            return False
        self.busy = True
        self.log_line(f"▸ {what}", f"bold {PURPLE}")
        return True

    def _done(self) -> None:
        self.busy = False
        self.refresh_nav()
        self.show(self.current)

    # actions
    def action_judge(self) -> None:
        if self._start(f"judging {self.current.op} · {self.current.title}"):
            self._judge_worker(self.current)

    def action_improve(self) -> None:
        if self._start(f"asking Carmy to improve {self.current.op} · {self.current.title}"):
            self._improve_worker(self.current)

    def action_run(self) -> None:
        if self._start(f"cooking {self.current.op}: 3 rounds × 3 drafts"):
            self._loop_worker(self.current.op, None, rounds=3)

    def action_broken(self) -> None:
        if self._start(f"testing the judge against seeded bugs in {self.current.op}"):
            self._broken_worker(self.current.op)

    # workers (thread): everything here must talk to the UI through call_from_thread
    def _say(self, msg: str, style: str = INK) -> None:
        self.call_from_thread(self.log_line, msg, style)

    def _judge(self, item: Item) -> None:
        peak = judge.peak_gbps(self.backend)["peak_gbps"]
        v = judge.judge(item.op, item.kernel, backend=self.backend, peak=peak, round_seed=0,
                        hidden_seed=secrets.randbits(31))
        item.verdict, item.feedback = v, judge.feedback(v)
        if v.get("correct"):
            h = v.get("hidden") or {}
            self._say(f"  ✓ verified  {v['speedup_geomean']:.2f}× vs MLX  ·  hidden {h.get('total', 0) - h.get('failed', 0)}"
                      f"/{h.get('total', 0)}", GREEN)
        else:
            self._say(f"  ✗ {v['stage']}: {(v.get('error') or item.feedback).splitlines()[0][:160]}", RED)

    @work(thread=True, group="gpu")
    def _judge_worker(self, item: Item) -> None:
        try:
            self._judge(item)
        except Exception as e:
            self._say(f"  ! {e}", RED)
        finally:
            self.call_from_thread(self._done)

    @work(thread=True, group="gpu")
    def _improve_worker(self, item: Item) -> None:
        try:
            if item.verdict is None:
                self._say("  judging it first, so Carmy gets real feedback", DIM)
                self._judge(item)
            start = {"id": f"{item.run or 'golden'}:{item.attempt or item.op}", "kernel": item.kernel.to_json(),
                     "verdict": item.verdict, "feedback": item.feedback}
            self._loop(item.op, start, rounds=1)
        except Exception as e:
            self._say(f"  ! {e}", RED)
        finally:
            self.call_from_thread(self._done)

    @work(thread=True, group="gpu")
    def _loop_worker(self, op: str, start, rounds: int) -> None:
        try:
            self._loop(op, start, rounds)
        except Exception as e:
            self._say(f"  ! {e}", RED)
        finally:
            self.call_from_thread(self._done)

    def _loop(self, op: str, start, rounds: int) -> None:
        def on_event(t, d):
            if t == "round_started":
                self._say(f"  round {d['round']}: Carmy is writing 3 drafts…", DIM)
            elif t == "attempt_submitted":
                self._say(f"  {d['attempt']}  {(d['plan'] or '')[:110]}", INK)
            elif t == "judge_result":
                if d["correct"]:
                    h = d.get("hidden") or {}
                    self._say(f"     ✓ {d['speedup']:.2f}×  ·  {(d['pct_peak_min'] or 0):.0%} of peak  ·  hidden "
                              f"{h.get('total', 0) - h.get('failed', 0)}/{h.get('total', 0)}", GREEN)
                else:
                    self._say(f"     ✗ {d['stage']}", RED)
            elif t == "champion":
                self._say(f"  ★ champion {d['attempt']}  {d['speedup']:.2f}×", f"bold {PURPLE}")
            elif t in ("carmy_error", "reflect_error"):
                self._say(f"  ! {d['error'][:200]}", RED)
            elif t == "stopped":
                self._say(f"  stopped: {d['reason']}", DIM)

        sm = loop.run(op, rounds=rounds, k=3, model=self.model, backend=self.backend, runs_dir=self.runs_dir,
                      memory_dir=self.memory_dir, peak=judge.peak_gbps(self.backend), on_event=on_event, start=start)
        best = f"{sm['speedup']:.2f}×" if sm.get("speedup") else "no verified kernel"
        self._say(f"  done: {sm['verified']}/{sm['attempts']} verified · best {best} · saved to {sm['run_dir']}",
                  f"bold {GREEN}")

    @work(thread=True, group="gpu")
    def _broken_worker(self, op: str) -> None:
        try:
            g = judge.judge(op, broken.golden(op), backend=self.backend, skip_timing=True)
            if not g["correct"]:
                self._say(f"  ✗ the golden kernel fails, so fix the tests first: {judge.feedback(g)[:200]}", RED)
                return
            self._say("  golden kernel ✓ verified", GREEN)
            killed = fooled = 0
            ms = broken.mutants(op)
            for m, kernel in ms:
                v = judge.judge(op, kernel, backend=self.backend, skip_timing=True)
                dead = not v["correct"]
                killed += dead
                fooled += bool(v.get("naive_pass_strict")) and dead
                naive = "naive@1e-4 passes" if v.get("naive_pass_strict") else "naive@1e-4 fails"
                self._say(f"  {'✓ killed  ' if dead else '✗ SURVIVED'}  {m.name:<22} {m.family:<10} {naive}",
                          GREEN if dead else RED)
            self._say(f"  carmen killed {killed}/{len(ms)}; a KernelBench-style check (1e-4) would have accepted "
                      f"{fooled}/{len(ms)} of them", f"bold {PURPLE}")
        except Exception as e:
            self._say(f"  ! {e}", RED)
        finally:
            self.call_from_thread(self._done)


def main(backend: str = "metal", runs_dir: str = "runs", memory_dir: str = "memory",
         model: str = carmy.DEFAULT_MODEL) -> None:
    Carmen(backend, Path(runs_dir), Path(memory_dir), model).run()
