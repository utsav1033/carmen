"""carmen's terminal app. One path through it: pick an op -> cook -> inspect -> plate.

    home      what are we cooking? (ops with their best results)
    kitchen   live: stage rail, three draft cards, champion strip
    inspect   one kernel: code | verdict, diff vs champion, improve from here
    plated    the result: best kernel, proof, export
    trust     the judge vs seeded bugs          history   past runs

The app never computes anything itself: it calls the same loop and judge as the CLI,
in a worker thread, and renders the events they emit.
"""

from __future__ import annotations

import difflib
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from . import broken, carmy, judge, loop, ops
from .backends import Kernel

# ── palette: charcoal and white, green only for "verified", red only for failures ─
BG, PANEL, EDGE, EDGE_HI = "#0b0b0c", "#121214", "#242427", "#4a4a50"
WHITE, INK, DIM, FAINT = "#ffffff", "#e4e4e7", "#8b8b92", "#4a4a50"
SHADE = "#5c5c66"
GREEN, RED, AMBER = "#6fdc9c", "#ff6b6b", "#e5c07b"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

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
    run: str | None = None
    attempt: str | None = None


@dataclass
class RunInfo:
    dir: Path
    op: str
    mode: str
    summary: dict
    attempts: list[Item] = field(default_factory=list)


def load_attempt(run_dir: Path, aid: str, op: str) -> Item | None:
    a = Path(run_dir) / "attempts" / aid
    if not (a / "kernel.json").exists():
        return None
    v = json.loads((a / "verdict.json").read_text()) if (a / "verdict.json").exists() else None
    fb = (a / "feedback.txt").read_text() if (a / "feedback.txt").exists() else ""
    return Item(op, f"attempt {aid}", Kernel.from_json(json.loads((a / "kernel.json").read_text())), v, fb,
                Path(run_dir).name, aid)


def load_runs(runs_dir: Path) -> list[RunInfo]:
    out = []
    if not Path(runs_dir).is_dir():
        return out
    for d in sorted((p for p in Path(runs_dir).iterdir() if p.is_dir()), reverse=True):
        ev = d / "events.jsonl"
        if not ev.exists() or not ev.read_text().strip():
            continue
        first = json.loads(ev.read_text().splitlines()[0])
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        info = RunInfo(d, first.get("op", "?"), first.get("mode", "?"), summary)
        adir = d / "attempts"
        names = sorted((p.name for p in adir.iterdir()), key=lambda s: [int(x) for x in s.split("-")]) \
            if adir.is_dir() else []
        info.attempts = [a for a in (load_attempt(d, n, info.op) for n in names) if a]
        out.append(info)
    return out


def cached_peak(backend: str) -> dict | None:
    p = judge.CACHE / f"peak-{backend}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── small renderers ──────────────────────────────────────────────────────────────
def speed(x, bold: bool = True) -> Text:
    if x is None:
        return Text("—", style=DIM)
    color = GREEN if x >= 1.05 else INK if x >= 0.95 else RED
    return Text(f"{x:.2f}×", style=f"{'bold ' if bold else ''}{color}")


def bar(frac, width: int = 10) -> Text:
    if frac is None:
        return Text("—", style=DIM)
    n = max(0, min(width, round(frac * width)))
    return Text("━" * n, style=INK) + Text("━" * (width - n), style=EDGE) + Text(f" {frac:.0%}", style=DIM)


def plain_failure(v: dict) -> str:
    """One human sentence for why a kernel was rejected."""
    if v.get("stage") in ("static", "compile", "timeout", "crash"):
        what = {"static": "rejected before running", "compile": "doesn't compile",
                "timeout": "hung the GPU", "crash": "crashed the GPU"}[v["stage"]]
        return what
    for c in v.get("configs", []):
        if c["failures"]:
            f = c["failures"][0]
            pats = " ".join(f["patterns"])
            if "tail" in f.get("where", ""):
                what = "tail bug"
            elif "race" in pats:
                what = "race condition"
            elif "past the end" in pats:
                what = "writes past the end"
            elif "NaN" in pats:
                what = "NaN / unwritten output"
            else:
                what = "wrong values"
            small = f.get("smallest_failing")
            return what + (f", fails at {small}" if small else "")
    return "wrong"


def verdict_line(v: dict | None) -> Text:
    if v is None:
        return Text("not judged yet", style=DIM)
    if v.get("correct"):
        h = v.get("hidden") or {}
        t = Text("✓ verified", style=f"bold {GREEN}")
        if h:
            t.append(f"   hidden {h['total'] - h['failed']}/{h['total']}", style=GREEN if not h["failed"] else RED)
        return t
    return Text(f"✗ {plain_failure(v)}", style=f"bold {RED}")


def verdict_panel(item: Item) -> Group:
    v = item.verdict
    parts = [Text.assemble((item.op, f"bold {WHITE}"), ("  ·  ", FAINT), (item.title, DIM)), verdict_line(v), Text("")]
    parts.append(Text(item.kernel.plan or "no plan recorded", style=INK if item.kernel.plan else DIM))
    parts.append(Text("configs  " + "   ".join(" ".join(f"{k}={x}" for k, x in c.items())
                                              for c in item.kernel.configs), style=DIM))
    if v and v.get("correct"):
        s = Table.grid(padding=(0, 3))
        for _ in range(5):
            s.add_column()
        h = v.get("hidden") or {}
        pct = v.get("pct_peak_min")
        s.add_row(Text("vs mlx", style=DIM), Text("vs mx.compile", style=DIM), Text("on hidden sizes", style=DIM),
                  Text("worst % of peak", style=DIM), Text("naive check", style=DIM))
        s.add_row(speed(v.get("speedup_geomean")), speed(v.get("speedup_compiled_geomean")),
                  speed(h.get("speedup_geomean")),
                  Text("—" if pct is None else f"{pct:.0%}", style=f"bold {INK}"),
                  Text("passes" if v.get("naive_pass") else "fails", style=INK))
        parts += [Text(""), s]
        if v.get("suspect_timing"):
            parts.append(Text("⚠ faster than the memory system allows: timing or kernel is suspect", style=AMBER))
        t = Table(box=box.SIMPLE_HEAD, expand=True, header_style=DIM, border_style=EDGE, pad_edge=False)
        for col, j in (("shape", "left"), ("", "left"), ("ms", "right"), ("mlx ms", "right"),
                       ("vs mlx", "right"), ("vs compile", "right"), ("of peak", "left")):
            t.add_column(col, justify=j, no_wrap=True)
        for r in v.get("timing", []):
            t.add_row(f"{r['shape'][0]}×{r['shape'][1]}", r["dtype"].replace("float", "f"), f"{r['ms']:.3f}",
                      f"{r['baseline_ms']:.3f}", speed(r["speedup"], bold=False),
                      speed(r.get("speedup_compiled"), bold=False), bar(r.get("pct_peak")))
        parts += [Text(""), t]
    if item.feedback:
        parts += [Text(""), Text("what Carmy was told", style=f"bold {INK}"), Text(item.feedback, style=DIM)]
    return Group(*parts)


def code_view(item: Item) -> Syntax:
    src = item.kernel.source.rstrip() + ("\n\n// header\n" + item.kernel.header if item.kernel.header else "")
    return Syntax(src, "cpp", theme="github-dark", line_numbers=True, background_color=PANEL)


CSS = f"""
Screen {{ background: {BG}; color: {INK}; }}
.pane {{ background: {PANEL}; border: round {EDGE}; border-title-color: {DIM}; border-subtitle-color: {FAINT};
         padding: 0 1; }}
.pane:focus-within {{ border: round {EDGE_HI}; }}
#top {{ height: 8; padding: 1 2 0 2; }}
#logo {{ width: 58; }}
#meta {{ padding: 1 0 0 2; color: {DIM}; }}
Footer, FooterKey {{ background: {BG}; }}
FooterKey .footer-key--key {{ color: {WHITE}; background: {BG}; text-style: bold; }}
FooterKey .footer-key--description {{ color: {DIM}; background: {BG}; }}
OptionList {{ background: {PANEL}; border: round {EDGE}; padding: 1 1; }}
OptionList:focus {{ border: round {EDGE_HI}; }}
OptionList > .option-list--option-highlighted {{ background: #1c1c20; color: {WHITE}; text-style: bold; }}
OptionList:focus > .option-list--option-highlighted {{ background: #26262b; color: {WHITE}; text-style: bold; }}
#intro {{ margin: 1 2 0 2; height: auto; padding: 0 2; }}
#ops {{ margin: 1 2 0 2; height: auto; max-height: 45%; }}
HomeScreen #ops {{ padding: 0 1; }}
#below {{ height: 1fr; min-height: 11; margin: 1 2 0 2; }}
#loop {{ width: 56; height: 100%; padding: 0 2; }}
#about {{ width: 1fr; height: 100%; margin-left: 1; padding: 0 2; }}
.crumb {{ height: 1; margin: 0 2; }}
#judgelog {{ height: 1fr; margin: 1 2 0 2; }}
#rail {{ height: 3; padding: 1 2 0 2; }}
#cards {{ height: auto; padding: 0 1; }}
.card {{ width: 1fr; height: 12; margin: 0 1; background: {PANEL}; border: round {EDGE}; padding: 1 2; }}
.card:focus {{ border: round {EDGE_HI}; }}
.card.win {{ border: round {GREEN}; }}
#strip {{ height: 5; margin: 0 2; }}
#code {{ width: 1fr; }}
#verdict {{ width: 78; }}
#plated {{ margin: 1 2; height: 1fr; padding: 1 3; }}
#trust {{ margin: 1 2; height: 1fr; padding: 1 2; }}
"""


class Header(Horizontal):
    def __init__(self):
        super().__init__(id="top")

    def compose(self) -> ComposeResult:
        yield Static(logo(), id="logo")
        yield Static(self.meta(), id="meta")

    def meta(self) -> Text:
        app: Carmen = self.app
        peak = cached_peak(app.backend)
        gpu = f"{peak['chip']} · {peak['peak_gbps']:.0f} GB/s peak" if peak else "GPU not measured yet"
        t = Text("let the model cook. trust nothing it can't prove.\n\n", style=f"italic {INK}")
        t.append("gpu    ", style=FAINT).append(gpu + "\n", style=INK)
        t.append("model  ", style=FAINT).append(app.model, style=INK)
        return t


FLOW = ("home", "cook", "inspect", "plated")


def crumb(here: str) -> Static:
    """Where you are in home > cook > inspect > plated, shown on every screen."""
    t = Text("  ")
    for i, step in enumerate(FLOW):
        t.append(step, style=f"bold {WHITE}" if step == here else FAINT)
        if i < len(FLOW) - 1:
            t.append("  ›  ", style=EDGE)
    return Static(t, classes="crumb")


# ── home ─────────────────────────────────────────────────────────────────────────
INTRO = Text.assemble(
    ("An AI (", INK), ("Carmy", f"bold {WHITE}"), (") writes GPU kernels for Apple chips. A strict ", INK),
    ("judge", f"bold {WHITE}"), (" runs each one on your GPU, "
     "proves it is correct on hundreds of inputs it never saw, and measures real speed against Apple's MLX. ", INK),
    ("Only proven kernels survive. Each round builds on the best one, and verified lessons carry over.", DIM),
)


def loop_diagram() -> Text:
    """The self-improving loop, drawn with the return arrow on the left."""
    t = Text()
    steps = [("1 draft ", "Carmy writes 3 kernels in parallel"),
             ("2 judge ", "correct on 100s of inputs? how fast?"),
             ("3 keep  ", "fastest proven kernel = champion"),
             ("4 learn ", "scorecard + what failed + notes")]
    for i, (name, what) in enumerate(steps):
        rail = "┌─→ " if i == 0 else "└── " if i == len(steps) - 1 else "│   "
        t.append(rail, style=GREEN).append(name, style=f"bold {WHITE}").append(f" {what}\n", style=DIM)
    t.append("\n↺ ", style=f"bold {GREEN}").append("each round builds on the best so far\n", style=INK)
    t.append("  stops at the chip's speed limit, or when it\n  stops getting faster\n", style=DIM)
    t.append("  notes are kept, so the next cook starts smarter", style=DIM)
    return t


def about_text(op_id: str, runs: list) -> Text:
    """Plain-English note on the highlighted kernel."""
    if op_id.startswith("wip:"):
        w = next(x for x in ops.WIP if x["name"] == op_id[4:])
        return Text.assemble((w["name"], f"bold {WHITE}"), ("   work in progress\n\n", AMBER), (w["about"], INK),
                             ("\n\ncoming soon", DIM))
    spec = ops.get(op_id)
    mine = [r for r in runs if r.op == spec.name]
    best = max((r.summary.get("speedup") or 0 for r in mine), default=0)
    t = Text.assemble((spec.name, f"bold {WHITE}"), ("\n\n", ""), (spec.about, INK), ("\n\n", ""))
    if best:
        t.append_text(Text.assemble(("best so far  ", DIM), speed(best), (" vs MLX\n", DIM)))
    else:
        t.append("not cooked yet\n", style=DIM)
    t.append_text(Text.assemble(("enter", f"bold {WHITE}"), (" cook   ", DIM), ("t", f"bold {WHITE}"),
                                (" trust the judge   ", DIM), ("h", f"bold {WHITE}"), (" history", DIM)))
    return t


class HomeScreen(Screen):
    BINDINGS = [Binding("enter", "cook", "cook", priority=True), Binding("t", "trust", "trust the judge"),
                Binding("h", "history", "history"), Binding("q", "app.quit", "quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield crumb("home")
        yield Static(INTRO, id="intro", classes="pane")
        yield OptionList(id="ops")
        with Horizontal(id="below"):
            yield Static(loop_diagram(), id="loop", classes="pane")
            yield Static("", id="about", classes="pane")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#intro").border_title = "what is carmen"
        self.query_one("#ops").border_title = "pick a kernel to cook"
        self.query_one("#loop").border_title = "how it works"
        self.query_one("#about").border_title = "about this kernel"
        self.refresh_ops()

    def on_screen_resume(self) -> None:
        self.refresh_ops()

    def refresh_ops(self) -> None:
        self.runs = load_runs(self.app.runs_dir)
        ol = self.query_one("#ops", OptionList)
        ol.clear_options()
        for spec in ops.OPS.values():
            mine = [r for r in self.runs if r.op == spec.name]
            best = max((r.summary.get("speedup") or 0 for r in mine), default=0)
            status = (Text.assemble(("best ", DIM), speed(best),
                                    (f" · verified · {len(mine)} run{'s' if len(mine) != 1 else ''}", DIM))
                      if best else Text("not cooked yet", style=DIM))
            summary = spec.summary if len(spec.summary) <= 50 else spec.summary[:49] + "…"
            ol.add_option(Option(Text.assemble((f"  {spec.name:<18}", f"bold {WHITE}"), (f"{summary:<54}", DIM),
                                               status), id=spec.name))
        for w in ops.WIP:
            ol.add_option(Option(Text.assemble((f"  {w['name']:<18}", FAINT), (f"{w['summary']:<54}", FAINT),
                                               ("wip", AMBER)), id=f"wip:{w['name']}"))
        ol.highlighted = 0
        ol.focus()
        self.show_about(self.selected_op())

    def show_about(self, op_id: str) -> None:
        self.query_one("#about", Static).update(about_text(op_id, self.runs))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self.show_about(event.option.id)

    def selected_op(self) -> str:
        ol = self.query_one("#ops", OptionList)
        return ol.get_option_at_index(ol.highlighted or 0).id

    def _cookable(self, op_id: str) -> bool:
        if op_id.startswith("wip:"):
            self.notify(f"{op_id[4:]} is a work in progress: coming soon", severity="warning")
            return False
        return True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._cookable(event.option.id):
            self.app.push_screen(KitchenScreen(event.option.id))

    def action_cook(self) -> None:
        if self._cookable(self.selected_op()):
            self.app.push_screen(KitchenScreen(self.selected_op()))

    def action_trust(self) -> None:
        if self._cookable(self.selected_op()):
            self.app.push_screen(TrustScreen(self.selected_op()))

    def action_history(self) -> None:
        self.app.push_screen(HistoryScreen())


# ── kitchen ──────────────────────────────────────────────────────────────────────
STAGES = ("draft", "judge", "improve", "plated")


class Card(Static, can_focus=True):
    BINDINGS = [Binding("enter", "open", "open")]

    def __init__(self, idx: int):
        super().__init__(classes="card")
        self.idx = idx
        self.aid: str | None = None
        self.state = "idle"
        self.plan = ""
        self.result: dict | None = None

    def action_open(self) -> None:
        self.screen.open_card(self)


class KitchenScreen(Screen):
    BINDINGS = [Binding("s", "stop", "stop after this round"), Binding("escape", "leave", "back"),
                Binding("left", "app.focus_previous", show=False), Binding("right", "app.focus_next", show=False)]

    def __init__(self, op: str, start: Item | None = None, rounds: int = 3, k: int = 3):
        super().__init__()
        self.op, self.start, self.rounds, self.k = op, start, rounds, k
        self.stage, self.round = "draft", 0
        self.champion: dict | None = None
        self.history: list[float] = []
        self.run_dir: Path | None = None
        self.stop_requested = False
        self.finished = False
        self.tick = 0
        self.note = ""

    def compose(self) -> ComposeResult:
        yield crumb("cook")
        yield Static(id="rail")
        with Horizontal(id="cards"):
            for i in range(self.k):
                yield Card(i)
        yield RichLog(id="judgelog", classes="pane", wrap=True, markup=False)
        yield Static(id="strip", classes="pane")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#strip").border_title = "champion"
        self.query_one("#judgelog").border_title = "what the judge said"
        self.render_all()
        self.set_interval(0.12, self.spin)
        self.cook()

    # rendering
    def spin(self) -> None:
        self.tick += 1
        for c in self.query(Card):
            if c.state in ("writing", "judging"):
                self.render_card(c)

    def render_all(self) -> None:
        self.render_rail()
        for c in self.query(Card):
            self.render_card(c)
        self.render_strip()

    def render_rail(self) -> None:
        t = Text.assemble((self.op, f"bold {WHITE}"), "     ")
        cur = STAGES.index(self.stage)
        for i, s in enumerate(STAGES):
            t.append(f"{'●' if i <= cur else '○'} {s}", style=f"bold {WHITE}" if i == cur else INK if i < cur else FAINT)
            if i < len(STAGES) - 1:
                t.append("  ───  ", style=EDGE)
        if not self.finished:
            t.append(f"     round {self.round + 1} of {self.rounds}", style=DIM)
        self.query_one("#rail", Static).update(t)

    def render_card(self, c: Card) -> None:
        sp = SPIN[self.tick % len(SPIN)]
        head = Text(c.aid or f"draft {c.idx + 1}", style=f"bold {WHITE}")
        body = Text(c.plan[:240], style=INK)
        if c.state == "idle":
            status = Text("waiting", style=FAINT)
        elif c.state == "writing":
            status = Text(f"{sp} Carmy is writing", style=DIM)
        elif c.state == "queued":
            status = Text("· waiting for the judge", style=DIM)
        elif c.state == "judging":
            status = Text(f"{sp} judging: compile · 40+ tests · hidden draw · timing", style=DIM)
        elif c.state == "error":
            status = Text("✗ the model call failed", style=RED)
        else:
            v = c.result or {}
            if v.get("correct"):
                status = Text.assemble(("✓ verified  ", f"bold {GREEN}"), speed(v.get("speedup")), (" vs mlx", DIM))
                h = v.get("hidden")
                if h:
                    status.append(f"\n  hidden {h['total'] - h['failed']}/{h['total']}",
                                  style=GREEN if not h["failed"] else RED)
            else:
                status = Text(f"✗ {v.get('why', 'rejected')}", style=f"bold {RED}")
                status.append("\n  Carmy gets the located error next round", style=DIM)
        c.set_class(bool(self.champion and self.champion.get("attempt") == c.aid), "win")
        c.update(Group(head, Text(""), body, Text(""), status))

    def render_strip(self) -> None:
        if not self.champion:
            self.query_one("#strip", Static).update(Text(self.note or "no verified kernel yet", style=DIM))
            return
        ch = self.champion
        t = Text.assemble(("★ ", GREEN), (ch["attempt"], f"bold {WHITE}"), "   ", speed(ch["speedup"]), (" vs mlx", DIM))
        if ch.get("pct") is not None:
            t.append(f"   {ch['pct']:.0%} of peak bandwidth", style=DIM)
        if self.history:
            t.append("\nbest each round  ", style=FAINT)
            for i, v in enumerate(self.history):
                if i:
                    t.append("  →  ", style=EDGE)
                t.append_text(speed(v, bold=False) if v else Text("none", style=DIM))
        if self.note:
            t.append(f"\n{self.note}", style=DIM)
        self.query_one("#strip", Static).update(t)

    # events from the loop (delivered on the UI thread)
    def on_loop_event(self, t: str, d: dict) -> None:
        cards = list(self.query(Card))
        if t == "run_started":
            self.run_dir = Path(d["run_dir"])
        elif t == "round_started":
            self.round = d["round"]
            self.query_one("#judgelog", RichLog).write(Text(f"round {d['round'] + 1}", style=FAINT))
            self.stage = "draft" if d["round"] == 0 and self.start is None else "improve"
            for c in cards:
                c.aid, c.state, c.plan, c.result = None, "writing", "", None
        elif t == "attempt_submitted":
            self.stage = "judge"
            for c in cards:
                if c.state == "writing":
                    c.state = "queued"
            c = cards[int(d["attempt"].split("-")[1])]
            c.aid, c.plan, c.state = d["attempt"], d.get("plan") or "", "judging"
        elif t == "carmy_error":
            cards[d["attempt"]].state = "error"
            self.query_one("#judgelog", RichLog).write(Text(f"draft {d['attempt'] + 1}  ✗ model call failed: "
                                                            f"{d['error'][:200]}", style=RED))
        elif t == "judge_result":
            c = cards[int(d["attempt"].split("-")[1])]
            c.state, c.result = "done", dict(d)
            item = load_attempt(self.run_dir, d["attempt"], self.op) if self.run_dir else None
            if not d["correct"]:
                c.result["why"] = plain_failure(item.verdict) if item and item.verdict else d["stage"]
            self.log_result(d, item)
        elif t == "champion":
            self.champion = {"attempt": d["attempt"], "speedup": d["speedup"], "pct": d.get("pct_peak_min")}
            self.query_one("#judgelog", RichLog).write(Text(f"★ {d['attempt']} is the new champion", style=GREEN))
        elif t == "round_finished":
            best = max((c.result.get("speedup") or 0 for c in cards if c.result and c.result.get("correct")),
                       default=0.0)
            self.history.append(best)
        elif t == "stopped":
            self.note = d["reason"]
        self.render_all()

    def log_result(self, d: dict, item: Item | None) -> None:
        log = self.query_one("#judgelog", RichLog)
        head = Text(f"{d['attempt']}  ", style=f"bold {WHITE}")
        if d["correct"]:
            h = d.get("hidden") or {}
            head.append("✓ verified  ", style=f"bold {GREEN}").append_text(speed(d.get("speedup")))
            head.append(" vs mlx", style=DIM)
            if d.get("speedup_compiled"):
                head.append(" · ", style=DIM).append_text(speed(d["speedup_compiled"], bold=False))
                head.append(" vs mx.compile", style=DIM)
            if d.get("pct_peak_min") is not None:
                head.append(f" · {d['pct_peak_min']:.0%} of peak", style=DIM)
            if h:
                head.append(f" · hidden {h['total'] - h['failed']}/{h['total']}", style=DIM)
            log.write(head)
            return
        head.append(f"✗ {plain_failure(item.verdict) if item and item.verdict else d['stage']}", style=f"bold {RED}")
        log.write(head)
        if item and item.feedback:
            for line in item.feedback.strip().splitlines()[:3]:
                log.write(Text("    " + line[:220], style=DIM))

    def on_loop_done(self, summary: dict | None, error: str | None) -> None:
        self.finished, self.stage = True, "plated"
        self.app.busy = False
        self.render_all()
        if error:
            self.note = error
            self.render_strip()
            self.notify(error, severity="error", timeout=12)
            return
        self.app.push_screen(PlatedScreen(self.op, summary, self.run_dir))

    @work(thread=True)
    def cook(self) -> None:
        app: Carmen = self.app
        if app.busy:
            app.call_from_thread(self.on_loop_done, None, "already cooking something; one GPU job at a time")
            return
        app.busy = True
        try:
            start = None
            if self.start is not None:
                item = self.start
                if item.verdict is None:
                    peak = judge.peak_gbps(app.backend)["peak_gbps"]
                    item.verdict = judge.judge(item.op, item.kernel, backend=app.backend, peak=peak,
                                               round_seed=0, hidden_seed=secrets.randbits(31))
                    item.feedback = judge.feedback(item.verdict)
                start = {"id": f"{item.run or 'golden'}:{item.attempt or item.op}", "kernel": item.kernel.to_json(),
                         "verdict": item.verdict, "feedback": item.feedback}
            sm = loop.run(self.op, rounds=self.rounds, k=self.k, model=app.model, backend=app.backend,
                          runs_dir=app.runs_dir, memory_dir=app.memory_dir, peak=judge.peak_gbps(app.backend),
                          on_event=lambda t, d: app.call_from_thread(self.on_loop_event, t, d), start=start,
                          should_stop=lambda: self.stop_requested)
            app.call_from_thread(self.on_loop_done, sm, None)
        except carmy.CarmyAuthError as e:
            app.call_from_thread(self.on_loop_done, None, f"{e}. Check ANTHROPIC_API_KEY in .env.")
        except Exception as e:
            app.call_from_thread(self.on_loop_done, None, f"{type(e).__name__}: {e}")

    def open_card(self, c: Card) -> None:
        if not (c.aid and self.run_dir):
            return
        item = load_attempt(self.run_dir, c.aid, self.op)
        champ = load_attempt(self.run_dir, self.champion["attempt"], self.op) if self.champion else None
        if item:
            self.app.push_screen(InspectScreen(item, champ))

    def action_stop(self) -> None:
        self.stop_requested = True
        self.note = "stopping after this round…"
        self.render_strip()

    def action_leave(self) -> None:
        if not self.finished:
            self.notify("still cooking: press s to stop after this round", severity="warning")
            return
        self.app.pop_screen()


# ── inspect ──────────────────────────────────────────────────────────────────────
class InspectScreen(Screen):
    BINDINGS = [Binding("i", "improve", "improve from here"), Binding("j", "judge", "judge again"),
                Binding("d", "diff", "diff vs champion"), Binding("e", "export", "export .metal"),
                Binding("escape", "app.pop_screen", "back")]

    def __init__(self, item: Item, champion: Item | None = None):
        super().__init__()
        self.item, self.champion, self.showing_diff = item, champion, False

    def compose(self) -> ComposeResult:
        yield crumb("inspect")
        with Horizontal():
            with VerticalScroll(id="code", classes="pane"):
                yield Static(id="src")
            with VerticalScroll(id="verdict", classes="pane"):
                yield Static(id="info")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#verdict").border_title = "verdict"
        self.show()

    def show(self) -> None:
        code = self.query_one("#code")
        if self.showing_diff and self.champion:
            diff = "\n".join(difflib.unified_diff(self.champion.kernel.source.splitlines(),
                                                  self.item.kernel.source.splitlines(),
                                                  f"champion {self.champion.attempt}", f"this {self.item.attempt}",
                                                  lineterm=""))
            self.query_one("#src", Static).update(Syntax(diff or "no differences", "diff", theme="github-dark",
                                                         background_color=PANEL))
            code.border_title = "diff vs champion"
        else:
            self.query_one("#src", Static).update(code_view(self.item))
            code.border_title = "kernel"
        code.border_subtitle = f"{self.item.op} · {self.item.title}"
        self.query_one("#info", Static).update(verdict_panel(self.item))

    def action_diff(self) -> None:
        if not self.champion or self.champion.attempt == self.item.attempt:
            self.notify("this is the champion; nothing to compare against")
            return
        self.showing_diff = not self.showing_diff
        self.show()

    def action_improve(self) -> None:
        self.app.push_screen(KitchenScreen(self.item.op, start=self.item, rounds=2))

    def action_export(self) -> None:
        out = Path("exports")
        out.mkdir(exist_ok=True)
        name = f"{self.item.op}-{self.item.run or 'golden'}-{self.item.attempt or 'kernel'}.metal"
        v = self.item.verdict or {}
        head = (f"// carmen export · {self.item.op} · {self.item.title}\n"
                f"// verified: {bool(v.get('correct'))} · speed vs mlx: {v.get('speedup_geomean')}\n"
                f"// configs: {self.item.kernel.configs}\n\n")
        (out / name).write_text(head + (self.item.kernel.header + "\n" if self.item.kernel.header else "")
                                + self.item.kernel.source)
        self.notify(f"saved exports/{name}")

    def action_judge(self) -> None:
        if self.app.busy:
            self.notify("the GPU is busy with another job", severity="warning")
            return
        self.app.busy = True
        self.notify("judging… (about 20–60 s)")
        self._judge()

    @work(thread=True)
    def _judge(self) -> None:
        app: Carmen = self.app
        try:
            peak = judge.peak_gbps(app.backend)["peak_gbps"]
            v = judge.judge(self.item.op, self.item.kernel, backend=app.backend, peak=peak, round_seed=0,
                            hidden_seed=secrets.randbits(31))
            self.item.verdict, self.item.feedback = v, judge.feedback(v)
            app.call_from_thread(self.show)
        except Exception as e:
            app.call_from_thread(self.notify, f"{type(e).__name__}: {e}", severity="error")
        finally:
            app.busy = False


# ── plated ───────────────────────────────────────────────────────────────────────
class PlatedScreen(Screen):
    BINDINGS = [Binding("enter", "open", "open the winner"), Binding("c", "again", "cook again"),
                Binding("escape", "home", "home")]

    def __init__(self, op: str, summary: dict, run_dir: Path | None):
        super().__init__()
        self.op, self.sm, self.run_dir = op, summary, run_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield crumb("plated")
        yield Static(id="plated", classes="pane")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#plated").border_title = "plated"
        sm = self.sm
        t = Table.grid(padding=(0, 4))
        t.add_column(style=DIM, justify="right")
        t.add_column()
        if sm.get("champion"):
            pct = sm.get("pct_peak_min")
            t.add_row("best kernel", Text(f"{self.op} · {sm['champion']}", style=f"bold {WHITE}"))
            t.add_row("speed vs mlx", speed(sm.get("speedup")))
            if sm.get("speedup_compiled"):
                t.add_row("vs mx.compile(mlx)", speed(sm.get("speedup_compiled")))
            t.add_row("on sizes it never saw", speed(sm.get("hidden_speedup")))
            t.add_row("worst % of peak", Text("—" if pct is None else f"{pct:.0%}", style=INK))
            if sm.get("hidden_total"):
                t.add_row("hidden tests", Text(f"{sm['hidden_total'] - sm['hidden_failed']}/{sm['hidden_total']} passed",
                                               style=GREEN if not sm["hidden_failed"] else RED))
        else:
            t.add_row("result", Text("no kernel passed the judge", style=f"bold {RED}"))
        t.add_row("", "")
        t.add_row("attempts", Text(f"{sm['verified']} of {sm['attempts']} verified · first try "
                                   f"{sm['first_round_verified']}", style=INK))
        t.add_row("naive check", Text(f"would have passed {sm['naive_pass_but_wrong']} wrong kernel(s)", style=INK))
        t.add_row("saved to", Text(str(self.run_dir or sm.get("run_dir", "")), style=DIM))
        ok = bool(sm.get("champion"))
        head = Text("✓ plated" if ok else "✗ nothing plated", style=f"bold {GREEN if ok else RED}")
        self.query_one("#plated", Static).update(Group(head, Text(""), t))

    def action_open(self) -> None:
        if self.sm.get("champion") and self.run_dir:
            item = load_attempt(self.run_dir, self.sm["champion"], self.op)
            if item:
                self.app.push_screen(InspectScreen(item))

    def action_again(self) -> None:
        self.app.switch_screen(KitchenScreen(self.op))

    def action_home(self) -> None:
        while len(self.app.screen_stack) > 2:
            self.app.pop_screen()


# ── trust ────────────────────────────────────────────────────────────────────────
class TrustScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "back")]

    def __init__(self, op: str):
        super().__init__()
        self.op, self.rows = op, []
        self.golden: bool | None = None
        self.done = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="trust", classes="pane")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#trust").border_title = f"trust the judge · {self.op}"
        self.render_table()
        if self.app.busy:
            self.notify("the GPU is busy with another job", severity="warning")
            return
        self.app.busy = True
        self.run_mutants()

    def render_table(self) -> None:
        intro = Text("The judge must reject every deliberately broken kernel below and accept the golden one.\n"
                     "For comparison: would a KernelBench-style check (one shape, allclose 1e-4) have noticed?\n",
                     style=DIM)
        g = Text("golden kernel  ", style=INK).append(
            "checking…" if self.golden is None else ("✓ accepted" if self.golden else "✗ rejected: fix the tests"),
            style=DIM if self.golden is None else GREEN if self.golden else RED)
        t = Table(box=box.SIMPLE_HEAD, expand=True, header_style=DIM, border_style=EDGE)
        for col in ("broken kernel", "bug type", "carmen", "naive check", "where carmen found it"):
            t.add_column(col, no_wrap=True)
        for r in self.rows:
            t.add_row(*r)
        kills = sum(1 for r in self.rows if "caught" in r[2].plain)
        fooled = sum(1 for r in self.rows if "missed" in r[3].plain)
        foot = Text(f"\ncarmen caught {kills}/{len(self.rows)} · the naive check missed {fooled}/{len(self.rows)}"
                    if self.done else "", style=f"bold {INK}")
        self.query_one("#trust", Static).update(Group(intro, g, Text(""), t, foot))

    @work(thread=True)
    def run_mutants(self) -> None:
        app: Carmen = self.app
        try:
            g = judge.judge(self.op, broken.golden(self.op), backend=app.backend, skip_timing=True)
            self.golden = bool(g["correct"])
            app.call_from_thread(self.render_table)
            if not self.golden:
                return
            for m, kernel in broken.mutants(self.op):
                v = judge.judge(self.op, kernel, backend=app.backend, skip_timing=True)
                caught = not v["correct"]
                fails = v.get("configs", [{}])[0].get("failures") if v.get("configs") else None
                where = fails[0].get("where", "") if fails else ((v.get("error") or "").splitlines() or [""])[0]
                self.rows.append([Text(m.name, style=INK), Text(m.family, style=DIM),
                                  Text("✓ caught", style=GREEN) if caught else Text("✗ missed", style=RED),
                                  Text("missed", style=AMBER) if v.get("naive_pass_strict") and caught
                                  else Text("caught", style=DIM), Text(where[:48], style=DIM)])
                app.call_from_thread(self.render_table)
            self.done = True
            app.call_from_thread(self.render_table)
        except Exception as e:
            app.call_from_thread(self.notify, f"{type(e).__name__}: {e}", severity="error")
        finally:
            app.busy = False


# ── history ──────────────────────────────────────────────────────────────────────
class HistoryScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield OptionList(id="ops")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ops").border_title = "past runs"
        self.runs = load_runs(self.app.runs_dir)
        ol = self.query_one("#ops", OptionList)
        if not self.runs:
            ol.add_option(Option(Text("nothing cooked yet", style=DIM), disabled=True))
        for i, r in enumerate(self.runs):
            n = r.dir.name
            when = f"{n[4:6]}/{n[6:8]} {n[9:11]}:{n[11:13]}"
            ok = sum(1 for a in r.attempts if a.verdict and a.verdict.get("correct"))
            ol.add_option(Option(Text.assemble((f"{when}   ", DIM), (f"{r.op:<16}", f"bold {WHITE}"),
                                               (f"{'best-of-N' if r.mode == 'bon' else 'loop':<11}", DIM),
                                               speed(r.summary.get("speedup")),
                                               (f"   {ok}/{len(r.attempts)} verified", DIM)), id=str(i)))
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        r = self.runs[int(event.option.id)]
        champ = next((a for a in r.attempts if a.attempt == r.summary.get("champion")), None)
        target = champ or (r.attempts[0] if r.attempts else None)
        if target:
            self.app.push_screen(InspectScreen(target, champ))


# ── app ──────────────────────────────────────────────────────────────────────────
class Carmen(App):
    TITLE = "carmen"
    CSS = CSS

    def __init__(self, backend: str = "metal", runs_dir: Path = Path("runs"), memory_dir: Path = Path("memory"),
                 model: str = carmy.DEFAULT_MODEL):
        super().__init__()
        self.backend, self.runs_dir, self.memory_dir, self.model = backend, Path(runs_dir), Path(memory_dir), model
        self.busy = False

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())


def main(backend: str = "metal", runs_dir: str = "runs", memory_dir: str = "memory",
         model: str = carmy.DEFAULT_MODEL) -> None:
    Carmen(backend, Path(runs_dir), Path(memory_dir), model).run()
