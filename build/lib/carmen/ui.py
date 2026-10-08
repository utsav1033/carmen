"""Terminal styling with no dependencies. Respects NO_COLOR and non-TTY output."""

from __future__ import annotations

import os
import sys

_ON = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34",
          "magenta": "35", "cyan": "36", "grey": "90"}


def s(text, *styles) -> str:
    if not _ON or not styles:
        return str(text)
    return f"\033[{';'.join(_CODES[x] for x in styles)}m{text}\033[0m"


OK = s("✓", "green", "bold")
BAD = s("✗", "red", "bold")
WARN = s("!", "yellow", "bold")
DOT = s("·", "grey")


def banner(sub: str = "") -> None:
    print(s("carmen", "bold", "magenta") + s("  kernels you can trust", "grey") + (s(f"  ·  {sub}", "grey") if sub else ""))


def rule(title: str = "") -> None:
    line = "─" * max(4, 56 - len(title))
    print(s(f"── {title} {line}" if title else "─" * 60, "grey"))


def _visible_len(text: str) -> int:
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def table(headers: list[str], rows: list[list], align: str | None = None) -> None:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max([_visible_len(h)] + [_visible_len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    align = align or "l" * len(headers)

    def fmt(row, style=()):
        out = []
        for i, c in enumerate(row):
            pad = " " * (widths[i] - _visible_len(c))
            out.append((pad + c) if align[i] == "r" else (c + pad))
        return s("  ".join(out), *style) if style else "  ".join(out)

    print(fmt(headers, ("grey",)))
    for r in cells:
        print(fmt(r))


def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def speed(x) -> str:
    if x is None:
        return "—"
    txt = f"{x:.2f}×"
    return s(txt, "green", "bold") if x >= 1.05 else s(txt, "yellow") if x >= 0.95 else s(txt, "red")
