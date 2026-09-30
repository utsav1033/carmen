"""The playbook: lessons Carmy carries between rounds and runs.

Rules (adapted from SAGE, arXiv 2609.35568, and the Prime Agent incident where an
unverified self-refinement saved a reward hack as a skill):

- A lesson gets credit only from the judge's verdict, never from Carmy saying it worked.
- Credit goes to lessons Carmy *says it used*; lessons shown but not used lose a little.
- Two kinds: `chip` lessons (should carry across ops) and `op` lessons (carry across chips).
- A lesson becomes `resident` (always shown) only after repeated verified use; a lesson
  that keeps hurting is forgotten. When in doubt, forget.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUCCESS, FAILURE, UNUSED = 1.0, -0.2, -0.2
ETA_MIN = 0.05
MAX_SHOWN = 20
MAX_NEW_PER_ROUND = 3


@dataclass
class Lesson:
    id: str
    text: str
    kind: str  # "chip" or "op"
    op: str
    evidence: list[str] = field(default_factory=list)
    u: float = 0.0
    n_ret: int = 0
    n_ado: int = 0
    adopted_ops: list[str] = field(default_factory=list)
    status: str = "candidate"  # candidate | resident


def lint(text: str) -> str | None:
    """Reject lessons that smuggle in test shapes or are too vague to act on."""
    if len(text) < 25:
        return "too short to be actionable"
    if len(text) > 400:
        return "too long; one mechanism per lesson"
    if re.search(r"\b\d{3,}\b", text) and not re.search(r"\b(32|64|128|256|512|1024)\b", text):
        return "mentions a specific size; lessons must be conditions, not memorized shapes"
    return None


class Playbook:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lessons: dict[str, Lesson] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text()):
                self.lessons[d["id"]] = Lesson(**d)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(l) for l in self.lessons.values()], indent=2))

    def shown(self, op: str) -> list[Lesson]:
        pool = [l for l in self.lessons.values() if l.kind == "chip" or l.op == op]
        pool.sort(key=lambda l: (l.status != "resident", -l.u))
        return pool[:MAX_SHOWN]

    def render(self, op: str) -> str:
        items = self.shown(op)
        if not items:
            return ""
        return "\n".join(f"[{l.id}] ({l.kind}) {l.text}" for l in items)

    def credit(self, shown_ids: list[str], used_ids: list[str], verified: bool, op: str) -> None:
        used = [i for i in used_ids if i in shown_ids and i in self.lessons]
        z = SUCCESS if verified else FAILURE
        for lid in shown_ids:
            l = self.lessons.get(lid)
            if l is None:
                continue
            l.n_ret += 1
            xi = z / len(used) if lid in used else UNUSED
            if lid in used:
                l.n_ado += 1
                if verified and op not in l.adopted_ops:
                    l.adopted_ops.append(op)
            eta = max(1.0 / (1 + l.n_ret), ETA_MIN)
            l.u = (1 - eta) * l.u + eta * xi
        self._update_status()

    def _update_status(self) -> None:
        for lid, l in list(self.lessons.items()):
            needed_ops = 2 if l.kind == "chip" else 1
            if l.u > 0 and l.n_ado >= 2 and len(l.adopted_ops) >= needed_ops:
                l.status = "resident"
            elif l.status == "resident" and l.u <= 0:
                l.status = "candidate"
            if l.u < -0.15 and l.n_ret >= 6:
                del self.lessons[lid]  # when in doubt, forget

    def add(self, proposals: list[dict], op: str, verified_attempts: set[str]) -> list[tuple[dict, str]]:
        """Add reflector proposals. Returns (proposal, reason) for each one rejected."""
        rejected = []
        existing = {l.text.lower() for l in self.lessons.values()}
        added = 0
        for p in proposals:
            evidence = [e for e in p.get("evidence", []) if e in verified_attempts]
            reason = lint(p.get("text", ""))
            if reason is None and not evidence:
                reason = "no evidence: must cite an attempt the judge verified"
            if reason is None and p["text"].lower() in existing:
                reason = "duplicate"
            if reason is None and p.get("kind") not in ("chip", "op"):
                reason = "kind must be chip or op"
            if reason is None and added >= MAX_NEW_PER_ROUND:
                reason = "too many new lessons this round"
            if reason:
                rejected.append((p, reason))
                continue
            lid = f"L{len(self.lessons) + 1:03d}"
            while lid in self.lessons:
                lid = f"L{int(lid[1:]) + 1:03d}"
            self.lessons[lid] = Lesson(lid, p["text"].strip(), p["kind"], op, evidence)
            existing.add(p["text"].lower())
            added += 1
        return rejected
