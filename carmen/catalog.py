"""The models the app offers: how big they are, whether they fit this Mac, whether they're downloaded,
and what's been measured on them.

Everything here is plain Python (no MLX), so the home screen can draw instantly. Sizes are the
4-bit safetensors on Hugging Face (mlx-community), which is also what each decoded word reads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class Model:
    alias: str
    id: str
    label: str
    gb: float  # weights on disk, ~ what every decoded word reads
    note: str = ""


MODELS = (
    Model("qwen0.5b", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "qwen 2.5 0.5b", 0.28,
          "small enough that kernel launches matter: carmen's fused kernels help here"),
    Model("llama1b", "mlx-community/Llama-3.2-1B-Instruct-4bit", "llama 3.2 1b", 0.70),
    Model("qwen1.5b", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "qwen 2.5 1.5b", 0.87),
    Model("qwen3b", "mlx-community/Qwen2.5-3B-Instruct-4bit", "qwen 2.5 3b", 1.74,
          "runs at the memory limit: speculative decoding is the lever"),
    Model("llama3b", "mlx-community/Llama-3.2-3B-Instruct-4bit", "llama 3.2 3b", 1.81,
          "big KV cache per token: long prompts are where a faster cache pays"),
    Model("qwen7b", "mlx-community/Qwen2.5-7B-Instruct-4bit", "qwen 2.5 7b", 4.28),
)

SYSTEM_GB = 4.0  # left for macOS and whatever else is open
SPEEDUP_COPIES = 3  # carmen speedup loads stock, compile and carmen side by side


def find(alias_or_id: str) -> Model:
    """A catalog model by alias or id; any other id becomes an uncatalogued model of unknown size."""
    for m in MODELS:
        if alias_or_id in (m.alias, m.id):
            return m
    return Model(alias_or_id, alias_or_id, alias_or_id.split("/")[-1].lower(), 0.0)


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def fits(gb: float, copies: int = 1, ram: float | None = None) -> str:
    """'yes', 'tight' or 'no' for `copies` copies of a `gb` model plus room for the system;
    'unknown' when the size or the RAM can't be read."""
    ram = ram_gb() if ram is None else ram
    if not gb or not ram:
        return "unknown"
    need = gb * copies + SYSTEM_GB
    return "yes" if need <= 0.75 * ram else "tight" if need <= ram else "no"


def hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def downloaded(model_id: str, cache: Path | None = None) -> bool:
    snaps = (cache or hf_cache()) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snaps.is_dir() and any(snaps.glob("*/*.safetensors"))


def _ratio(r: dict, stock: dict) -> float | None:
    from . import e2e
    p = e2e.paired(SimpleNamespace(decode_turns=r.get("decode_turns")), SimpleNamespace(decode_turns=stock.get("decode_turns")))
    return p[0] if p else None


def measured(model_id: str, runs_dir: Path) -> dict:
    """What this Mac has measured on the model, from the newest saved runs:
    {"fusion": x, "spec": {context: x}, "kv8": {context: x}, "quality": {"kv8": {context: top-1}},
    "limit_share": share}. Quality is how often a lossy lever's next word matches stock's."""
    out: dict = {}
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return out
    for path in sorted(runs_dir.glob("e2e-*.json"), key=lambda q: q.stat().st_mtime):  # newer runs overwrite
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = d.get("model", "")
        levers = name.endswith("-levers")
        if (name[: -len("-levers")] if levers else name) != model_id:
            continue
        rows = d.get("results") or []
        if levers:
            for ctx in sorted({r.get("context") for r in rows}):
                group = [r for r in rows if r.get("context") == ctx]
                stock = next((r for r in group if r["mode"] == "stock"), None)
                for r in group:
                    if stock and r["mode"] in ("spec", "kv8", "kv4", "w3") and (x := _ratio(r, stock)):
                        out.setdefault(r["mode"], {})[ctx] = x
                        if r.get("top1") is not None:
                            out.setdefault("quality", {}).setdefault(r["mode"], {})[ctx] = r["top1"]
                peak = d.get("peak_gbps")
                if stock and peak and stock.get("weight_bytes") and stock.get("kv_bytes") is not None and ctx == min(
                        r.get("context") for r in rows):
                    out["limit_share"] = stock["decode_tps"] * (stock["weight_bytes"] + stock["kv_bytes"]) / (peak * 1e9)
        else:
            stock = next((r for r in rows if r["mode"] == "stock"), None)
            carmen = next((r for r in rows if r["mode"] == "mlp+compile"), None)
            if stock and carmen and (x := _ratio(carmen, stock)):
                out["fusion"] = x
            limit = d.get("decode_speed_limit_tps")
            if stock and limit:
                out["limit_share"] = stock["decode_tps"] / limit
    return out


def summary(model_id: str, runs_dir: Path) -> str:
    """One line for a model's row: the measurements that say what helps it."""
    m = measured(model_id, runs_dir)
    parts = []
    if "fusion" in m:
        parts.append(f"carmen kernels {m['fusion']:.2f}×")
    for lever in ("spec", "kv8"):
        if lever in m:
            best_ctx = max(m[lever])
            parts.append(f"{lever} {m[lever][best_ctx]:.2f}× @{best_ctx // 1024 or best_ctx}{'k' if best_ctx >= 1024 else ''}")
    return " · ".join(parts)
