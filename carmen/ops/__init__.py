from .base import Case, OpSpec
from . import masked_softmax, softmax

OPS: dict[str, OpSpec] = {s.name: s for s in (softmax.SPEC, masked_softmax.SPEC)}


def get(name: str) -> OpSpec:
    if name not in OPS:
        raise KeyError(f"unknown op {name!r}; available: {', '.join(OPS)}")
    return OPS[name]


__all__ = ["Case", "OpSpec", "OPS", "get"]
