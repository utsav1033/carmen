from .base import Case, OpSpec
from . import add_rmsnorm, layernorm, masked_softmax, rmsnorm, softmax

OPS: dict[str, OpSpec] = {s.name: s for s in (softmax.SPEC, masked_softmax.SPEC, layernorm.SPEC, rmsnorm.SPEC,
                                       add_rmsnorm.SPEC)}

# Shown in the app as work in progress: not cookable yet. Each needs a reference, test inputs
# and a baseline before the judge can grade it.
WIP = [
    {"name": "attention", "summary": "full fused attention, FlashAttention-style",
     "about": "The heart of every transformer: scores, mask, softmax and the weighted sum of values, "
              "all in one kernel that never writes the big score matrix to memory. Hundreds of lines "
              "and easy to get subtly wrong, which is exactly where a strict judge earns its keep."},
    {"name": "rope", "summary": "rotary position embeddings",
     "about": "How a model knows word order: each pair of numbers in a query or key is rotated by an "
              "angle that depends on its position. Cheap math, but the indexing (which pairs, which "
              "angle) is where bugs hide."},
    {"name": "top_k", "summary": "pick the k most likely next tokens",
     "about": "The last step before the model picks its next word: find the k highest scores in a row "
              "of ~100k. A selection problem, not a sum, so it needs a different kind of kernel and a "
              "judge that checks ties and ordering."},
]


def get(name: str) -> OpSpec:
    if name not in OPS:
        raise KeyError(f"unknown op {name!r}; available: {', '.join(OPS)}")
    return OPS[name]


__all__ = ["Case", "OpSpec", "OPS", "WIP", "get"]
