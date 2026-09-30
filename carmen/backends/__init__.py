from .base import PAD, Adapter, Kernel, static_check


def get(name: str):
    if name == "metal":
        from .metal import MetalAdapter
        return MetalAdapter()
    raise KeyError(f"unknown backend {name!r}; available: metal")


__all__ = ["PAD", "Adapter", "Kernel", "static_check", "get"]
