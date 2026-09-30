import pytest


@pytest.fixture(autouse=True)
def fast_timing(monkeypatch):
    """The numpy stand-in has no launch overhead to amortize; one launch per sample is enough."""
    from carmen.judge import worker
    monkeypatch.setattr(worker, "INNER", 1)
