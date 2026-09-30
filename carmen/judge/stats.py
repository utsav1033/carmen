"""Timing statistics: interleaved pairs, medians, bootstrap intervals."""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np


def time_pair(cand: Callable[[], None], base: Callable[[], None], warmup: int = 5, reps: int = 25):
    """Alternate candidate and baseline (ABBA order) so thermal drift hits both equally."""
    for _ in range(warmup):
        cand()
        base()
    tc, tb = [], []
    for i in range(reps):
        order = (cand, base) if i % 2 == 0 else (base, cand)
        for fn in order:
            t = time.perf_counter()
            fn()
            (tc if fn is cand else tb).append(time.perf_counter() - t)
    return np.array(tc), np.array(tb)


def speedup(tc: np.ndarray, tb: np.ndarray, seed: int = 0, n_boot: int = 2000):
    """Median of paired log-ratios, with a 95% bootstrap interval. >1 means the kernel is faster."""
    r = np.log(tb) - np.log(tc)
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(r, (n_boot, len(r))), axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(math.exp(np.median(r))), float(math.exp(lo)), float(math.exp(hi))


def geomean(xs) -> float:
    xs = [x for x in xs if x > 0]
    return float(math.exp(sum(map(math.log, xs)) / len(xs))) if xs else 0.0
