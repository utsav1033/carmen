# The judge

How carmen decides whether a kernel is right and how fast it really is. Back to the [README](../README.md).

## Measured: how many bugs it catches

75 realistic bugs planted across seven kernels, including two that read 4-bit quantized weights the way real models store them, run on an M4. A KernelBench-style check (one shape, `allclose`) passed **35 of them at 1e-2, and 19 even at the strict 1e-4**. carmen's judge caught **all 75**, and said where each one was. Two matmul bugs (wrong leading dimension, swapped tile axes) are exactly right on square matrices, so a one-shape square check can never see them.

| kernel | seeded bugs | carmen caught | one-shape check passed (1e-2) | one-shape check passed (1e-4) |
|---|---|---|---|---|
| softmax | 12 | **12** | 10 | 7 |
| add_rmsnorm | 11 | **11** | 5 | 1 |
| matmul | 8 | **8** | 4 | 2 |
| attention | 10 | **10** | 3 | 2 |
| residual_rmsnorm | 13 | **13** | 6 | 2 |
| mlp_up (4-bit) | 13 | **13** | 5 | 4 |
| mlp_down (4-bit) | 8 | **8** | 2 | 1 |

## Where AI-written kernels fail, and how carmen catches each one

Every row is a failure mode documented in the papers below or seen in practice. The judge is built against this list.

| failure mode | seen in | carmen | how |
|---|---|---|---|
| tail and boundary bugs (odd lengths, the last element) | [Measuring the Checker](https://arxiv.org/abs/2609.22220) | ✅ | lengths 1, 2, 3, 7, 31, 33, 255, 257, 1025, 4097, 70001; a failure is shrunk to the smallest input that still fails |
| precision: fp16 accumulation, overflow in `exp` or `x²` | [Measuring the Checker](https://arxiv.org/abs/2609.22220) | ✅ | float64 answer key, values up to ±10⁴, tolerance relative to each row's scale |
| race conditions (missing barriers) | [Measuring the Checker](https://arxiv.org/abs/2609.22220) | ⚠️ partly | the 4 largest inputs of different kinds and dtypes run 5× and must match byte for byte; a race that never fires on this GPU can still slip through |
| indexing: wrong strides, overlapping rows | [Measuring the Checker](https://arxiv.org/abs/2609.22220) | ✅ | many rows, full-output comparison, located by row and column |
| semantics: NaN, −inf, fully masked rows, eps | [Test-Input Generation for Tensor Programs](https://arxiv.org/abs/2606.27396) | ✅ | trap inputs per kernel: all −inf rows, zero-variance rows, eps-dominated rows, cancelling residuals |
| fast only on the shapes or config it was tested on | [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722), [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) | ✅ | every config is checked; each round draws secret sizes and data the model never sees, for correctness and speed |
| an output that is never written still passes | [KernelBench #165](https://github.com/ScalingIntelligence/KernelBench) | ✅ | the output is pre-filled with NaN; errors are relative to each row, so all-zeros fails |
| writing past the end of the output | seen in practice | ✅ | a guard zone after the output must stay untouched |
| skipping work or gaming the timer | [Reward Hacking in Self-Improving Code Agents](https://openreview.net/forum?id=ikrQWGgxYg), [Auditing Harness Tampering](https://arxiv.org/abs/2609.00069) | ✅ | the model only returns kernel text; it never runs code in the judge; speeds beyond measured memory bandwidth are flagged |
| a launch grid smaller than the work, silently truncated | [MLX #4534](https://github.com/ml-explore/mlx/issues/4534) | ✅ by design | the harness sets the launch shape, never the model |
| bf16, the dtype most models ship in | seen in practice | ✅ | bf16 cases in the visible battery, the fresh fuzz and the hidden draw |
| empty tensors | seen in practice | ✅ by design | the harness never launches on an empty tensor |
| a slow baseline making a kernel look fast | [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) | ✅ | every speedup is reported against plain MLX **and** `mx.compile` |
| attention: causal boundary off by one, a missing KV-cache offset, the softmax rescale | seen in practice | ✅ new | queries ≠ keys, head sizes 1 to 160, scores in the hundreds, values that ramp with the key index; 10 seeded bugs |
| 32-bit index overflow on tensors over 2³¹ elements | seen in practice | ❌ | needs 8+ GB buffers per test; out of reach on most Macs |
| non-contiguous or strided inputs | seen in practice | ❌ | inputs are always contiguous rows today |
| matmul-style bugs: 2D tile edges, the K tail, indexing that's only right on square matrices | [Measuring the Checker](https://arxiv.org/abs/2609.22220) | ✅ new | non-square shapes on every tile edge, K from 1 to 4097, 8 seeded matmul bugs, all killed on an M4 |

## How the judge decides

Every candidate runs in its own process with a timeout, because a bad kernel can hang the GPU. Checks fail fast, in order:

| gate | catches |
|---|---|
| **static** | oversized source, `#include`/`asm`, illegal launch configs |
| **compile** | the Metal compiler's own errors, sent back verbatim |
| **no-tolerance checks** | outputs pre-filled with NaN, so an unwritten element can't hide; NaN/±inf positions must match the reference; 5 identical runs must be byte-identical (races); a sentinel pad catches writes past the end; rows must sum to 1 |
| **visible battery** | tail lengths (1, 2, 3, 7, 31, 33, 255, 257, 1025, 4097, 70001), ±1e4 values (overflow without max-subtraction), very negative rows, constant rows, −inf masks, fully masked rows, spikes in the last element, fp32 and fp16 |
| **fresh fuzz** | new random off-grid shapes every round; failures join a permanent regression corpus |
| **hidden draw** | secret-seeded sizes off the power-of-two grid and different data statistics, for **correctness and speed**. It's reported, never shown to Carmy, and never used to pick the winner |
| **timing** | only after everything passes: interleaved A/B against the stock MLX op, median, bootstrap 95% CI, % of measured peak bandwidth |

**Tolerance** is fixed in advance, not tuned per kernel: output rounding plus float32 accumulation error (`2·eps + 8·eps₃₂·√n`), measured **relative to each row's scale**, so all-zeros can't sneak through. The stock library's own error can loosen it, but only up to a hard 16× cap. Any test input the stock library itself can't pass is dropped as unfair, and the count is logged.

**Feedback is built for the model.** Not "failed", but where and how. Here's real judge output for a kernel whose outputs are rounded through fp16. A KernelBench-style check passes it:

```
[TG=256] FAILED 20 test(s). First: neg_inf_mask[45x4938, float32] -> values outside tolerance,
in 0.7% of elements, scattered across 44 of 45 rows. Max row-relative error 0.000390824,
allowed 6.7254e-05. At row 0, column 11: got 0.00128269, expected 0.00128309.
Smallest failing input of this kind: rows=1, n=2.
```

And for one that forgets the last element of each row:

```
[TG=256] FAILED 42 test(s). First: neg_inf_mask[45x4938, float32] -> NaN where a number was
expected (unwritten output, or overflow), in only the last elements of rows (columns >= 4937
of 4938): tail handling. At row 0, column 4937: got nan, expected 0.
```

**The judge is measured too.** `carmen broken` runs 12 seeded bugs (10 for the norms) across five fault families (boundary, precision, sync, indexing, semantic; see [Du et al.](https://arxiv.org/abs/2609.22220)), and reports how many carmen kills and how many a KernelBench-style check would have let through.

## What carmen reports

For every run (`carmen report`), from judge verdicts alone:

- **first-round verified**: how often Carmy is right on the first try (tests the premise that models are good)
- **naive check fooled**: kernels a KernelBench-style check accepts that carmen proves wrong
- **hidden tests clean**: verified kernels that also pass inputs they never saw
- **speedup vs MLX**, with a CI, on visible *and* hidden sizes, plus worst-case % of peak bandwidth
- **loop vs best-of-N**: run the same op with `--mode bon` at the same call budget to see whether feedback and memory earn their keep

## Philosophy

1. **The generator is powerful and untrusted. The judge is simple and trusted.** Intelligence goes into generating, never into grading.
2. **The judge encodes what the op *means*,** not a list of tests. That part doesn't depend on the chip. Only compile/run/time does.
3. **Test where bugs live, somewhere the model can't predict:** edges, extremes, special values, and data drawn fresh every round.
4. **Self-correction is an external verdict plus a located error.** Models fix bugs well once told *where* they are.
5. **Measure the measurer.** A judge without a measured blind-spot rate is just a claim.
6. **Remember only verified causes.** When in doubt, forget.
7. **Start as a loop and a judge.** Every extra part has to earn its place against a failure seen in a real run.
8. **Report mechanisms honestly,** including when the harness fails. A verified 1.0× is a result. An unverified 3× isn't.

## Adding an op or a chip

An **op** is one file in `carmen/ops/`: a float64 `reference`, `invariants`, input `generators`, a visible battery, and the kinds used for fuzz and hidden draws. The judge, the loop and Carmy don't change.

A **chip** is one file in `carmen/backends/`: `build`, `upload`, `run`, `launch`, `baseline`, `measure_peak_gbps`. `metal.py` is ~110 lines. CUDA/ROCm is a sibling file, not a new test suite.

```
carmen/
  tui.py        the terminal app (Textual): browse, judge, improve, cook
  ops/          what each op means: reference, invariants, generators (chip-independent)
  backends/     how to compile, run and time on a chip (metal.py is the only Metal-aware file)
  judge/        gates, tolerance, diagnosis, timing stats; worker.py is the GPU-side process
  carmy.py      the model: stateless calls, structured output, no tools
  loop.py       draft → verify → improve → stop at the roofline
  memory.py     the playbook: verdict-credited lessons, chip vs op
  broken.py     seeded bugs for measuring the judge
kernels/golden/ hand-checked reference kernels (the judge's positive control)
tests/          the judge's logic, tested on a numpy stand-in for the GPU
```

