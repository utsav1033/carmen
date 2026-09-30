<p align="center">
  <img src="assets/banner.svg" alt="carmen: let the model cook. trust nothing it can't prove." width="100%">
</p>

<p align="center">
  <b>A self-correcting harness that gets LLMs to write fast GPU kernels, and proves they're right.</b><br>
  <sub>Apple Metal today. Any chip tomorrow: one adapter file.</sub>
</p>

---

LLMs write GPU kernels that look fast. A lot of them are quietly wrong.

- A KernelBench-style check (one shape, `allclose` at 1e-2) **misses 16.9% of injected kernel bugs, and 78.6% of precision bugs** ([Measuring the Checker](https://arxiv.org/abs/2609.22220)).
- Standard eval said **1.43×**. Hidden inputs said **0.88×** ([KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified)).
- On Apple Metal, **30% of LLM "wins" failed on a config the model hadn't seen**, with nobody prompting it to cheat ([Gaming Without an Attacker](https://arxiv.org/abs/2608.08722)).
- An all-zeros softmax **passes** KernelBench's check once rows get long ([issue #165](https://github.com/ScalingIntelligence/KernelBench)).

Generation is cheap now. **Trust is the bottleneck.** carmen is built around that.

## The idea

> The model is the intelligence. The judge is the truth. Correction is the loop between them,
> and memory is that loop's verified residue.

- **Carmy** (the model) writes the kernel. It's trusted to be smart and **never trusted to grade**. It has no tools; it can only return kernel text.
- **The judge** is plain, deterministic Python. It knows what the op *means* (a float64 answer key, properties every correct answer has, where bugs hide), runs the kernel on the real GPU, and says **where** it's wrong, not just *that* it's wrong.
- **The loop** gets from *correct* to *fast*: three Carmy drafts in parallel, the judge ranks them, the best becomes the champion, and each round attacks one bottleneck. It stops near the chip's measured memory bandwidth, the physical ceiling.
- **Memory** keeps only what the judge proved. Lessons are credited by verdicts, never by the model saying "that worked".

```mermaid
flowchart LR
    C["Carmy × 3<br/>parallel drafts"] -->|kernel text| J{{"judge<br/>compile · float64 · invariants<br/>fresh fuzz · hidden draw · timing"}}
    J -->|"located error<br/>+ speed profile"| C
    J -->|verified & faster| K(["champion"])
    K -->|"one structural change"| C
    J -->|"proven lessons"| P[("playbook")]
    P --> C
    J -.->|"hidden results<br/>(never shown to Carmy)"| R["report"]
```

## Quickstart

On an Apple Silicon Mac:

```bash
git clone https://github.com/utsav1033/kernel-sahab && cd kernel-sahab
pip install -e '.[metal,dev]'
cp .env.example .env               # put your ANTHROPIC_API_KEY in it

carmen peak                          # measure your GPU's real memory bandwidth (cached)
carmen broken softmax                # prove the judge works: it must kill every seeded bug
carmen judge softmax kernels/golden/softmax.metal --hidden
carmen run masked_softmax            # let Carmy cook
carmen report runs/<id>
```

**Keys and settings** live in `.env` (gitignored; see `.env.example`). carmen loads it on every command, and anything you `export` in your shell overrides it. Use `--env path/to/file` to load a different file.

**Using an Anthropic-compatible proxy?** Set `ANTHROPIC_BASE_URL` next to your key in `.env`. carmen then sends plain Messages API requests (no Anthropic-only extras) and never prints the URL. If the proxy strips structured outputs, add `CARMEN_STRUCTURED=0`.

### The app

Run **`carmen`** with no arguments. It follows one path:

1. **Home:** pick an op. Each shows its best verified result so far.
2. **Kitchen:** watch it cook live. A stage rail (draft → judge → improve → plated), three draft cards that go *writing → judging → ✓ verified 1.31× / ✗ tail bug at n=33*, what the judge said, and the champion with its speed by round. `s` stops after the current round.
3. **Inspect:** open any card for the code beside its verdict: speed per shape, % of peak, hidden results, and exactly what Carmy was told. `d` diffs it against the champion, `i` improves from it, `e` exports the `.metal`.
4. **Plated:** the result. Best kernel, speed vs MLX on seen and unseen sizes, and how many wrong kernels a naive check would have passed.

`t` on the home screen shows the judge catching seeded bugs live; `h` opens history.

### Commands

| command | what it does |
|---|---|
| `carmen` / `carmen ui` | the terminal app |
| `carmen ops` | list the ops the judge knows |
| `carmen peak` | copy-kernel bandwidth, used as the roofline for every speed claim |
| `carmen judge <op> <file>` | judge a kernel you wrote; `--tg 128 --tg 256` sweeps threadgroup sizes, `--hidden` adds a secret draw |
| `carmen broken <op>` | run 12 seeded broken kernels through the judge *and* through a KernelBench-style check, side by side |
| `carmen run <op>` | the self-correcting loop; `--mode bon` runs the best-of-N control arm at the same budget |
| `carmen report <run>` | the numbers below, for one run |
| `carmen playbook` | what Carmy has learned, and how much each lesson is worth |

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

**The judge is measured too.** `carmen broken` runs 12 seeded bugs across five fault families (boundary, precision, sync, indexing, semantic; see [Du et al.](https://arxiv.org/abs/2609.22220)), and reports how many carmen kills and how many a KernelBench-style check would have let through.

## Philosophy

1. **The generator is powerful and untrusted. The judge is simple and trusted.** Intelligence goes into generating, never into grading.
2. **The judge encodes what the op *means*,** not a list of tests. That part doesn't depend on the chip. Only compile/run/time does.
3. **Test where bugs live, somewhere the model can't predict:** edges, extremes, special values, and data drawn fresh every round.
4. **Self-correction is an external verdict plus a located error.** Models fix bugs well once told *where* they are.
5. **Measure the measurer.** A judge without a measured blind-spot rate is just a claim.
6. **Remember only verified causes.** When in doubt, forget.
7. **Start as a loop and a judge.** Every extra part has to earn its place against a failure seen in a real run.
8. **Report mechanisms honestly,** including when the harness fails. A verified 1.0× is a result. An unverified 3× isn't.

## What carmen reports

For every run (`carmen report`), from judge verdicts alone:

- **first-round verified**: how often Carmy is right on the first try (tests the premise that models are good)
- **naive check fooled**: kernels a KernelBench-style check accepts that carmen proves wrong
- **hidden tests clean**: verified kernels that also pass inputs they never saw
- **speedup vs MLX**, with a CI, on visible *and* hidden sizes, plus worst-case % of peak bandwidth
- **loop vs best-of-N**: run the same op with `--mode bon` at the same call budget to see whether feedback and memory earn their keep

## Adding an op or a chip

An **op** is one file in `carmen/ops/`: a float64 `reference`, `invariants`, input `generators`, a visible battery, and the kinds used for fuzz and hidden draws. The judge, the loop and Carmy don't change. Layernorm is next.

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

## Status

Honest version:

- ✅ The judge's logic, the loop, memory and the CLI are tested (`pytest`, 20 tests) against a numpy stand-in for the GPU. The tests show the judge catching kernels that a KernelBench-style check passes.
- ⚠️ **The Metal backend and the golden kernels haven't been run on Apple Silicon yet.** They're written against MLX's documented `mx.fast.metal_kernel` API. First real run: `carmen broken softmax` on a Mac.
- 🔜 layernorm, a hacker-fixer pass that attacks the judge before freezing it, hardware counters, and a dashboard over `runs/`.

## Built on the shoulders of

[Measuring the Checker](https://arxiv.org/abs/2609.22220) · [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) · [Metal-Sci](https://arxiv.org/abs/2605.09708) · [Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960) · [Test-Input Generation for Tensor Programs](https://arxiv.org/abs/2606.27396) · [Reward Hacking in Self-Improving Code Agents](https://openreview.net/forum?id=ikrQWGgxYg) · [Meta-Harness](https://arxiv.org/abs/2603.28052) · [Auditing Harness Tampering](https://arxiv.org/abs/2609.00069) · [SAGE](https://arxiv.org/abs/2609.35568) · [Prime Agent](https://arxiv.org/abs/2608.23552) · [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) · [Contract-Grade Verifier](https://github.com/RishiShah99/lethe) · [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler)

<sub>Named after Carmen "Carmy" Berzatto. Yes, chef. Banner set in Space Grotesk and JetBrains Mono (SIL Open Font License).</sub>
