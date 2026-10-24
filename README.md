<p align="center">
  <img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/banner.png" alt="A pencil-and-ink sketch: a chef's knife has just diced a GPU chip into small tiles on a cutting board. Beside it, hand-lettered: carmen. Let the model cook. Trust nothing it can't prove." width="100%">
</p>

<p align="center">
  <b>An AI writes GPU kernels for Apple Silicon. A strict judge checks every one.<br>
  Then I measured what actually makes a model on a Mac faster.</b>
</p>

<p align="center">
  <a href="https://github.com/utsav1033/carmen/actions/workflows/tests.yml"><img src="https://github.com/utsav1033/carmen/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
</p>

## The short version

- **An AI wrote Metal kernels that beat Apple's MLX by up to 1.8×** on single operations, and a strict judge proved them correct.
- **Inside a real model they barely matter:** +2% on Qwen 2.5 0.5B, and 5–7% *slower* on 1.5B and 3B.
- **Why:** on an M4, stock MLX already decodes Qwen 3B and 7B **at the memory-bandwidth limit**. There's almost nothing left for kernels to win.
- **What does help is reading less memory per token.** Speculative decoding made Qwen 3B **1.48× faster** at an 8k prompt, with the same output. On 7B it gave less (1.13–1.19×), because a 0.5B draft guesses 7B's words less often.

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/spec-3b.png" alt="Qwen 2.5 3B on an M4 at 8k context: normal decoding 44.5 tokens/s, 8-bit KV cache 37.8, speculative decoding 66.0, above the ~46 tokens/s memory-bandwidth limit" width="85%"></p>

## The problem

**Kernels.** Every step of a transformer (each matmul, norm, softmax, residual add) runs on the GPU as a
kernel: a small program that thousands of threads execute at once. A model's speed is the sum of its kernels'
time, plus a fixed cost for launching each one. Faster kernels, or fusing several into one so intermediate
results never leave the chip, is the standard way to make inference faster.

**The bet.** Good kernels are slow, expert work, and LLMs can now write them. Papers report big speedups from
AI-written kernels. carmen asks: *can an AI make a real model on my Mac faster, with proof?*

**What makes it hard.**

1. **AI-written kernels are often quietly wrong.** GPU bugs hide: a row's last element isn't written, a sum in
   fp16 drifts, a missing barrier fails once in a hundred runs, indexing that's only right on square matrices.
   The usual check (one input size, a loose `allclose`) misses most of these: it lets through
   [78.6% of precision bugs](https://arxiv.org/abs/2609.22220), and [a reported 1.43× was 0.88×](https://github.com/facebookresearch/kernel_bench_verified)
   once tested properly.
2. **Search loops game their benchmark without anyone trying.** When an LLM iterates against a score,
   [30% of its "wins" on Metal broke](https://arxiv.org/abs/2608.08722) on a configuration it hadn't been tested on.
3. **A kernel that's faster alone may not make the model faster.** Decoding is limited by how fast the chip reads
   the weights: every token reads every weight once. If a model already runs near that limit, a faster kernel
   for the remaining few percent changes nothing. You only find out by measuring the whole model, carefully.

## How carmen works

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/map.png" alt="carmen, the map. Left, the kernel factory: the op spec defines what to compute; Carmy, the LLM, writes kernel code; the judge checks it is correct on hundreds of inputs including secret ones, faster than MLX and mx.compile, and still fast in the model; failures go back with where and why, passing kernels become the champion, proven lessons go to the playbook, and the next round makes one change to the champion. Right, the measurement side: carmen speedup, levers and profile all run on a real model (Qwen via mlx-lm), taking turns with stock. The finding: a kernel the judge proves faster can still do nothing for the model; only the measurement side shows it." width="95%"></p>

**The writer can't grade itself.** Carmy (an LLM) returns kernel source and nothing else: no tools, no access to
the tests or the timer. Everything it learns comes from the judge's verdicts.

**The judge is strict, specific and partly secret.** Plain Python, in a separate process. Each kernel is checked
against a float64 answer key on hundreds of inputs: edge lengths (1, 7, 33, 257, 4097...), huge values, masked
rows, fp32, fp16 and bf16. The output buffer is pre-filled with NaN so unwritten elements show up, a guard zone
catches writes past the end, and big inputs run five times and must match byte for byte (races). There are fresh
random sizes every round, plus a hidden set the model never sees, for speed as well as correctness. Failures come
back located: *"NaN in the last column of every row: tail handling, smallest failing input 1×2"*. Speed is timed
interleaved against MLX **and** MLX's own fusion (`mx.compile`), with a confidence range.

**The judge is tested too.** I planted 75 realistic bugs in seven kernels: the usual one-size check catches 40,
carmen's judge catches all 75. [How the judge works →](https://github.com/utsav1033/carmen/blob/main/docs/judge.md)

**The loop.** Three drafts per round, in parallel. The fastest verified kernel becomes the champion, and the next
round must make one structural change to it. Lessons carry across runs only if the judge's verdicts back them.

**Then it measures the real model.** `carmen speedup` and `carmen levers` run a whole model (Qwen, through mlx-lm)
stock vs changed, taking turns run by run so heat and background load hit both alike, and check that the
generated tokens still match.

## Results

All numbers are from one MacBook Pro with an M4 (16 GB), greedy decoding; whole-model runs on MLX 0.32. Each comparison takes turns with stock MLX
run by run, and the range in brackets is a 95% confidence range.

**1. Single kernels: the AI wins where MLX runs several small kernels, and loses where Apple hand-tuned one.**

| kernel | vs MLX | vs MLX's own fusion (`mx.compile`) |
|---|---|---|
| masked softmax (MLX: 3 kernels) | **1.82×** | **1.38×** |
| add + rmsnorm (MLX: 2 kernels) | **1.41×** | **1.43×** |
| 4-bit MLP down-projection (MLX: 2 kernels) | 1.02× | 1.06× |
| softmax · rmsnorm · layernorm (MLX: 1 tuned kernel) | 0.95–0.99× | |
| matmul · attention (MLX: heavily tuned) | 0.93× · 0.54× | |

**2. Whole models: those wins mostly disappear.** carmen's fused MLP kernels (8 kernels per layer become 2), tuned on Qwen 0.5B:

| Qwen 2.5, 4-bit | carmen's kernels vs stock | stock speed vs the memory limit |
|---|---|---|
| 0.5B | **1.02×** [1.01–1.02] | 64% |
| 1.5B | 0.93× [0.93–0.95] | 87% |
| 3B | 0.95× [0.93–0.97] | 97% |
| 7B | not run | 103–105% |

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/speed-limit.png" alt="Stock MLX speed as a share of the memory-bandwidth limit: Qwen 0.5B 64%, 1.5B 87%, 3B 97%. Kernel fusion can only shrink the remaining part." width="85%"></p>

Generating each word means reading every weight in the model once. I measured the M4 moving ~94 GB/s, so a 3B model
(1.74 GB of weights) can't go much faster than ~53 words per second. Stock MLX already does ~52. The rest of the time,
the part fusion can shrink, is a few percent.

The limit is only as good as my bandwidth measurement, and that's conservative: stock 7B comes out at 103–105% of it,
which is impossible, so the chip actually moves a little more than 94 GB/s (Apple quotes 120). Read the last column as
"how close to the limit", not as exact percentages. The conclusion doesn't change: from 3B up, stock MLX is at the wall.

**3. What does help on bigger models: reading fewer bytes per word.** 10 turns each:

| technique | model | 512-token prompt | 8,192-token prompt | output |
|---|---|---|---|---|
| **speculative decoding** (Qwen 0.5B guesses, the big model checks several words at once) | 3B | **1.24×** [1.20–1.28] | **1.48×** [1.46–1.53] | same as stock* |
| | 7B | **1.19×** [1.18–1.22] | **1.13×** [1.12–1.15] | same as stock* |
| 8-bit KV cache | 3B | 0.93× [0.93–0.94] | 0.86× [0.83–0.89] | 98–99% same next word |
| | 7B | 0.98× [0.98–0.98] | 0.92× [0.91–0.92] | 95–98% same next word |
| 4-bit KV cache | 3B | | | broken: "The skyuser\nquiv" |

<sub>*Same words until a late fp16 rounding flip (3B: word 95 of 256 at 512 tokens, 207 at 8k; 7B: 211 and 81), as expected from checking words in a batch.</sub>

- On 3B, speculative decoding runs at 66 words/s at 8k, **past the ~46 words/s limit** for normal decoding, because it reads the weights once for several words.
- **It pays less on 7B, not more.** The 0.5B draft agrees with 7B less often: 46–52% of words came from the draft, against
  57–71% with 3B. A bigger model diverges more from a tiny one, and that outweighs the draft being relatively cheaper.
  A better-matched draft (Qwen 1.5B) is the obvious next test.
- The 8-bit KV cache reads less memory but runs *slower*. MLX computes attention on it as four separate steps
  instead of its usual single fused kernel, and that overhead grows with the prompt. A fused kernel for it is carmen's next target.
- The 4-bit KV cache breaks Qwen, whose attention keys have outliers that 4 bits can't hold. Speed alone would have called it a win.

## What I learned

- **AI-written kernels don't beat hand-tuned code. They beat code nobody fused.**
- **A kernel that wins alone can lose inside a model.** One scored 1.15× in the judge and made Qwen 18% slower: it was
  timed at the wrong sizes, against a weak baseline. Now `carmen run <op> --for qwen0.5b` judges at the model's real sizes and
  checks the winner inside the model.
- **Past ~1B parameters, the model is limited by memory, not by kernels.** The wins come from reading fewer bytes
  per word (speculative decoding, a smaller KV cache), not from faster kernels for the same work.
- **Measuring honestly took more work than writing kernels.** Running versions one after another instead of taking turns
  produced two fake wins (+40% prefill, 1.09× decode). A browser in the background swung speed by ±27%. A Mac that was
  swapping memory made stock look like 82% of the limit instead of 97%. And the limit itself is a measurement: 7B
  "beating" it showed my bandwidth number was conservative.
- **Speculative decoding's payoff depends on the draft, not only on model size.** It went 1.48× on 3B but 1.13× on 7B,
  with the same 0.5B draft.

New to all this? [How it all works](https://github.com/utsav1033/carmen/blob/main/docs/how-it-works.md) explains LLMs, GPUs and kernels from scratch.

## Try it

On an Apple Silicon Mac:

```bash
pip install carmen-kernels          # or: uv tool install carmen-kernels
carmen                              # the app
carmen speedup                      # Qwen 2.5 0.5B, stock vs carmen's kernels, on your Mac (no API key)
carmen speedup qwen0.5b qwen3b      # several models in turn, then one table side by side
carmen levers qwen3b                # speculative decoding, KV cache quantization, 3-bit weights vs stock (no API key)
carmen key                          # only to cook new kernels: paste an Anthropic key once, saved for you only
```

| command | what it does |
|---|---|
| `carmen` | the terminal app: what's proven, speed up a model, pick a kernel, watch it cook |
| `carmen speedup [models…]` | stock MLX vs `mx.compile` vs carmen's bundled kernels on real models, taking turns; no API key (`--quick` for ~2 min a model) |
| `carmen levers [model]` | speculative decoding, 8/4-bit KV cache and 3-bit weights vs stock, at 512 and 8k prompts, with drift scored for the lossy ones; no API key |
| `carmen run <op> [--for qwen0.5b]` | let the AI cook one kernel; `--for` judges it at a real model's sizes |
| `carmen broken <op>` | plant bugs and check the judge catches every one, next to the standard one-size check |
| `carmen judge <op> <file>` | judge a kernel you wrote |
| `carmen e2e [model]` | run a real model stock vs with carmen's kernels: speed, noise range, same answers? |
| `carmen profile [model]` | where a model spends its time, step by step |
| `carmen bench` | the feedback loop vs plain best-of-N, same budget |

From source: `git clone https://github.com/utsav1033/carmen`, then `pip install -e '.[dev]'`. Keys go in `carmen key`, a `.env` in the folder you run from, or your shell; the shell wins.

## Built on the shoulders of

[Measuring the Checker](https://arxiv.org/abs/2609.22220) · [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) · [Metal-Sci](https://arxiv.org/abs/2605.09708) · [Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960) · [Test-Input Generation for Tensor Programs](https://arxiv.org/abs/2606.27396) · [Reward Hacking in Self-Improving Code Agents](https://openreview.net/forum?id=ikrQWGgxYg) · [Meta-Harness](https://arxiv.org/abs/2603.28052) · [Auditing Harness Tampering](https://arxiv.org/abs/2609.00069) · [SoL-Pi](https://arxiv.org/abs/2609.20519) · [SAGE](https://arxiv.org/abs/2609.35568) · [Prime Agent](https://arxiv.org/abs/2608.23552) · [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) · [Contract-Grade Verifier](https://github.com/RishiShah99/lethe) · [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler)

<sub>Named after Carmen "Carmy" Berzatto. Yes, chef. Banner hand-drawn with rough.js, lettered in Caveat (SIL Open Font License). Diagrams drawn with rough.js in Virgil (Excalidraw's font, OFL).</sub>
