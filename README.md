<p align="center">
  <img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/banner.svg" alt="carmen: let the model cook. trust nothing it can't prove." width="100%">
</p>

<p align="center">
  <b>An AI writes GPU kernels for Apple Silicon. A strict judge checks every one.<br>
  Then I measured what actually makes a model on a Mac faster.</b>
</p>

<p align="center">
  <a href="https://github.com/utsav1033/carmen/actions/workflows/tests.yml"><img src="https://github.com/utsav1033/carmen/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
</p>

## The short version

- **The AI's kernels beat Apple's MLX by up to 1.8×** on single operations.
- **Inside a real model they barely matter:** +2% on Qwen 2.5 0.5B, and 5–7% *slower* on 1.5B and 3B.
- **Why:** on an M4, stock MLX already runs Qwen 3B at **97% of the memory speed limit**. There's almost nothing left for faster kernels to win.
- **What does help is reading less memory per word.** Speculative decoding made Qwen 3B **1.48× faster** at long context, with the same output.

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/spec-3b.png" alt="Qwen 2.5 3B on an M4 at 8k context: normal decoding 44.5 tokens/s, 8-bit KV cache 37.8, speculative decoding 66.0, above the ~46 tokens/s memory-bandwidth limit" width="85%"></p>

## What's a kernel?

A language model is a long list of small math steps: multiply by a matrix, normalize, add, repeat for every layer. Each step
runs on the GPU as a small program called a **kernel**. Making a kernel faster, or merging several into one (**fusion**),
is the classic way to speed a model up. Kernels are hard to write and easy to get subtly wrong, which makes them a
good test for an AI that writes code.

## How carmen works

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/loop.png" alt="The AI writes a kernel and sends the code to the judge. The judge checks it is right on hundreds of inputs including secret ones, faster than Apple's MLX, and still fast inside a real model. Failures go back to the AI with exactly where the kernel is wrong. Passing kernels become the champion, and the next round improves on it." width="90%"></p>

- **The AI (Carmy)** writes a kernel in Metal, Apple's GPU language. It only ever returns code; it never runs or grades anything.
- **The judge** is plain Python. It compares every kernel against an exact answer key on hundreds of inputs, including
  secret ones drawn fresh each round, races it against MLX, and says *where* a kernel is wrong ("row 45, last column"),
  not just that it is.
- **The loop** keeps the fastest correct kernel as the champion, and the next round tries to improve it. Lessons are
  kept only when the judge proved them.

## Why the judge matters

Most AI-kernel benchmarks check one input size against a loose tolerance. I planted **75 realistic bugs** in seven
kernels (off-by-one at the end of a row, precision loss, missing synchronization, bugs that only show on non-square
matrices) and checked both ways:

| | bugs caught |
|---|---|
| a one-size, loose-tolerance check (the common standard) | 40 of 75 |
| **carmen's judge** | **75 of 75** |

That matches published work: the standard check misses most precision bugs ([Measuring the Checker](https://arxiv.org/abs/2609.22220)),
and reported speedups shrink on inputs the AI hadn't seen ([KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified)).
Writing kernels is cheap now. **Trusting them is the hard part.** [How the judge works →](https://github.com/utsav1033/carmen/blob/main/docs/judge.md)

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

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/speed-limit.png" alt="Stock MLX speed as a share of the memory-bandwidth limit: Qwen 0.5B 64%, 1.5B 87%, 3B 97%. Kernel fusion can only shrink the remaining part." width="85%"></p>

Generating each word means reading every weight in the model once. The M4 moves about 94 GB/s, so a 3B model
(1.74 GB of weights) can't go faster than ~53 words per second. Stock MLX already does ~52. The rest of the time,
the part fusion can shrink, is only 3%.

**3. What does help on bigger models: reading fewer bytes per word.** Qwen 2.5 3B, 10 turns each:

| technique | 512-token prompt | 8,192-token prompt | output |
|---|---|---|---|
| **speculative decoding** (0.5B guesses, 3B checks several words at once) | **1.24×** [1.20–1.28] | **1.48×** [1.46–1.53] | same as stock* |
| 8-bit KV cache | 0.93× [0.93–0.94] | 0.86× [0.83–0.89] | 98–99% same next word |
| 4-bit KV cache | | | broken: "The skyuser\nquiv" |

<sub>*Same words until a late fp16 rounding flip (word 95 of 256 at 512 tokens, 207 at 8k), as expected from checking words in a batch.</sub>

- Speculative decoding runs at 66 words/s at 8k, **past the ~46 words/s limit**, because it reads the weights once for several words.
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
  swapping memory made stock look like 82% of the limit instead of 97%.

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

## Glossary

- **Kernel:** one small program the GPU runs, like "normalize these rows".
- **Fusion:** merging several kernels into one, so data stays on the chip instead of going back to memory in between.
- **Decode:** generating the answer one word (token) at a time. Each word reads every weight once.
- **Memory speed limit:** model size ÷ memory bandwidth. The fastest decode can go, however good the kernels are.
- **KV cache:** what the model remembers about the prompt so far. It grows with the prompt and is read for every new word.
- **Speculative decoding:** a small model guesses a few words, the big model checks them all in one pass.
- **Quantization (4-bit, 8-bit):** storing numbers in fewer bits, to read less memory. Saves bytes, can cost accuracy.
- **`mx.compile`:** MLX's built-in automatic fusion. The fairest baseline, because every MLX user gets it for free.
- **Paired turns:** stock and the change take turns run by run, so a Mac that heats up or gets busy hits both alike.

## Built on the shoulders of

[Measuring the Checker](https://arxiv.org/abs/2609.22220) · [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) · [Metal-Sci](https://arxiv.org/abs/2605.09708) · [Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960) · [Test-Input Generation for Tensor Programs](https://arxiv.org/abs/2606.27396) · [Reward Hacking in Self-Improving Code Agents](https://openreview.net/forum?id=ikrQWGgxYg) · [Meta-Harness](https://arxiv.org/abs/2603.28052) · [Auditing Harness Tampering](https://arxiv.org/abs/2609.00069) · [SoL-Pi](https://arxiv.org/abs/2609.20519) · [SAGE](https://arxiv.org/abs/2609.35568) · [Prime Agent](https://arxiv.org/abs/2608.23552) · [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) · [Contract-Grade Verifier](https://github.com/RishiShah99/lethe) · [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler)

<sub>Named after Carmen "Carmy" Berzatto. Yes, chef. Banner set in Space Grotesk and JetBrains Mono (SIL Open Font License). Diagrams drawn with rough.js in Virgil (Excalidraw's font, OFL).</sub>
