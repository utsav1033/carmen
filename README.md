<p align="center">
  <img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/banner.png" alt="A doodle of a curly-haired chef with arms crossed, next to the hand-lettered word carmen and the line: let the model cook. Trust nothing it can't prove." width="100%">
</p>

<p align="center">
  <b>An LLM writes GPU kernels for Apple Silicon and carmen tests every one.<br>
  Then it measures whether a real model on a Mac actually gets faster.</b>
</p>

<p align="center">
  <a href="https://github.com/utsav1033/carmen/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/utsav1033/carmen/tests.yml?branch=main&style=flat-square&label=tests" alt="tests"></a>
  <img src="https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M4-262626?style=flat-square&logo=apple&logoColor=white" alt="runs on Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.10%2B-262626?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/measuring-no%20API%20key-2f9e44?style=flat-square" alt="measuring needs no API key">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-262626?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#tldr">TL;DR</a> ·
  <a href="#what-it-looks-like">What it looks like</a> ·
  <a href="#try-it"><b>Try it</b></a> ·
  <a href="#the-problem">The problem</a> ·
  <a href="#how-carmen-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#what-i-learned">What I learned</a>
</p>

## TL;DR

- Claude wrote Metal kernels up to **1.8× faster than Apple's MLX** on single operations, checked on hundreds of inputs.
- In real models they barely helped: **+2%** on Qwen 0.5B, **5–7% slower** on 1.5B and 3B. MLX already runs those models about as fast as the Mac's memory allows.
- Reading less memory per word did help: **speculative decoding, a feature mlx-lm already has, made Qwen 3B 1.48× faster** at an 8k prompt, with the same output. carmen measured it; it didn't invent it.

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/spec-3b.png" alt="Qwen 2.5 3B on an M4 at 8k context: normal decoding 44.5 tokens/s, 8-bit KV cache 37.8, speculative decoding 66.0, above the ~46 tokens/s memory-bandwidth limit" width="85%"></p>

## What it looks like

<p align="center">
  <img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/app-home.webp" alt="carmen's terminal app, home screen: what's been proven so far (1.03× decode on Qwen 0.5B, 75/75 planted bugs caught, 1.82× vs MLX on masked softmax), a list of models with their size, whether they fit this Mac and the best result measured on each, and the kernels carmen has written with their speedups" width="49%">
  <img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/app-model.webp" alt="carmen's model page for Qwen 2.5 3B: size, memory fit, the draft model, and everything measured on this Mac (speculative decoding 1.24× and 1.48×, 8-bit KV cache 0.93× and 0.86×, 4-bit KV cache marked broken output), next to what carmen can do: measure speedup or levers, or cook a kernel for this model's shapes" width="49%">
</p>

Run `carmen`, press `m`, pick a model: it shows what's been measured on it and what you can run, no API key needed.

## The problem

LLMs can write GPU kernels, but a kernel can pass a quick test and still be wrong. Most AI-kernel benchmarks try a
single input and accept any answer that's roughly close. So a kernel that works on 1,024 numbers but forgets the last
one when there are 1,025, or one whose answers are slightly off, still counts as correct
([one study found this kind of check misses most precision bugs](https://arxiv.org/abs/2609.22220)).

And even a kernel that really is correct and faster on its own may not make the model any faster.
I wanted to know if an LLM could make a model on my Mac faster, and prove it.

## How carmen works

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/map.png" alt="How carmen works: the task says what to compute and what the exact answer is. Claude writes the kernel and sends back code only. carmen's tests check it is correct on hundreds of inputs, including ones Claude never sees, faster than MLX and mx.compile, and still fast in the model. A failure goes back to Claude with where and why; a pass becomes the champion, the fastest correct kernel so far. Tips the tests proved go into a playbook, and the next round makes one change to the champion." width="75%"></p>

At its core it's a loop: Claude writes a Metal kernel, carmen tests it, and the fastest correct one is kept and
improved. Plenty of projects do that. The difference is that carmen doesn't trust the loop, because a plain loop
fools itself in three ways:

- **The tests miss bugs.** So carmen's tests are tested: I planted 75 realistic bugs in correct kernels.
  carmen caught all 75; the usual one-input check caught 40. Failures tell Claude exactly where it went wrong.
- **The model learns to pass the tests instead of being correct.** So part of the test set is hidden: inputs Claude
  never sees, for correctness and for speed. Without that, [30% of LLM kernel "wins" on Metal broke](https://arxiv.org/abs/2608.08722)
  on inputs nobody had tested.
- **"Faster" doesn't mean the model gets faster.** So every win is timed against MLX's own fusion with error bars,
  then checked again inside a real model. That's how carmen found that kernels which win on their own barely move Qwen at all.

[How the testing works →](https://github.com/utsav1033/carmen/blob/main/docs/judge.md)

## Results

One M4 MacBook Pro (16 GB), MLX 0.32, greedy decoding, paired runs. Brackets are 95% confidence ranges.

**Single kernels** win where MLX runs several small kernels, and lose where Apple hand-tuned one:

| kernel | MLX | carmen | speedup |
|---|---|---|---|
| masked softmax | 3 kernels | 1 fused kernel | **1.82×** |
| add + rmsnorm | 2 kernels | 1 fused kernel | **1.41×** |
| matmul | Apple's tuned kernel | 1 kernel | 0.93× |
| attention | Apple's tuned kernel | 1 kernel | 0.54× |

Against MLX's own automatic fusion (`mx.compile`), the first two are still 1.38× and 1.43×.

**In real models,** carmen's fused MLP kernels (tuned on 0.5B) mostly disappear, because MLX is already close to the fastest the Mac can go:

| Qwen 2.5, 4-bit | MLX | carmen | speedup | fastest possible* |
|---|---|---|---|---|
| 0.5B | 214 tok/s | 220 tok/s | **1.02×** | ~337 tok/s |
| 1.5B | 94 tok/s | 87 tok/s | 0.93× | ~108 tok/s |
| 3B | 44 tok/s | 42 tok/s | 0.95× | ~53 tok/s |

<sub>*Each new word means reading all of the model's weights from memory once. My M4 reads about 94 GB/s, so a 1.74 GB
model (3B) can't go much past ~53 words per second, however good the kernels are. MLX is already at 44 of 53; on 0.5B it's
at 214 of 337, which is where carmen's kernels still had room.</sub>

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/speed-limit.png" alt="Stock MLX speed as a share of the memory-bandwidth limit: Qwen 0.5B 64%, 1.5B 87%, 3B 97%. Kernel fusion can only shrink the remaining part." width="85%"></p>

**What reads fewer bytes per word does help.** These are existing mlx-lm features, not carmen's: carmen turns them on and measures them the same way (`carmen levers`).

Decode speed in tokens per second, stock MLX → with the technique:

| technique | model | 512-token prompt | 8k-token prompt | output |
|---|---|---|---|---|
| speculative decoding | 3B | 51.8 → 64.7 · **1.24×** | 44.5 → 66.0 · **1.48×** | same as stock |
| | 7B | 24.1 → 28.6 · **1.19×** | 22.1 → 24.8 · **1.13×** | same as stock |
| 8-bit KV cache | 3B | 51.8 → 48.7 · 0.93× | 44.5 → 37.8 · 0.86× | 98–99% same |
| 4-bit KV cache | 3B | | | broken |

<details>
<summary>Notes on these numbers</summary>

- Speculative decoding beats the speed limit (66 vs ~46 tok/s on 3B at 8k) because it reads the weights once for several words. It helps less on 7B because the 0.5B draft is accepted less often (46–52% of words vs 57–71%).
- The 8-bit KV cache is slower because MLX runs attention on it as four separate steps instead of one fused kernel. That's carmen's next target.
- The 4-bit KV cache turns Qwen's output into gibberish ("The skyuser\nquiv"): its attention keys have outliers 4 bits can't hold.
- The "fastest possible" figures use my measured ~94 GB/s, which is a bit low (7B actually runs slightly faster than its figure), so treat them as approximate.
- Spec output matches stock until a late fp16 rounding flip, as expected when the big model checks words in a batch.

</details>

## What I learned

- AI kernels don't beat hand-tuned code. They beat code nobody fused.
- A kernel that wins alone can lose inside a model.
- Past ~1B parameters, memory is the limit, not kernels.
- Getting numbers I could trust took more work than writing the kernels.

New to this? [How it all works](https://github.com/utsav1033/carmen/blob/main/docs/how-it-works.md) explains LLMs, GPUs and kernels from scratch.

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

<details>
<summary>All commands</summary>

| command | what it does |
|---|---|
| `carmen` | the terminal app: what's proven, speed up a model, pick a kernel, watch it cook |
| `carmen speedup [models…]` | stock MLX vs `mx.compile` vs carmen's bundled kernels on real models, taking turns; no API key (`--quick` for ~2 min a model) |
| `carmen levers [model]` | speculative decoding, 8/4-bit KV cache and 3-bit weights vs stock, at 512 and 8k prompts, with drift scored for the lossy ones; no API key |
| `carmen run <op> [--for qwen0.5b]` | let Claude write one kernel; `--for` tests it at a real model's sizes |
| `carmen broken <op>` | plant bugs and check carmen's tests catch every one, next to the usual one-size check |
| `carmen judge <op> <file>` | test a kernel you wrote |
| `carmen e2e [model]` | run a real model stock vs with carmen's kernels: speed, noise range, same answers? |
| `carmen profile [model]` | where a model spends its time, step by step |
| `carmen bench` | the feedback loop vs plain best-of-N, same budget |

</details>

From source: `git clone https://github.com/utsav1033/carmen`, then `pip install -e '.[dev]'`. Keys go in `carmen key`, a `.env` in the folder you run from, or your shell; the shell wins.

## Built on the shoulders of

[Measuring the Checker](https://arxiv.org/abs/2609.22220) · [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) · [Metal-Sci](https://arxiv.org/abs/2605.09708) · [Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960) · [Test-Input Generation for Tensor Programs](https://arxiv.org/abs/2606.27396) · [Reward Hacking in Self-Improving Code Agents](https://openreview.net/forum?id=ikrQWGgxYg) · [Meta-Harness](https://arxiv.org/abs/2603.28052) · [Auditing Harness Tampering](https://arxiv.org/abs/2609.00069) · [SoL-Pi](https://arxiv.org/abs/2609.20519) · [SAGE](https://arxiv.org/abs/2609.35568) · [Prime Agent](https://arxiv.org/abs/2608.23552) · [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) · [Contract-Grade Verifier](https://github.com/RishiShah99/lethe) · [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler)

<sub>Named after Carmen "Carmy" Berzatto. Banner drawing used with the artist's permission: fan art, not affiliated with Jeremy Allen White or *The Bear*. Lettered in Caveat (SIL Open Font License). Diagrams drawn with rough.js in Virgil (Excalidraw's font, OFL).</sub>
