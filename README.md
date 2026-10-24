<p align="center">
  <img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/banner.png" alt="A doodle of a curly-haired chef with arms crossed, next to the hand-lettered word carmen and the line: let the model cook. Trust nothing it can't prove." width="100%">
</p>

<p align="center">
  <b>An AI writes GPU kernels for Apple Silicon. A strict judge checks every one.<br>
  Then I measured what actually makes a model on a Mac faster.</b>
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
  <a href="#try-it"><b>Try it</b></a> ·
  <a href="#the-problem">The problem</a> ·
  <a href="#how-carmen-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#what-i-learned">What I learned</a>
</p>

## TL;DR

- An AI wrote Metal kernels up to **1.8× faster than Apple's MLX**, and a strict judge proved them correct.
- In real models they barely matter: **+2%** on Qwen 0.5B, **5–7% slower** on 1.5B and 3B, because MLX already runs 3B+ at the memory-bandwidth limit.
- What helps is reading less memory per token: **speculative decoding, 1.48×** on Qwen 3B at 8k context, same output.

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/spec-3b.png" alt="Qwen 2.5 3B on an M4 at 8k context: normal decoding 44.5 tokens/s, 8-bit KV cache 37.8, speculative decoding 66.0, above the ~46 tokens/s memory-bandwidth limit" width="85%"></p>

## The problem

AI can write GPU kernels now, but its kernels are often quietly wrong ([the usual check misses 78.6% of precision bugs](https://arxiv.org/abs/2609.22220)),
search loops game their own benchmarks ([30% of Metal "wins" broke on unseen configs](https://arxiv.org/abs/2608.08722)),
and a kernel that's faster alone can do nothing for a model that's already limited by memory. carmen asks:
*can an AI make a real model on my Mac faster, with proof?*

## How carmen works

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/map.png" alt="carmen, the map. Left, the kernel factory: the op spec defines what to compute; Carmy, the LLM, writes kernel code; the judge checks it is correct on hundreds of inputs including secret ones, faster than MLX and mx.compile, and still fast in the model; failures go back with where and why, passing kernels become the champion, proven lessons go to the playbook, and the next round makes one change to the champion. Right, the measurement side: carmen speedup, levers and profile all run on a real model (Qwen via mlx-lm), taking turns with stock. The finding: a kernel the judge proves faster can still do nothing for the model; only the measurement side shows it." width="95%"></p>

- **The writer can't grade itself:** the LLM returns kernel source only. No tools, no access to the tests.
- **The judge is strict and partly secret:** a float64 answer key on hundreds of inputs, edge sizes, races, writes past the end, a hidden set the model never sees, and timing against MLX and `mx.compile`. It caught **75/75** planted bugs; the usual check caught 40. [More →](https://github.com/utsav1033/carmen/blob/main/docs/judge.md)
- **Then it measures the real model,** taking turns with stock run by run, and checks the output still matches.

## Results

One M4 MacBook Pro (16 GB), MLX 0.32, greedy decoding, paired runs. Brackets are 95% confidence ranges.

**Single kernels** win where MLX runs several small kernels, and lose where Apple hand-tuned one:

| kernel | vs MLX | vs `mx.compile` |
|---|---|---|
| masked softmax | **1.82×** | **1.38×** |
| add + rmsnorm | **1.41×** | **1.43×** |
| matmul · attention | 0.93× · 0.54× | |

**In real models,** carmen's fused MLP kernels (tuned on 0.5B) mostly disappear, because stock MLX is already near the memory limit:

| Qwen 2.5, 4-bit | carmen's kernels | stock vs memory limit |
|---|---|---|
| 0.5B | **1.02×** [1.01–1.02] | 64% |
| 1.5B | 0.93× [0.93–0.95] | 87% |
| 3B | 0.95× [0.93–0.97] | 97% |

<p align="center"><img src="https://raw.githubusercontent.com/utsav1033/carmen/main/assets/speed-limit.png" alt="Stock MLX speed as a share of the memory-bandwidth limit: Qwen 0.5B 64%, 1.5B 87%, 3B 97%. Kernel fusion can only shrink the remaining part." width="85%"></p>

**What reads fewer bytes per word does help:**

| technique | model | 512 tokens | 8k tokens | output |
|---|---|---|---|---|
| speculative decoding | 3B | **1.24×** | **1.48×** | same as stock |
| | 7B | **1.19×** | **1.13×** | same as stock |
| 8-bit KV cache | 3B | 0.93× | 0.86× | 98–99% same |
| 4-bit KV cache | 3B | | | broken |

<details>
<summary>Notes on these numbers</summary>

- Speculative decoding beats the speed limit (66 vs ~46 tok/s on 3B at 8k) because it reads the weights once for several words. It helps less on 7B because the 0.5B draft is accepted less often (46–52% of words vs 57–71%).
- The 8-bit KV cache is slower because MLX runs attention on it as four separate steps instead of one fused kernel. That's carmen's next target.
- The 4-bit KV cache turns Qwen's output into gibberish ("The skyuser\nquiv"): its attention keys have outliers 4 bits can't hold.
- The memory limit uses the measured ~94 GB/s, which is conservative (stock 7B measures 103–105% of it). Read the percentages as "how close to the wall".
- Spec output matches stock until a late fp16 rounding flip, as expected when the big model checks words in a batch.

</details>

## What I learned

- AI kernels don't beat hand-tuned code. They beat code nobody fused.
- A kernel that wins alone can lose inside a model.
- Past ~1B parameters, memory is the limit, not kernels.
- Measuring honestly took more work than writing kernels.

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
| `carmen run <op> [--for qwen0.5b]` | let the AI cook one kernel; `--for` judges it at a real model's sizes |
| `carmen broken <op>` | plant bugs and check the judge catches every one, next to the standard one-size check |
| `carmen judge <op> <file>` | judge a kernel you wrote |
| `carmen e2e [model]` | run a real model stock vs with carmen's kernels: speed, noise range, same answers? |
| `carmen profile [model]` | where a model spends its time, step by step |
| `carmen bench` | the feedback loop vs plain best-of-N, same budget |

</details>

From source: `git clone https://github.com/utsav1033/carmen`, then `pip install -e '.[dev]'`. Keys go in `carmen key`, a `.env` in the folder you run from, or your shell; the shell wins.

## Built on the shoulders of

[Measuring the Checker](https://arxiv.org/abs/2609.22220) · [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) · [Metal-Sci](https://arxiv.org/abs/2605.09708) · [Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960) · [Test-Input Generation for Tensor Programs](https://arxiv.org/abs/2606.27396) · [Reward Hacking in Self-Improving Code Agents](https://openreview.net/forum?id=ikrQWGgxYg) · [Meta-Harness](https://arxiv.org/abs/2603.28052) · [Auditing Harness Tampering](https://arxiv.org/abs/2609.00069) · [SoL-Pi](https://arxiv.org/abs/2609.20519) · [SAGE](https://arxiv.org/abs/2609.35568) · [Prime Agent](https://arxiv.org/abs/2608.23552) · [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) · [Contract-Grade Verifier](https://github.com/RishiShah99/lethe) · [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler)

<sub>Named after Carmen "Carmy" Berzatto. Yes, chef. Banner drawing used with the artist's permission: fan art, not affiliated with Jeremy Allen White or *The Bear*. Lettered in Caveat (SIL Open Font License). Diagrams drawn with rough.js in Virgil (Excalidraw's font, OFL).</sub>
