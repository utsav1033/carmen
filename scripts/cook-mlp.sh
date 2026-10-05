#!/usr/bin/env bash
# The decode MLP block, end to end. ~45-90 min on an M-series Mac, 24 Carmy calls.
# Stops at the first step that fails, so you never pay for calls on top of a broken setup.
set -euo pipefail
MODEL="${1:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
cd "$(dirname "$0")/.."
mkdir -p runs
LOG="runs/cook-mlp-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "== 1/5 judge check on your GPU: hand-written kernels must pass, every broken version must be caught (free)"
carmen broken mlp_up
carmen broken mlp_down

echo; echo "== 2/5 hand-written kernels inside Qwen at decode: 2 kernels per layer instead of 8 (free)"
carmen e2e "$MODEL" --modes stock,mlp --kernel golden --repeats 5

echo; echo "== 3/5 Carmy cooks mlp_up (12 calls)"
carmen run mlp_up --rounds 4 --k 3

echo; echo "== 4/5 Carmy cooks mlp_down (12 calls)"
carmen run mlp_down --rounds 4 --k 3

echo; echo "== 5/5 Carmy's kernels inside Qwen at decode (free)"
carmen e2e "$MODEL" --modes stock,mlp --repeats 5

echo; echo "done. everything above is saved in $LOG"
