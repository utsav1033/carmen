#!/usr/bin/env bash
# Does Qwen 0.5B write words faster, with the same answers? Run on an Apple Silicon Mac
# with `pip install -e '.[models]'` done. No API calls, no cost. ~5-10 minutes.
# Close other heavy apps first: the GPU is shared.
set -euo pipefail
MODEL="${1:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"

echo "== 1/2: one residual add + norm call, timed several ways (GPU warmed first)"
carmen e2e "$MODEL" --calls

echo
echo "== 2/2: the whole model, 5 versions taking turns, 5 turns each, plus the decode speed limit"
carmen e2e "$MODEL" --modes stock,plumbing,compile,kernel,all --repeats 5
