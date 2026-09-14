#!/usr/bin/env bash
# exp_045 — run a sequential queue of fits pinned to one GPU.
#   queue_run.sh <gpu_index> <arch:fold:seed> [<arch:fold:seed> ...]
# At most two of these may run at once (config resources.parallel: fit 2 max,
# one GPU each).
set -euo pipefail
GPU="$1"; shift
ROOT="$HOME/SungjinPark/experiments/exp_045"
PY="$HOME/SungjinPark/venv/bin/python"
mkdir -p "$ROOT/logs"
for spec in "$@"; do
  IFS=':' read -r arch fold seed <<< "$spec"
  tag="${arch}_fold${fold}_seed${seed}"
  echo "[queue gpu$GPU] START $tag $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$PY" "$ROOT/src/train_nn.py" --fold "$fold" --seed "$seed" --arch "$arch" \
    > "$ROOT/logs/${tag}.log" 2>&1
  echo "[queue gpu$GPU] DONE  $tag $(date +%H:%M:%S) :: $(tail -n 1 "$ROOT/logs/${tag}.log")"
done
echo "[queue gpu$GPU] QUEUE COMPLETE $(date +%H:%M:%S)"
