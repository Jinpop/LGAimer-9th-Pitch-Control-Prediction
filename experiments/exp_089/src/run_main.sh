#!/usr/bin/env bash
# exp_089 stage1(복제 검증, le2024) + stage2(정직면, oot23) — 잠근 레시피 5시드씩.
# 잠근 것: init=ftt (w,b ~ U(±1/√D)) · dropout 0.0 · AdamW(wd 0.01) · lr 1e-3 · epoch 2 · batch 4096
set -uo pipefail
cd /home/<user>/SungjinPark/exp089
PY=/home/<user>/SungjinPark/venv/bin/python
I=0
for SPLIT in le2024 oot23; do
  for SEED in 42 777 2024 101 7; do
    TAG="M_${SPLIT}_s${SEED}"
    [ -f "out/fit_${TAG}.json" ] && { echo "skip $TAG"; continue; }
    DEV="cuda:$((I % 2))"; I=$((I+1))
    echo "=== $TAG on $DEV"
    $PY src/train089.py "$SPLIT" "$SEED" "$DEV" ftt 0.0 adamw0.01 "$TAG" \
        > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
  done
done
echo MAIN_DONE
