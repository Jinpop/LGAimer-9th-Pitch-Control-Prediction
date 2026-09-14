#!/usr/bin/env bash
# exp_090 stage 1 보강 — 꺾이는 지점 주변을 촘촘히. 인자로 준 스텝마다 5시드.
# 사용: run_fine.sh <N1> [N2 ...]
set -uo pipefail
cd /home/<user>/SungjinPark/exp090
PY=/home/<user>/SungjinPark/venv/bin/python
mkdir -p logs out/preds
I=0
for N in "$@"; do
  for SEED in 42 777 2024 101 7; do
    TAG="K${N}_oot23_s${SEED}"
    [ -f "out/fit_${TAG}.json" ] && continue
    DEV="cuda:$((I % 2))"; I=$((I+1))
    ( STEPS090=$N $PY src/train090.py oot23 "$SEED" "$DEV" ftt 0.0 adamw0.01 "$TAG" \
        > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG" ) &
    [ $((I % 2)) -eq 0 ] && wait
  done
done
wait
echo FINE_DONE
