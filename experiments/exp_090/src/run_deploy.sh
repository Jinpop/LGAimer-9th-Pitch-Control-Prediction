#!/usr/bin/env bash
# exp_090 stage 2 — 배포 구간(le2024, 1,475,092행) 재학습. 인자로 준 총 스텝마다 5시드.
# 사용: run_deploy.sh <STEPS1> [STEPS2 ...]
# 학습 레시피는 stage 1 과 **완전히 같다** — 바뀌는 것은 학습 구간과 총 스텝뿐.
set -uo pipefail
cd /home/<user>/SungjinPark/exp090
PY=/home/<user>/SungjinPark/venv/bin/python
mkdir -p logs out/preds out/npz
I=0
for N in "$@"; do
  for SEED in 42 777 2024 101 7; do
    TAG="D${N}_le2024_s${SEED}"
    [ -f "out/fit_${TAG}.json" ] && continue
    DEV="cuda:$((I % 2))"; I=$((I+1))
    ( SAVE_NPZ=1 STEPS090=$N $PY src/train090.py le2024 "$SEED" "$DEV" ftt 0.0 \
        adamw0.01 "$TAG" > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG" ) &
    [ $((I % 2)) -eq 0 ] && wait
  done
done
wait
echo DEPLOY_DONE
