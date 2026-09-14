#!/usr/bin/env bash
# exp_090 stage 1 — OOT 학습량 훑기. 잠근 레시피는 exp_089 와 같다:
#   init=ftt · dropout 0.0 · AdamW(wd 0.01) · lr 1e-3 상수(스케줄러 없음) · batch 4096 · D 96
# 축은 총 옵티마이저 스텝뿐. 544/720/816/1088 은 exp_089 산출물을 재사용한다(여기서 안 돈다).
set -uo pipefail
cd /home/<user>/SungjinPark/exp090
PY=/home/<user>/SungjinPark/venv/bin/python
mkdir -p logs out/preds

# ── 재현 게이트: exp_089 의 S720_oot23_s42 를 같은 인자로 다시 적합 ──────────
STEPS090=720 $PY src/train090.py oot23 42 cuda:0 ftt 0.0 adamw0.01 REPRO_S720_s42 \
    > logs/REPRO_S720_s42.log 2>&1 || echo "FAILED REPRO"

lane () {   # $1 = cuda index, 나머지 = 스텝 지점들
  local DEV="cuda:$1"; shift
  for N in "$@"; do
    for SEED in 42 777 2024 101 7; do
      TAG="K${N}_oot23_s${SEED}"
      [ -f "out/fit_${TAG}.json" ] && continue
      STEPS090=$N $PY src/train090.py oot23 "$SEED" "$DEV" ftt 0.0 adamw0.01 "$TAG" \
          > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
    done
  done
  echo "LANE_$1_DONE"
}

lane 0 8160 2720 2176 1360 &
lane 1 5440 4080 3264 1632 &
wait
echo STEPS_DONE
