#!/usr/bin/env bash
# exp_092 — 축 D(피처 부분집합)·E(손실) fit 일괄. 학습량 1632 스텝 고정.
# 🔴 CPU: NT092=1 x 동시 6 = CPU 스레드 6/40. 계산은 GPU 2장이 한다 (S102).
#    다른 세션의 단독 추론 시간 측정을 안 깨는 것이 목적이다.
set -uo pipefail
cd /home/<user>/SungjinPark/exp092
PY=/home/<user>/SungjinPark/venv/bin/python
mkdir -p logs out/preds out/npz

$PY src/mkjobs092.py out/jobs.tsv out/plan.json

NW=${NW:-6}
worker () {
  local I=$1 DEV="cuda:$(( $1 % 2 ))"
  local n=0
  # 🔴 '|' 구분자 — 탭은 IFS 공백문자라 빈 필드가 뭉개진다 (1차 발사가 그것으로 전멸)
  while IFS='|' read -r TAG MASK LOSS ROWS OFF SEED; do
    n=$((n+1))
    [ $(( (n-1) % NW )) -eq "$I" ] || continue
    [ -f "out/fit_${TAG}.json" ] && continue
    NT092=1 HID092=256,128,64 STEPS092=1632 MASK092="$MASK" LOSS092="$LOSS" \
      ROWS092="$ROWS" OFF092="$OFF" \
      nice -n 5 $PY src/train092.py oot23 "$SEED" "$DEV" ftt 0.0 adamw0.01 "$TAG" \
        > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
  done < out/jobs.tsv
  echo "WORKER_${I}_DONE"
}

for i in $(seq 0 $((NW-1))); do worker "$i" & done
wait
echo ALL_FITS_DONE
ls out/fit_*.json | wc -l
