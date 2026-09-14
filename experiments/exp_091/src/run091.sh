#!/usr/bin/env bash
# exp_091 — 축 셋(시드 수·구조·정규화) fit 일괄. 학습량은 1632 스텝 고정.
# 🔴 CPU 상한: NT091=1 x 동시 6 = CPU 스레드 6. 계산은 GPU 2장이 한다.
#    NT091=1 이 z 를 안 바꾼다는 것은 재현 게이트(GATE_NT1_s42)가 sha 로 증명했다.
set -uo pipefail
cd /home/<user>/SungjinPark/exp091
PY=/home/<user>/SungjinPark/venv/bin/python
mkdir -p logs out/preds out/npz

$PY src/mkjobs091.py out/jobs.tsv out/plan.json

NW=6
worker () {
  local I=$1 DEV="cuda:$(( $1 % 2 ))"
  local n=0
  while IFS=$'\t' read -r TAG HID DR OPT SEED; do
    n=$((n+1))
    [ $(( (n-1) % NW )) -eq "$I" ] || continue
    [ -f "out/fit_${TAG}.json" ] && continue
    NT091=1 HID091="$HID" STEPS091=1632 $PY src/train091.py \
        oot23 "$SEED" "$DEV" ftt "$DR" "$OPT" "$TAG" \
        > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
  done < out/jobs.tsv
  echo "WORKER_${I}_DONE"
}

for i in $(seq 0 $((NW-1))); do worker "$i" & done
wait
echo ALL_FITS_DONE
ls out/fit_*.json | wc -l
