#!/usr/bin/env bash
# exp_089 — 복원되지 않는 세 knob(w/b 초기화 · Dropout · 최적화기)을 식별하는 probe.
# 판정은 ① 접힌 지문(A·c0·T) 이 출하 5시드 범위에 드는가 ② 팀원 5시드 평균과의 rho.
set -uo pipefail
cd /home/<user>/SungjinPark/exp089
PY=/home/<user>/SungjinPark/venv/bin/python
for INIT in randn randn_w ones0 ftt; do
  for DROP in 0.0 0.1; do
    for OPT in adam adamw0.01; do
      TAG="P_${INIT}_d${DROP}_${OPT}"
      [ -f "out/fit_${TAG}.json" ] && { echo "skip $TAG"; continue; }
      echo "=== $TAG"
      $PY src/train089.py le2024 42 cuda:0 "$INIT" "$DROP" "$OPT" "$TAG" \
          > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
    done
  done
done
echo PROBE_DONE
