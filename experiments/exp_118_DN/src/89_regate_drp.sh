#!/usr/bin/env bash
# 🔴 DRP 산문 수정이 예측을 안 바꿨음을 기계로 증명한다 — 3프레임 재실행 후 sha 대조.
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames
PY=/home/<user>/SungjinPark/venv/bin/python
if [ -f "$R/work/FROZEN9" ]; then echo "DRP_REGATE_DONE"; exit 0; fi
SPECS=""
for f in real5 keep110 cov; do SPECS="$SPECS pkg/DRP_SHIP:$f:drps2__$f"; done
for f in real5 keep110 cov; do SPECS="$SPECS pkg/ANC_DRP:$f:ancrp2__$f"; done
bash "$R/src/gates_parallel.sh" "$R" "$F" "$PY" 5 $SPECS
touch "$R/work/FROZEN9"
echo "DRP_REGATE_DONE"
