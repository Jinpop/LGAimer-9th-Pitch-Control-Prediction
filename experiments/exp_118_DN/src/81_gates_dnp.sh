#!/usr/bin/env bash
# 후보 DNP 게이트 배치 — 효율 규약 7, 동시 5개(JOBS=5), 실행당 스레드 6 고정.
#
#   DNP_SHIP 10프레임  (출하 후보, u = +0.008)
#   ANC_DNP   6프레임  (항등 앵커, u = 0.0, DNP 산문)
# PAR 10판은 1차 배치에서 이미 돌았고 pkg/PAR 은 그 뒤로 안 건드렸다 — 재실행하지 않는다.
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames
PY=/home/<user>/SungjinPark/venv/bin/python
if [ -f "$R/work/FROZEN2" ]; then echo "DNP_BATCH_DONE"; exit 0; fi
mkdir -p "$R/runs" "$R/logs"
SPECS=""
for f in real5 keep110 cov band caD caE mixB mixC degen cov_shuf_al; do
  SPECS="$SPECS pkg/DNP_SHIP:$f:dnp2__$f"
done
for f in real5 keep110 cov band caD degen; do SPECS="$SPECS pkg/ANC_DNP:$f:ancp__$f"; done
bash "$R/src/gates_parallel.sh" "$R" "$F" "$PY" 5 $SPECS
touch "$R/work/FROZEN2"
echo "DNP_BATCH_DONE"
