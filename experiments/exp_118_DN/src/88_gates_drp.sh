#!/usr/bin/env bash
# 후보 DRP 게이트 배치 — 규약 7, 동시 5개(JOBS=5), 실행당 스레드 6 고정.
#   DRP_SHIP 10프레임 (출하 후보, pstr u = +0.008)
#   ANC_DRP   6프레임 (항등 앵커, u = 0.0)
# 🔴 곡률은 새로 안 잰다 — 같은 축·같은 center/scale 의 ±0.008 두 판(dr__ / drp__)이
#    이미 배치 4 에서 돌아 work/curvature_dr.json 에 있다 (Q cov 1.8313 · band 1.8652).
# PAR 10판은 1차 배치 것을 재사용한다.
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames
PY=/home/<user>/SungjinPark/venv/bin/python
if [ -f "$R/work/FROZEN8" ]; then echo "DRP_BATCH_DONE"; exit 0; fi
SPECS=""
for f in real5 keep110 cov band caD caE mixB mixC degen cov_shuf_al; do
  SPECS="$SPECS pkg/DRP_SHIP:$f:drps__$f"
done
for f in real5 keep110 cov band caD degen; do SPECS="$SPECS pkg/ANC_DRP:$f:ancrp__$f"; done
bash "$R/src/gates_parallel.sh" "$R" "$F" "$PY" 5 $SPECS
touch "$R/work/FROZEN8"
echo "DRP_BATCH_DONE"
