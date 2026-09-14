#!/usr/bin/env bash
# 후보 DN 게이트 배치 — 효율 규약 7, 동시 5개(JOBS=5), 실행당 스레드 6 고정.
#
# 🔴 par__* 를 재사용하지 않는다: 부모가 DM zip 이고 우리 서버에 그 실행 기록이 없다.
#    28판 전부 이 배치에서 새로 돈다.
#
#   PAR   10프레임  (부모 기준선)
#   DN    10프레임  (출하 후보, u = -0.008)
#   ANC_DN 6프레임  (항등 앵커, u = 0.0)
#   DNP    2프레임  (곡률 측정 전용, u = +0.008 — 출하 안 함)
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames
PY=/home/<user>/SungjinPark/venv/bin/python
if [ -f "$R/work/FROZEN1" ]; then echo "DN_BATCH_DONE"; exit 0; fi
mkdir -p "$R/runs" "$R/logs"
SPECS=""
for f in real5 keep110 cov band caD caE mixB mixC degen cov_shuf_al; do
  SPECS="$SPECS pkg/PAR:$f:par__$f"
done
for f in real5 keep110 cov band caD caE mixB mixC degen cov_shuf_al; do
  SPECS="$SPECS pkg/DN:$f:dn__$f"
done
for f in real5 keep110 cov band caD degen; do SPECS="$SPECS pkg/ANC_DN:$f:anc__$f"; done
for f in cov band;                          do SPECS="$SPECS pkg/DNP:$f:dnp__$f"; done
bash "$R/src/gates_parallel.sh" "$R" "$F" "$PY" 5 $SPECS
touch "$R/work/FROZEN1"
echo "DN_BATCH_DONE"
