#!/usr/bin/env bash
# 후보 DS 게이트 배치 — 효율 규약 7, 동시 5개(JOBS=5), 실행당 스레드 6 고정.
#   DS     10프레임 (출하 후보, 축6 -0.008 + 축7 -0.008)
#   ANC_DS  6프레임 (항등 앵커, 두 축 동시 0.0)
#   ZS6     3프레임 (축6 만 0.0 -> 후보 DR 과 sha 동일해야 한다)
#   ZS7     3프레임 (축7 만 0.0 -> 후보 DN 과 sha 동일해야 한다)
# PAR 10판은 1차 배치 것을 재사용한다 (pkg/PAR 은 그 뒤로 안 건드렸다).
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames
PY=/home/<user>/SungjinPark/venv/bin/python
if [ -f "$R/work/FROZEN5" ]; then echo "DS_BATCH_DONE"; exit 0; fi
mkdir -p "$R/runs" "$R/logs"
SPECS=""
for f in real5 keep110 cov band caD caE mixB mixC degen cov_shuf_al; do
  SPECS="$SPECS pkg/DS:$f:ds__$f"
done
for f in real5 keep110 cov band caD degen; do SPECS="$SPECS pkg/ANC_DS:$f:ancs__$f"; done
for f in real5 keep110 cov; do SPECS="$SPECS pkg/ZS6:$f:zs6__$f"; done
for f in real5 keep110 cov; do SPECS="$SPECS pkg/ZS7:$f:zs7__$f"; done
bash "$R/src/gates_parallel.sh" "$R" "$F" "$PY" 5 $SPECS
touch "$R/work/FROZEN5"
echo "DS_BATCH_DONE"
