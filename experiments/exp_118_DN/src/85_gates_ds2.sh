#!/usr/bin/env bash
# 후보 DS(개정) 게이트 배치 — 효율 규약 7, 동시 5개(JOBS=5), 실행당 스레드 6 고정.
#   DS2      10프레임 (출하 후보: mid -0.00913 + bmid -0.00506)
#   ANC_DS2   6프레임 (앵커 ① 두 축 동시 부모값 -> 부모 sha)
#   ZB        3프레임 (앵커 ② bmid 만 0.0)
#   ZM        3프레임 (앵커 ② 기준: 부모 5축 script 로 mid 만 -0.00913)
#   ZMID      3프레임 (앵커 ③ mid 만 부모값)
# PAR 10판은 1차 배치 것을 재사용한다.
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames
PY=/home/<user>/SungjinPark/venv/bin/python
if [ -f "$R/work/FROZEN6" ]; then echo "DS2_BATCH_DONE"; exit 0; fi
mkdir -p "$R/runs" "$R/logs"
SPECS=""
for f in real5 keep110 cov band caD caE mixB mixC degen cov_shuf_al; do
  SPECS="$SPECS pkg/DS2:$f:ds2__$f"
done
for f in real5 keep110 cov band caD degen; do SPECS="$SPECS pkg/ANC_DS2:$f:ancs2__$f"; done
for f in real5 keep110 cov; do SPECS="$SPECS pkg/ZB:$f:zb__$f"; done
for f in real5 keep110 cov; do SPECS="$SPECS pkg/ZM:$f:zm__$f"; done
for f in real5 keep110 cov; do SPECS="$SPECS pkg/ZMID:$f:zmid__$f"; done
bash "$R/src/gates_parallel.sh" "$R" "$F" "$PY" 5 $SPECS
touch "$R/work/FROZEN6"
echo "DS2_BATCH_DONE"
