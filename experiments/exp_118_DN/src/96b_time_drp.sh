#!/usr/bin/env bash
# G6 — DRP 단독 창 추론 시간.  경합 없이 1회 (규약 7 예외 · S120·S226).
# PAR 는 같은 조건에서 이미 쟀다 (178s, work/time.json).
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames/cov
PY=/home/<user>/SungjinPark/venv/bin/python
echo "다른 script.py 실행 수(자기 제외 검증): $(pgrep -cf '[s]cript.py')"
d=$R/work/time_DRP; rm -rf "$d"; cp -r "$R/pkg/DRP_SHIP" "$d"
rm -rf "$d/data" "$d/output" "$d/mate/data" "$d/mate/output"
rm -f  "$d/_module_origins_root.json" "$d/mate/_module_origins.json"
mkdir -p "$d/data"; cp "$F/test.csv" "$F/sample_submission.csv" "$d/data/"
t0=$(date +%s)
( cd "$d" && OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    "$PY" script.py ) > "$R/logs/time_DRP.log" 2>&1
rc=$?; t1=$(date +%s)
s=$(sha256sum "$d/output/submission.csv" | cut -d' ' -f1)
echo "DRP rc=$rc $((t1-t0))s sha=$s"
$PY - "$((t1-t0))" "$s" "$rc" <<'PY'
import json, sys
H="/home/<user>/SungjinPark/build_DN"
o=json.load(open(f"{H}/work/time.json",encoding="utf-8"))
o["DRP"]={"rc":int(sys.argv[3]),"seconds":int(sys.argv[1]),"out_sha256":sys.argv[2]}
o["delta_seconds_DRP_minus_PAR"]=o["DRP"]["seconds"]-o["PAR"]["seconds"]
json.dump(o,open(f"{H}/work/time.json","w"),ensure_ascii=False,indent=1)
print("time.json += DRP", o["DRP"]["seconds"], "s (PAR", o["PAR"]["seconds"], "s)")
PY
rm -rf "$d"
