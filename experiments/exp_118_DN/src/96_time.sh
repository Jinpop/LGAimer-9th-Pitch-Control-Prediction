#!/usr/bin/env bash
# G11 — 단독 창 추론 시간.  경합 없이 순차로 잰다 (규약 7 예외 · S120·S226).
set -uo pipefail
R=/home/<user>/SungjinPark/build_DN
F=/home/<user>/SungjinPark/build_CA/frames/cov
PY=/home/<user>/SungjinPark/venv/bin/python
echo "다른 script.py 실행 수(자기 제외 검증): $(pgrep -cf '[s]cript.py')"
: > "$R/work/time_raw.txt"
for tag in DS2 PAR DN DR DQ DNP_SHIP; do
  d=$R/work/time_$tag
  rm -rf "$d"                       # 이 스크립트가 직전에 만든 자기 임시 사본만 지운다
  cp -r "$R/pkg/$tag" "$d"
  rm -rf "$d/data" "$d/output" "$d/mate/data" "$d/mate/output"
  rm -f  "$d/_module_origins_root.json" "$d/mate/_module_origins.json"
  mkdir -p "$d/data"
  cp "$F/test.csv" "$F/sample_submission.csv" "$d/data/"
  t0=$(date +%s)
  ( cd "$d" && OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      "$PY" script.py ) > "$R/logs/time_$tag.log" 2>&1
  rc=$?; t1=$(date +%s)
  s=$(sha256sum "$d/output/submission.csv" | cut -d' ' -f1)
  echo "$tag rc=$rc $((t1-t0))s sha=$s"
  printf '%s\t%s\t%s\t%s\n' "$tag" "$rc" "$((t1-t0))" "$s" >> "$R/work/time_raw.txt"
done
$PY - <<'PY'
import json
H = "/home/<user>/SungjinPark/build_DN"
o = {"frame": "cov (245,789행)", "threads": 6,
     "🔴": ("단독 창에서 순차로 쟀다(S120 — 다른 세션이 붙으면 +13.5% 튄다). "
            "S226 — 이 규모에서 델타 자체는 측정 불가다. 예산 판단용 절대값만 쓴다.")}
for line in open(f"{H}/work/time_raw.txt"):
    t, rc, s, sha = line.rstrip("\n").split("\t")
    o[{"DNP_SHIP": "DNP", "DS2": "DS"}.get(t, t)] = {"rc": int(rc),
        "seconds": int(s), "out_sha256": sha}
o["delta_seconds_DN_minus_PAR"] = o["DN"]["seconds"] - o["PAR"]["seconds"]
o["delta_seconds_DNP_minus_PAR"] = o["DNP"]["seconds"] - o["PAR"]["seconds"]
o["delta_seconds_DQ_minus_PAR"] = o["DQ"]["seconds"] - o["PAR"]["seconds"]
o["delta_seconds_DR_minus_PAR"] = o["DR"]["seconds"] - o["PAR"]["seconds"]
o["delta_seconds_DS_minus_PAR"] = o["DS"]["seconds"] - o["PAR"]["seconds"]
json.dump(o, open(f"{H}/work/time.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=1))
PY
