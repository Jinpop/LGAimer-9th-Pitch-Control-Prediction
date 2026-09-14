#!/usr/bin/env bash
# exp_079 stage3 — 추론 시간.  build_BO/run_BO.sh 의 run_solo 와 같은 조건
# (cov 프레임 = 대회 크기 test.csv, OMP/OPENBLAS/MKL = 6, 단독 실행).
#
# 사용:  stage3_timing.sh <TARGET_TREES>
#
# 🔴 어떤 원본도 지우거나 덮어쓰지 않는다.  pkg 는 rsync 로 새 위치에 복사하고
#    data/output 만 제외한다.  BO 기준선도 이 세션에서 다시 잰다(S13 — 세션이
#    다르면 조건이 다르다).
set -uo pipefail
T="$1"
H=/home/<user>/SungjinPark
E=$H/exp079
FRAMES=$H/build_AL/frames
PY=$H/venv/bin/python
mkdir -p "$E/probe" "$E/logs" "$E/out"

prep () {   # prep <dst_name>
  local d="$E/probe/$1"
  if [ ! -d "$d" ]; then
    rsync -a --exclude 'data' --exclude 'output' "$H/build_BO/pkg_BO/" "$d/"
    mkdir -p "$d/data" "$d/output"
    cp "$FRAMES/cov/test.csv" "$FRAMES/cov/sample_submission.csv" "$d/data/"
  fi
  echo "$d"
}

solo () {   # solo <dir> <tag>
  local d="$1" tag="$2" t0 e rc=0
  t0=$(date +%s)
  ( cd "$d" && OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      "$PY" script.py ) > "$E/logs/solo_$tag.log" 2>&1 || rc=$?
  e=$(( $(date +%s) - t0 ))
  if [ -f "$d/output/submission.csv" ]; then
    cp "$d/output/submission.csv" "$E/out/solo_$tag.csv"
    echo "$tag rc=$rc ${e}초 sha=$(sha256sum "$E/out/solo_$tag.csv" | cut -d' ' -f1)"
  else
    echo "$tag rc=$rc 출력없음 🔴"
  fi
  echo "$e" > "$E/out/.t_$tag"
}

echo "=== stage3 추론 시간 · 목표 ${T}그루 · $(date +%F' '%T) ==="

BASE=$(prep base354)
echo "--- 기준선: BO 원본(v6-Cat 354그루) 단독 ---"
solo "$BASE" "base354"

EXT=$(prep "ext${T}")
for s in 42 777 2024 101; do
  cp "$E/models/depext3_s${s}_T${T}_t6.cbm" "$EXT/mate/model/cat_${s}.cbm"
done
echo "--- 연장판: v6-Cat ${T}그루 단독 ---"
solo "$EXT" "ext${T}"

TB=$(cat "$E/out/.t_base354"); TE=$(cat "$E/out/.t_ext${T}")
echo
echo "기준선(354)      ${TB}초  → DACON 환산 $((TB*100/160))초"
echo "연장판(${T})     ${TE}초  → DACON 환산 $((TE*100/160))초"
echo "증가분           $((TE-TB))초  → DACON 환산 $(((TE-TB)*100/160))초"
echo "예산 600초 대비  $((TE*100/160))초 = $((TE*100/160*100/600))%"
echo "=== 종료 $(date +%F' '%T) ==="
