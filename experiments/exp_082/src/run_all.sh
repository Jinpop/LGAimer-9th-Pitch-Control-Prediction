#!/usr/bin/env bash
# exp_082 실행 — 실행당 6스레드 고정, 동시 6. 산출물 보존(rm 없음, exp082 아래만).
set -uo pipefail
H=/home/<user>/SungjinPark
OUT=$H/exp082
FR=$H/build_BT/frames
PY=$H/venv/bin/python
T=6
mkdir -p "$OUT/runs" "$OUT/logs" "$OUT/work"

full() {   # <pkg> <frame> <tag>
  local pkg=$1 frame=$2 tag=$3
  local d=$OUT/work/$tag
  rm -rf "$d"
  cp -r "$OUT/$pkg" "$d"
  mkdir -p "$d/data"
  cp "$FR/$frame/test.csv" "$FR/$frame/sample_submission.csv" "$d/data/"
  local t0=$(date +%s) rc=0
  ( cd "$d" && OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
      NUMEXPR_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T "$PY" script.py ) \
      > "$OUT/logs/$tag.log" 2>&1 || rc=$?
  local t1=$(date +%s)
  [ -f "$d/output/submission.csv" ] && cp "$d/output/submission.csv" "$OUT/runs/$tag.csv"
  [ -f "$d/mate/output/failtype.csv" ] && cp "$d/mate/output/failtype.csv" "$OUT/runs/ft_$tag.csv"
  echo "[$(date +%H:%M:%S)] $tag rc=$rc $((t1-t0))s"
}

matecal() {
  local d=$OUT/work/mate_cal
  local t0=$(date +%s) rc=0
  ( cd "$d" && OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
      NUMEXPR_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T "$PY" script.py ) \
      > "$OUT/logs/matecal.log" 2>&1 || rc=$?
  local t1=$(date +%s)
  [ -f "$d/output/submission.csv" ] && cp "$d/output/submission.csv" "$OUT/runs/matecal_partner.csv"
  [ -f "$d/output/failtype.csv" ] && cp "$d/output/failtype.csv" "$OUT/runs/ft_matecal.csv"
  echo "[$(date +%H:%M:%S)] matecal rc=$rc $((t1-t0))s"
}

echo "=== exp082 실행 시작 $(date +%F' '%H:%M:%S) ==="
matecal &
full pkg_bonly cov   bonly__cov   &
full pkg_bonly band  bonly__band  &
full pkg_bonly real5 bonly__real5 &
full pkg_ident cov   ident__cov   &
full pkg_BOref cov   boref__cov   &
wait
echo "=== EXP082_RUNS_DONE $(date +%F' '%H:%M:%S) ==="
