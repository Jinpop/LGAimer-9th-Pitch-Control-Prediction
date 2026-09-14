#!/usr/bin/env bash
# exp_097 stageA — 패키지 1회 실행. 인자: <pkg경로> <frame경로> <태그> [threads]
set -euo pipefail
ROOT=$HOME/SungjinPark/exp097
PY=$HOME/SungjinPark/venv/bin/python
PKG=$1; FRAME=$2; TAG=$3; TH=${4:-4}
D=$ROOT/work/run_$TAG
rm -rf "$D"; mkdir -p "$ROOT/runs" "$ROOT/logs"
cp -r "$PKG" "$D"
rm -rf "$D/data" "$D/output" "$D/mate/data" "$D/mate/output" \
       "$D/__pycache__" "$D/mate/__pycache__"
mkdir -p "$D/data"
cp "$FRAME/test.csv" "$FRAME/sample_submission.csv" "$D/data/"
export PT_OUT="$ROOT/out/ledger_$TAG"
rm -rf "$PT_OUT"
rc=0
( cd "$D" && OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH \
    NUMEXPR_NUM_THREADS=$TH "$PY" script.py ) > "$ROOT/logs/$TAG.log" 2>&1 || rc=$?
if [ -f "$D/output/submission.csv" ]; then
  cp "$D/output/submission.csv" "$ROOT/runs/$TAG.csv"
fi
sha=$(sha256sum "$ROOT/runs/$TAG.csv" 2>/dev/null | cut -d' ' -f1 || echo "-")
echo "TAG=$TAG rc=$rc sha=$sha"
echo -e "$TAG\t$rc\t$sha" >> "$ROOT/runs/manifest.tsv"
