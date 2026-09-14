#!/usr/bin/env bash
# exp_087 — the remaining F seeds, three at a time (20-core quota / 6 threads).
# Refuses to start if wave 1's G1 failed.
set -uo pipefail
H=/home/<user>/SungjinPark
PY=$H/venv/bin/python
TR=$H/submissions/S_feat63_seed3/build/train_final_c5_feat63.py
ROOT=$H/exp087
PROG=$ROOT/logs/fits_progress.jsonl
[ -f "$ROOT/STOP_G1_FAILED" ] && { echo "STOP_G1_FAILED present — refusing"; exit 1; }
[ -f "$ROOT/res/g1_repro.json" ] || { echo "no G1 result — refusing"; exit 1; }
grep -q '"ALL_PASS": true' "$ROOT/res/g1_repro.json" || { echo "G1 not ALL_PASS — refusing"; exit 1; }
log() { printf '{"t":"%s","ev":%s}\n' "$(date -u +%FT%TZ)" "$1" >> "$PROG"; }
fit() {
  local s="$1" out="$ROOT/seeds/seed_$1"
  mkdir -p "$out/build"
  if [ -f "$out/build/train_report.json" ]; then log "{\"seed\":$s,\"skip\":true}"; return 0; fi
  log "{\"seed\":$s,\"start\":true}"
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
    "$PY" "$TR" --out "$out" --seed "$s" --cell-policy stage1_scaled \
    > "$ROOT/logs/fit_seed_$s.log" 2>&1
  local rc=$?
  log "{\"seed\":$s,\"done\":true,\"rc\":$rc,\"cbm\":$(find "$out/model" -name '*.cbm' 2>/dev/null | wc -l)}"
  echo "[$(date +%H:%M:%S)] seed $s rc=$rc cbm=$(find "$out/model" -name '*.cbm' 2>/dev/null | wc -l)"
}
n=0
for s in "$@"; do
  fit "$s" &
  n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
done
wait
echo "총 cbm: $(find $ROOT/seeds -name '*.cbm' | wc -l)"
echo "EXP087_REST_DONE $(date +%F' '%H:%M:%S)"
