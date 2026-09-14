#!/usr/bin/env bash
# exp_087 wave 1 — the G1 reproduction (seed 42) plus the two new seeds that
# every surviving scenario needs (n>=6 requires at least three new seeds).
#
# The trainer is candidate S's, IN PLACE and READ-ONLY:
#   ~/SungjinPark/submissions/S_feat63_seed3/build/train_final_c5_feat63.py
# It writes only <out>/model/*.cbm and <out>/build/train_report.json, so every
# artefact lands under ~/SungjinPark/exp087/.
#
# WHY G1 SHARES ITS WAVE.  The container CPU quota is 20 cores
# (cfs 2000000/100000) and each fit is pinned to 6 threads, so three fits fill
# it and a solo G1 wave would idle 12 cores for 90 minutes.  The gate's purpose
# is to stop the run before the OTHER THIRTEEN seeds are fitted, and it still
# does exactly that: on failure this script writes STOP_G1_FAILED and the
# follow-on script refuses to start.
set -uo pipefail
H=/home/<user>/SungjinPark
PY=$H/venv/bin/python
TR=$H/submissions/S_feat63_seed3/build/train_final_c5_feat63.py
ROOT=$H/exp087
PROG=$ROOT/logs/fits_progress.jsonl
mkdir -p "$ROOT/logs" "$ROOT/seeds" "$ROOT/g1" "$ROOT/res"
log() { printf '{"t":"%s","ev":%s}\n' "$(date -u +%FT%TZ)" "$1" >> "$PROG"; }

fit() {
  local s="$1" out="$2"
  mkdir -p "$out/build"
  if [ -f "$out/build/train_report.json" ]; then
    log "{\"seed\":$s,\"skip\":true}"; return 0; fi
  log "{\"seed\":$s,\"out\":\"$out\",\"start\":true}"
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
    "$PY" "$TR" --out "$out" --seed "$s" --cell-policy stage1_scaled \
    > "$ROOT/logs/fit_seed_$s.log" 2>&1
  local rc=$?
  log "{\"seed\":$s,\"done\":true,\"rc\":$rc,\"cbm\":$(find "$out/model" -name '*.cbm' 2>/dev/null | wc -l)}"
  echo "[$(date +%H:%M:%S)] seed $s rc=$rc cbm=$(find "$out/model" -name '*.cbm' 2>/dev/null | wc -l)"
}

log '{"driver":"exp087","wave":1,"members":["42_G1repro",1,3],"threads_per_fit":6,"concurrency":3}'
echo "[$(date +%H:%M:%S)] wave 1 start"
fit 42 "$ROOT/g1/seed_42"   & P1=$!
fit 1  "$ROOT/seeds/seed_1" & P2=$!
fit 3  "$ROOT/seeds/seed_3" & P3=$!
wait $P1; wait $P2; wait $P3
echo "[$(date +%H:%M:%S)] wave 1 fits done — running G1"

"$PY" "$ROOT/src/g1_gate.py" \
    --parent-zip "$H/build_CA2/submit_CA2_nact.zip" --branch F --seed 42 \
    --refit-model-dir "$ROOT/g1/seed_42/model" \
    --out "$ROOT/res/g1_repro.json" > "$ROOT/logs/g1_gate.log" 2>&1
G1=$?
log "{\"G1_rc\":$G1}"
if [ $G1 -ne 0 ]; then
  touch "$ROOT/STOP_G1_FAILED"
  echo "G1 FAILED — the remaining thirteen seeds were NOT fitted."
  echo "EXP087_WAVE1_HALTED_G1 $(date +%F' '%H:%M:%S)"; exit 1
fi
echo "G1 PASS"
echo "EXP087_WAVE1_DONE $(date +%F' '%H:%M:%S)"
