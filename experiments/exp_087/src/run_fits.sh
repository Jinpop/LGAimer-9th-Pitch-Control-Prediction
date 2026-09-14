#!/usr/bin/env bash
# exp_087 — F-branch seed expansion 3 -> 18.
#
# The trainer is candidate S's, IN PLACE and READ-ONLY:
#   ~/SungjinPark/submissions/S_feat63_seed3/build/train_final_c5_feat63.py
# It writes only <out>/model/*.cbm and <out>/build/train_report.json, so every
# artefact of this run lands under ~/SungjinPark/exp087/.  Nothing outside is
# touched.
#
# WAVE 1 carries the G1 reproduction (seed 42) plus the first two new seeds.
# Concurrency is 3 and the container CPU quota is 20 cores (cfs 2000000/100000)
# at 6 threads per fit, so a solo G1 wave would leave 12 of the 18 usable cores
# idle for 90 minutes and buy nothing: the gate's purpose is to stop the run
# before the OTHER thirteen seeds are fitted, and it still does exactly that.
# If G1 fails the driver writes STOP and exits with the two extra seeds thrown
# away.
#
# `-e` is deliberately absent: one seed failing must not kill the others.
set -uo pipefail
H=/home/<user>/SungjinPark
PY=$H/venv/bin/python
TR=$H/submissions/S_feat63_seed3/build/train_final_c5_feat63.py
ROOT=$H/exp087
PROG=$ROOT/logs/fits_progress.jsonl
mkdir -p "$ROOT/logs" "$ROOT/seeds" "$ROOT/g1" "$ROOT/res"

log() { printf '{"t":"%s","ev":%s}\n' "$(date -u +%FT%TZ)" "$1" >> "$PROG"; }

fit() {          # $1 = seed, $2 = out dir
  local s="$1" out="$2"
  mkdir -p "$out/build"
  if [ -f "$out/build/train_report.json" ]; then
    log "{\"seed\":$s,\"skip\":\"train_report.json present\"}"; return 0
  fi
  log "{\"seed\":$s,\"out\":\"$out\",\"start\":true}"
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  NUMEXPR_NUM_THREADS=6 \
    "$PY" "$TR" --out "$out" --seed "$s" --cell-policy stage1_scaled \
    > "$ROOT/logs/fit_seed_$s.log" 2>&1
  local rc=$?
  log "{\"seed\":$s,\"done\":true,\"rc\":$rc,\"cbm\":$(find "$out/model" -name '*.cbm' 2>/dev/null | wc -l)}"
  echo "[$(date +%H:%M:%S)] seed $s rc=$rc cbm=$(find "$out/model" -name '*.cbm' 2>/dev/null | wc -l)"
  return $rc
}

# ---------------------------------------------------------------- wave 1 (G1)
log '{"driver":"exp087","wave":1,"members":["42_G1repro",1,3],"threads_per_fit":6}'
fit 42 "$ROOT/g1/seed_42"      & P1=$!
fit 1  "$ROOT/seeds/seed_1"    & P2=$!
fit 3  "$ROOT/seeds/seed_3"    & P3=$!
wait $P1; R1=$?; wait $P2; wait $P3
echo "[$(date +%H:%M:%S)] wave 1 done (G1 fit rc=$R1)"

"$PY" "$ROOT/src/g1_gate.py" \
    --parent-zip "$H/build_CA2/submit_CA2_nact.zip" \
    --branch F --seed 42 \
    --refit-model-dir "$ROOT/g1/seed_42/model" \
    --out "$ROOT/res/g1_repro.json" > "$ROOT/logs/g1_gate.log" 2>&1
G1=$?
log "{\"G1\":$G1}"
if [ $G1 -ne 0 ]; then
  touch "$ROOT/STOP_G1_FAILED"
  echo "G1 FAILED — stopping.  The remaining thirteen seeds were NOT fitted."
  echo "EXP087_FITS_HALTED_G1 $(date +%F' '%H:%M:%S)"
  exit 1
fi
echo "[$(date +%H:%M:%S)] G1 PASS — continuing with the remaining seeds"

# ----------------------------------------------------------- waves 2..5 (13)
n=0
for s in 13 17 77 101 123 555 777 999 2019 2025 8191 31337 65537; do
  fit "$s" "$ROOT/seeds/seed_$s" &
  n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
done
wait
echo "총 cbm: $(find $ROOT/seeds -name '*.cbm' | wc -l)"
echo "EXP087_FITS_DONE $(date +%F' '%H:%M:%S)"
