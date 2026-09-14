#!/usr/bin/env bash
# exp_042 step 3 — promotion runs on fold_2024, three seeds, two arms.
#
# threads: 6 per fit, FIXED.  The search phase's thread exception does NOT
# apply here, because these times are compared with earlier measurements.
# At most three fits at a time = 18 of the container's 20 CPU.
#
# cap 3300 for the hpo arm's stage 1.  The winner's learning_rate 0.015743 is
# about half the current 0.03, so the argmin walks roughly twice as far; the
# search cell alone stopped at 1072 and the other cells / the stacker can go
# further.  Whether the cap was actually enough is read back out of the stage-1
# json (`any_argmin_at_cap`), not assumed.
#
# schedule (max 3 concurrent):
#   chain A: ctrl seed 42 -> ctrl seed 2024
#   chain B: ctrl seed 7
#   chain C: stage 1 for the hpo arm (unit 2023, seed 42, cap 3300)
#   then   : the three hpo fold_2024 units, 3-wide
#
# control arm = EMPTY parameter override with exp_030's stage1.json, i.e. the
# production pipeline exactly; aggregate042.py sha-compares its predictions
# against exp_030's stored ones (identity anchor).
set -euo pipefail

E=~/SungjinPark/experiments/exp_042
PY=~/SungjinPark/venv/bin/python
S=$E/src/b3_cat_hpo.py
CTRL_BASE=~/SungjinPark/experiments/exp_030/stage1.json
HPO_BASE=$E/stage1_hpo.json
PJ=$E/best_params.json
CAP=3300
mkdir -p "$E/logs"

run () {  # run <logname> <args...>
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name" >> "$E/logs/driver.log"
  "$PY" "$S" "$@" > "$E/logs/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] end   $name" >> "$E/logs/driver.log"
}

echo "[$(date +%H:%M:%S)] PHASE 1 (ctrl x3 + hpo stage1, max 3 concurrent, cap $CAP)" \
  >> "$E/logs/driver.log"

( run ctrl_s42 --unit 2024 --seed 42 --stage 2 --base-json "$CTRL_BASE" \
      --params-json '{}' --tag ctrl
  run ctrl_s2024 --unit 2024 --seed 2024 --stage 2 --base-json "$CTRL_BASE" \
      --params-json '{}' --tag ctrl ) &
( run ctrl_s7 --unit 2024 --seed 7 --stage 2 --base-json "$CTRL_BASE" \
      --params-json '{}' --tag ctrl ) &
( run stage1_hpo --unit 2023 --seed 42 --stage 1 --cap "$CAP" \
      --params-json "$PJ" --tag hpo ) &
wait

# the cap gate reads the file stage 1 wrote, not a value typed in this script
grep -q '"any_argmin_at_cap": false' "$HPO_BASE"

echo "[$(date +%H:%M:%S)] PHASE 2 (hpo fold_2024 x3, 3-wide)" >> "$E/logs/driver.log"
run hpo_s42   --unit 2024 --seed 42   --stage 2 --base-json "$HPO_BASE" \
    --params-json "$PJ" --tag hpo &
run hpo_s7    --unit 2024 --seed 7    --stage 2 --base-json "$HPO_BASE" \
    --params-json "$PJ" --tag hpo &
run hpo_s2024 --unit 2024 --seed 2024 --stage 2 --base-json "$HPO_BASE" \
    --params-json "$PJ" --tag hpo &
wait

echo "[$(date +%H:%M:%S)] ALL UNITS DONE" >> "$E/logs/driver.log"
"$PY" "$E/src/aggregate042.py" > "$E/logs/aggregate.log" 2>&1
echo "[$(date +%H:%M:%S)] AGGREGATED" >> "$E/logs/driver.log"
