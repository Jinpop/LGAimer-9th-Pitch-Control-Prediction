#!/usr/bin/env bash
# exp_044 — one fold_2024 promotion unit.  Called by driver_p3.sh through xargs.
# usage: run_unit.sh <arm: ctrl|pc> <seed>
set -uo pipefail
E=~/SungjinPark/experiments/exp_044
PY=~/SungjinPark/venv/bin/python
arm=$1
seed=$2
case "$arm" in
  ctrl) MAP="$E/maps/map_ctrl.json"
        BASE=~/SungjinPark/experiments/exp_042/stage1_hpo.json ;;
  pc)   MAP="$E/maps/map_pc.json"
        BASE="$E/stage1_pc.json" ;;
  *)    echo "unknown arm $arm" >&2; exit 2 ;;
esac
R="$E/res/res_C5B3_${arm}_u2024_seed${seed}_st2.json"
if [ -f "$R" ]; then
  echo "[$(date -u +%FT%TZ)] skip $arm seed $seed (result present)" >> "$E/logs/driver_p3.log"
  exit 0
fi
echo "[$(date -u +%FT%TZ)] start $arm seed $seed" >> "$E/logs/driver_p3.log"
"$PY" "$E/src/b3_cat_pc.py" --unit 2024 --seed "$seed" --stage 2 \
    --base-json "$BASE" --params-map-json "$MAP" --tag "$arm" \
    > "$E/logs/p3_${arm}_s${seed}.log" 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] end   $arm seed $seed rc=$rc" >> "$E/logs/driver_p3.log"
exit "$rc"
