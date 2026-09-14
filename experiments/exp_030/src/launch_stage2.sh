#!/bin/bash
# exp_030 stage 2 launcher.  ONE WAVE = up to 3 concurrent units (efficiency
# rule #2: fits inside one experiment may run 3-wide, threads stay at 6).
# Usage:  bash launch_stage2.sh <wave>
#   wave 1 : unit 2024 x seeds 42 / 7 / 2024        (adoption test)
#   wave 2 : unit 2023 seed 7, unit 2022 seeds 42/7 (beta supply blocks)
set -euo pipefail
EXP=~/SungjinPark/experiments/exp_030
PY=/home/<user>/SungjinPark/venv/bin/python
BASE=$EXP/stage1.json
cd "$EXP"
mkdir -p logs
test -f "$BASE"

launch () {   # unit seed
  local u=$1 s=$2
  setsid nohup "$PY" src/b3_cat.py --unit "$u" --seed "$s" --stage 2 \
      --base-json "$BASE" </dev/null > "logs/st2_u${u}_s${s}.log" 2>&1 &
}

case "$1" in
  1) launch 2024 42; launch 2024 7; launch 2024 2024 ;;
  2) launch 2023 7;  launch 2022 42; launch 2022 7 ;;
  *) echo "unknown wave $1" >&2; exit 1 ;;
esac
sleep 6
ps -eo pid=,args= | awk '$2=="/home/<user>/SungjinPark/venv/bin/python" && $3=="src/b3_cat.py"'
