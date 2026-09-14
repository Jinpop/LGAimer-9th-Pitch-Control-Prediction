#!/usr/bin/env bash
set -euo pipefail
D=~/SungjinPark/experiments/exp_036
CELLS='{"is_mid_only":577,"is_rev_only":1346,"is_both":449,"is_wayoff_true":897}'
SRC='exp_030 stage-2 fold_2024 CELL iterations, pinned verbatim; this run changes ONLY how far the stacker is trained'
mkdir -p "$D/logs" "$D/pred" "$D/curves" "$D/res"
cd "$D"
for S in 42 7 2024; do
  setsid nohup ~/SungjinPark/venv/bin/python src/stack_argmin.py \
    --unit 2024 --seed "$S" --stage 5 --cap 3000 --gate-k 481 \
    --cell-iter "$CELLS" --iter-src "$SRC" \
    </dev/null > "logs/f2024_s${S}_st5.log" 2>&1 &
  echo "launched seed $S pid $!"
  sleep 2
done
