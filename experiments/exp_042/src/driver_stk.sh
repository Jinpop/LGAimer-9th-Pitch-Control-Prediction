#!/usr/bin/env bash
# exp_042 addendum — measure the fold_2024 stacker argmin UNCENSORED under the
# exp_042 winning settings, so candidate Z's deployed stacker count can be
# derived by candidate W's OWN procedure instead of transcribing W's 655.
#
# W's 655 came from exp_036 measuring the CONTROL settings' fold_2024 stacker
# argmin at cap 3000 on three seeds (493/472/541), taking the geometric mean
# (501.18192) and extrapolating log-log to the deployment row count.  The cap
# is kept at 3000 so the PROCEDURE is unchanged; only the settings differ.
#
# threads 6 fixed, three at a time = 18 of the container's 20 CPU.
# --save-stack persists the four OOF columns, which exp_042's promotion run did
# not: with them on disk a future stacker-only study skips the cell refits.
set -uo pipefail
E=~/SungjinPark/experiments/exp_042
PY=~/SungjinPark/venv/bin/python
S=$E/src/b3_cat_hpo.py
HPO_BASE=$E/stage1_hpo.json
PJ=$E/best_params.json
CAP=3000
mkdir -p "$E/logs"
run () {
  local s=$1
  if [ -f "$E/res/res_C5B3_stkmeas_u2024_seed${s}_st2.json" ]; then
    echo "[$(date +%H:%M:%S)] skip seed $s (result already present)" >> "$E/logs/driver_stk.log"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] start stkmeas seed $s" >> "$E/logs/driver_stk.log"
  "$PY" "$S" --unit 2024 --seed "$s" --stage 2 --base-json "$HPO_BASE" \
      --params-json "$PJ" --tag stkmeas --stack-cap "$CAP" --save-stack \
      > "$E/logs/stkmeas_s$s.log" 2>&1
  echo "[$(date +%H:%M:%S)] end   stkmeas seed $s rc=$?" >> "$E/logs/driver_stk.log"
}
echo "[$(date +%H:%M:%S)] STACKER RE-MEASUREMENT cap $CAP seeds 42 7 2024" >> "$E/logs/driver_stk.log"
pids=""
for s in 42 7 2024; do run "$s" & pids="$pids $!"; done
for p in $pids; do wait "$p"; done
echo "[$(date +%H:%M:%S)] STACKER RE-MEASUREMENT DONE" >> "$E/logs/driver_stk.log"
