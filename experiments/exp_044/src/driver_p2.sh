#!/usr/bin/env bash
# exp_044 PHASE 2 — the stacker.
#
# The stacker's inputs are the forward-chaining OOF probabilities of the NEW
# cell models, so nothing here can start before phase 1 has picked all four.
#
#   step A  a real stage-1 run (unit 2023, seed 42) with the four new cell
#           settings and the exp_042 winner parked in the stacker slot, purely
#           to produce those OOF columns (--save-stack).  Its stacker number is
#           never promoted.
#   step B  the stacker search on those columns.  WIDE space only.
#   step C  a second real stage-1 run with the final map, so the treatment arm's
#           stage-1 iteration counts come out of exactly the same code path as
#           the control arm's did in exp_042 — no shortcut, no second detector.
#           Its cell argmins and its stack npz sha must repeat step A's; that is
#           checked in aggregate044.py against the two files on disk.
#
# step A and C are single units at 6 threads.  The remaining 12 vCPU are idle
# during them; that is deliberate — running a promotion unit alongside would
# change the contention those units were measured under.
set -uo pipefail
E=~/SungjinPark/experiments/exp_044
PY=~/SungjinPark/venv/bin/python
L="$E/logs/driver_p2.log"
mkdir -p "$E/logs" "$E/res" "$E/maps"
CAP=3300

step () {  # step <name> <cmd...>
  local n=$1; shift
  echo "[$(date -u +%FT%TZ)] start $n" >> "$L"
  "$@" > "$E/logs/$n.log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] end   $n rc=$rc" >> "$L"
  if [ "$rc" -ne 0 ]; then echo "[$(date -u +%FT%TZ)] ABORT at $n" >> "$L"; exit "$rc"; fi
}

# ---- A ---------------------------------------------------------------------
step mk_map_cellsA "$PY" "$E/src/mk_map.py" --which cellsA
step stage1_pcA "$PY" "$E/src/b3_cat_pc.py" --unit 2023 --seed 42 --stage 1 \
    --cap "$CAP" --params-map-json "$E/maps/map_pcA.json" --tag pcA --save-stack
# the cap gate reads the file stage 1 wrote, not a value typed in this script
grep -q '"any_argmin_at_cap": false' "$E/stage1_pcA.json" || {
  echo "[$(date -u +%FT%TZ)] ABORT: stage1_pcA hit the cap" >> "$L"; exit 3; }

# ---- B ---------------------------------------------------------------------
step hpo_stacker "$PY" "$E/src/hpo044.py" --mode stacker \
    --workers 9 --threads 2 --n-trials 55 --n-confirm 15 \
    --cap "$CAP" --recap 6600 --od-wait 250 \
    --stack-npz "$E/pred/stack_C5B3_pcA_u2023_seed42_st1.npz" \
    --stage1-json "$E/stage1_pcA.json"

# ---- C ---------------------------------------------------------------------
step mk_map_final "$PY" "$E/src/mk_map.py" --which final
step stage1_pc "$PY" "$E/src/b3_cat_pc.py" --unit 2023 --seed 42 --stage 1 \
    --cap "$CAP" --params-map-json "$E/maps/map_pc.json" --tag pc --save-stack
grep -q '"any_argmin_at_cap": false' "$E/stage1_pc.json" || {
  echo "[$(date -u +%FT%TZ)] ABORT: stage1_pc hit the cap" >> "$L"; exit 4; }

echo "[$(date -u +%FT%TZ)] PHASE 2 DONE" >> "$L"
