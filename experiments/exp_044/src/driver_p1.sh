#!/usr/bin/env bash
# exp_044 PHASE 1 — the four cell searches, one after another.
#
# 9 workers x 2 threads = 18 of the container's 20 vCPU.  Two are left for
# candidate Z's finishing work, which is pinned to them.  (exp_042 used 10x2;
# the search-phase thread exception applies here, NOT to the promotion runs.)
#
# is_wayoff_true runs FIRST on purpose: it is the cell exp_042 searched, so the
# first cell's wall time is directly comparable with exp_042's 224 min and the
# ETA can be re-based on a measurement instead of a projection.
set -uo pipefail
E=~/SungjinPark/experiments/exp_044
PY=~/SungjinPark/venv/bin/python
mkdir -p "$E/logs" "$E/res"
L="$E/logs/driver_p1.log"

for c in is_wayoff_true is_mid_only is_rev_only is_both; do
  if [ -f "$E/res/best_$c.json" ]; then
    echo "[$(date -u +%FT%TZ)] skip $c (best_$c.json already present)" >> "$L"
    continue
  fi
  echo "[$(date -u +%FT%TZ)] start $c" >> "$L"
  "$PY" "$E/src/hpo044.py" --mode cell --cell "$c" \
      --workers 9 --threads 2 --n-trials 55 --n-confirm 15 \
      --cap 3300 --recap 6600 --od-wait 250 \
      > "$E/logs/hpo_$c.log" 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] end   $c rc=$rc" >> "$L"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -u +%FT%TZ)] ABORT: $c failed" >> "$L"
    exit "$rc"
  fi
done
echo "[$(date -u +%FT%TZ)] PHASE 1 DONE" >> "$L"
