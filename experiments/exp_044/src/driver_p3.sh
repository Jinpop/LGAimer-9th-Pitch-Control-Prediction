#!/usr/bin/env bash
# exp_044 PHASE 3 — 18 fold_2024 units: 2 arms x 9 pre-registered seeds.
#
# threads 6 per fit, at most 3 units at a time = 18 of the 20 vCPU.  The search
# phase's thread exception does NOT apply here: these times and these numbers
# are compared with exp_042's, which were measured at 6 threads 3-wide.
#
# All 18 are computed fresh by this code, including the three control seeds
# exp_042 already ran.  Those three are the identity anchor: their prediction
# sha256 must equal exp_042's stored ones, which proves the per-model plumbing
# changed nothing on a path where both arms sit.  aggregate044.py does the
# comparison against the files on disk.
#
# The job list is interleaved ctrl/pc so a fault in either arm shows up early.
set -uo pipefail
E=~/SungjinPark/experiments/exp_044
mkdir -p "$E/logs"
J="$E/logs/p3_jobs.txt"
: > "$J"
for s in 42 7 2024 1 13 101 777 2025 31337; do
  echo "ctrl $s" >> "$J"
  echo "pc $s"   >> "$J"
done
echo "[$(date -u +%FT%TZ)] PHASE 3 START, $(wc -l < "$J") units, 3-wide" \
  >> "$E/logs/driver_p3.log"
xargs -a "$J" -P 3 -n 2 bash "$E/src/run_unit.sh"
rc=$?
echo "[$(date -u +%FT%TZ)] PHASE 3 DONE rc=$rc" >> "$E/logs/driver_p3.log"
exit "$rc"
