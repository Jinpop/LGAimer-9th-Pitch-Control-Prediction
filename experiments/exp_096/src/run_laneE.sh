#!/bin/bash
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
until grep -q LANE_D_DONE logs/laneD.log 2>/dev/null; do sleep 10; done
for I in std rank; do
  for S in 42 777 2024 101 7; do
    $PY src/foldb_run.py $I $S cuda:1 ${I}_le22_s$S >> logs/laneE.log 2>&1
  done
done
$PY src/foldb_eval.py > logs/foldb_eval.log 2>&1
echo "LANE_E_DONE" >> logs/laneE.log
