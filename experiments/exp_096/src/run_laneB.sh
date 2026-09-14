#!/bin/bash
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
for S in 42 777 2024 101 7; do
  $PY src/train096.py cos    $S cuda:1 cos_oot23_s$S    >> logs/laneB.log 2>&1
done
for S in 42 777 2024 101 7; do
  $PY src/train096.py rank   $S cuda:1 rank_oot23_s$S   >> logs/laneB.log 2>&1
done
for S in 42 777 2024 101 7; do
  $PY src/train096.py mlpP35 $S cuda:1 ctrl35_oot23_s$S >> logs/laneB.log 2>&1
done
echo "LANE_B_DONE" >> logs/laneB.log
