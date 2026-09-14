#!/bin/bash
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
export ATT_D=192 ATT_BLK=3 ATT_HEAD=8 ATT_MULT=2 MICRO=2048
for S in 42 777 2024 101 7; do
  $PY src/train096.py attn $S cuda:0 attn_oot23_s$S >> logs/laneA.log 2>&1
done
echo "LANE_A_DONE" >> logs/laneA.log
