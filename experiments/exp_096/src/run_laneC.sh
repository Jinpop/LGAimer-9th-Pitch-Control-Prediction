#!/bin/bash
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
for S in 42 777 2024 101 7; do
  SEQ_VARIANT=A $PY src/train096.py seq $S cuda:1 seqA_oot23_s$S >> logs/laneC.log 2>&1
done
for S in 42 777 2024 101 7; do
  SEQ_VARIANT=B $PY src/train096.py seq $S cuda:1 seqB_oot23_s$S >> logs/laneC.log 2>&1
done
for S in 42 777 2024 101 7; do
  SEQ_VARIANT=B SEQ_NULL=1 $PY src/train096.py seq $S cuda:1 seqN_oot23_s$S >> logs/laneC.log 2>&1
done
echo "LANE_C_DONE" >> logs/laneC.log
