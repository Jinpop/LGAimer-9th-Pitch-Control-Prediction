#!/bin/bash
set -uo pipefail
cd /home/<user>/SungjinPark/exp088
PY=/home/<user>/SungjinPark/venv/bin/python
C=/home/<user>/SungjinPark/exp088/work
run(){ OMP_NUM_THREADS=6 $PY src/train088.py F1_FM "$2" le2024 "$1" $C "$3" > logs/F1_FM_le2024_s$1.log 2>&1; }
# wave 1 — 동시 3
run 42 cuda:0 1 &
run 7 cuda:1 1 &
run 2024 cuda:0 1 &
wait
# wave 2 — 동시 2
run 101 cuda:1 2 &
run 777 cuda:0 1 &
wait
echo ALLDONE
