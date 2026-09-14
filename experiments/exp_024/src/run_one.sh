#!/usr/bin/env bash
# exp_024 stage-1 worker: one (unit, seed) B3 run.  Never aborts the pool.
u=$1; s=$2
cd ~/SungjinPark/experiments/exp_024
mkdir -p logs blocks pred
LOG=logs/b3_${u}_seed${s}.log
echo "$(date +%H:%M:%S) START unit=$u seed=$s" >> logs/stage1_status.txt
~/SungjinPark/venv/bin/python src/b3_block.py "$u" "$s" > "$LOG" 2>&1
rc=$?
echo "$(date +%H:%M:%S) END   unit=$u seed=$s rc=$rc" >> logs/stage1_status.txt
exit 0
