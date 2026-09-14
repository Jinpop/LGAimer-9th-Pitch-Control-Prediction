#!/bin/bash
# exp_051 stage 1 launcher — 3 seeds, 3-wide (효율 규약: 동시 3, fit당 threads 6)
set -euo pipefail
cd ~/SungjinPark/experiments/exp_051
mkdir -p logs oof pred curves res
PY=~/SungjinPark/venv/bin/python
for S in 42 7 2024; do
  setsid nohup "$PY" src/train51.py --unit 2024 --seed "$S" --stage 5 \
    --cap 1400 --rounds 655,1000,1400 \
    --iter-src "candidate AH local_reference_AH.json per_seed.$S.run.cell_iter_used, read in-script" \
    </dev/null > "logs/s1_seed$S.log" 2>&1 &
  echo "launched seed $S pid $!"
done
