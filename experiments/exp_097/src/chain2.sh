#!/bin/bash
B=/home/<user>/SungjinPark/exp097
PY=/home/<user>/SungjinPark/venv/bin/python
until grep -q CHAIN_DONE $B/logs/chain.log 2>/dev/null; do sleep 20; done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo "=== START VD(retry) $(date -Is) ==="
( cd $B/runs/VD && taskset -c 32-35 $PY script.py > $B/logs/run_VD.log 2>&1 )
echo "=== END VD rc=$? $(date -Is) ==="
cp $B/runs/VD/output/submission.csv $B/out/sub_VD.csv 2>/dev/null || echo "no VD submission"
echo CHAIN2_DONE
