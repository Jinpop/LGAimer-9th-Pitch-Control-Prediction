#!/bin/bash
# exp_102 axisA fits on the FIXED f64 OOT input. Same recipe as exp_097 chain3.sh.
B=/home/<user>/SungjinPark/exp102
PY=/home/<user>/SungjinPark/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== REPRO GATE $(date -Is) ==="
STEPS090=1632 taskset -c 20 $PY $B/src/train102.py f64o 42 cuda:0 ftt 0.0 \
  adamw0.01 R1632_f64o_s42 > $B/logs/repro.log 2>&1
echo "repro rc=$?"

# job list: "STEP SEED"
JOBS=$B/logs/jobsA.txt
: > $JOBS
for S in 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89; do
  echo "1632 $S" >> $JOBS
done
for STEP in 1568 1696; do
  for S in 42 777 2024 101 7; do echo "$STEP $S" >> $JOBS; done
done
N=$(wc -l < $JOBS)
echo "=== GRID START $N jobs $(date -Is) ==="

lane () {   # $1 lane idx (0..3)  $2 cpu  $3 gpu
  i=0
  while read STEP S; do
    if [ $(( i % 4 )) -eq $1 ]; then
      T="F${STEP}_f64o_s${S}"
      if [ ! -f "$B/out/preds/z_$T.npy" ]; then
        STEPS090=$STEP taskset -c $2 $PY $B/src/train102.py f64o $S cuda:$3 \
          ftt 0.0 adamw0.01 $T >> $B/logs/laneA$1.log 2>&1 \
          || echo "FAIL $T" >> $B/logs/laneA$1.log
      fi
    fi
    i=$((i+1))
  done < $JOBS
  echo "LANE_A$1_DONE" >> $B/logs/laneA$1.log
}
lane 0 20 0 &
lane 1 21 0 &
lane 2 22 1 &
lane 3 23 1 &
wait
echo "=== GRID END $(date -Is) ==="
ls $B/out/preds/ | wc -l
echo CHAIN_A_DONE
