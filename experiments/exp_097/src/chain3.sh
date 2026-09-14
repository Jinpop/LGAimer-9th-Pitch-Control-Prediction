#!/bin/bash
# exp_097 stageB: deep-arm learning-amount grid on the FIXED (float64-standardised)
# OOT input.  Waits for the stageA package runs to finish so the two do not
# contend for CPU.  Three lanes, cuda:0 only (cuda:1 carries exp_096 laneH).
B=/home/<user>/SungjinPark/exp097
PY=/home/<user>/SungjinPark/venv/bin/python
until grep -q CHAIN2_DONE $B/logs/chain2.log 2>/dev/null; do sleep 20; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---- reproduction gate: same code path as exp_090 on the SAME (defective) X --
echo "=== REPRO GATE $(date -Is) ==="
STEPS090=1632 taskset -c 32 $PY $B/src/train097.py oot23 42 cuda:0 ftt 0.0 \
  adamw0.01 K1632_oot23_s42 > $B/logs/repro_K1632_s42.log 2>&1
echo "repro rc=$?"

mklane () {           # $1 = lane index (0,1,2)  $2 = cpu
  for STEP in 544 720 816 1088 1360 1632 2176 2720 3264 4080; do
    i=0
    for S in 42 777 2024 101 7; do
      if [ $(( i % 3 )) -eq $1 ]; then
        T="F$STEP"_f64o_s$S
        if [ ! -f "$B/out/preds/z_$T.npy" ]; then
          STEPS090=$STEP taskset -c $2 $PY $B/src/train097.py f64o $S cuda:0 \
            ftt 0.0 adamw0.01 $T >> $B/logs/laneB$1.log 2>&1 \
            || echo "FAIL $T" >> $B/logs/laneB$1.log
        fi
      fi
      i=$((i+1))
    done
  done
  echo "LANE_B$1_DONE" >> $B/logs/laneB$1.log
}
echo "=== GRID START $(date -Is) ==="
mklane 0 32 &
mklane 1 33 &
mklane 2 34 &
wait
echo "=== GRID END $(date -Is) ==="
ls $B/out/preds/ | grep -c f64o
echo CHAIN3_DONE
