#!/bin/bash
# exp_102 axisB — re-read exp_091/092 top points on the FIXED f64 OOT input.
# Harness = exp_092 train092.py verbatim (only base/work dirs changed).
B=/home/<user>/SungjinPark/exp102
PY=/home/<user>/SungjinPark/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
until grep -q CHAIN_A_DONE $B/logs/runA.log 2>/dev/null; do sleep 10; done

echo "=== B REPRO GATE (default path must equal exp097 F1632_f64o_s42) $(date -Is) ==="
STEPS092=1632 NT092=1 taskset -c 20 $PY $B/src/train102b.py f64o 42 cuda:0 ftt 0.0 \
  adamw0.01 BG1632_f64o_s42 > $B/logs/reproB.log 2>&1
echo "reproB rc=$?"

M16="1,6,12,17,20,51,52,56,61,62,71,75,78,82,87,89"
JOBS=$B/logs/jobsB.txt
: > $JOBS
for S in 42 777 2024 101 7; do
  echo "Bdr02  $S DROP=0.2"                >> $JOBS
  echo "Bm16   $S MASK092=$M16"            >> $JOBS
  echo "Bls005 $S LOSS092=ls0.05"          >> $JOBS
  echo "Bw1dm1 $S HID092=256,128"          >> $JOBS
  echo "Bw2dm1 $S HID092=512,256"          >> $JOBS
done
echo "=== B GRID START $(wc -l < $JOBS) jobs $(date -Is) ==="

laneB () {   # $1 lane idx  $2 cpu  $3 gpu
  i=0
  while read NAME S EXTRA; do
    if [ $(( i % 4 )) -eq $1 ]; then
      T="${NAME}_f64o_s${S}"
      if [ ! -f "$B/out/preds/z_$T.npy" ]; then
        DROPV=0.0
        case "$EXTRA" in DROP=*) DROPV="${EXTRA#DROP=}"; EXTRA="" ;; esac
        env $EXTRA STEPS092=1632 NT092=1 taskset -c $2 $PY $B/src/train102b.py \
          f64o $S cuda:$3 ftt $DROPV adamw0.01 $T >> $B/logs/laneB$1.log 2>&1 \
          || echo "FAIL $T" >> $B/logs/laneB$1.log
      fi
    fi
    i=$((i+1))
  done < $JOBS
  echo "LANE_B$1_DONE" >> $B/logs/laneB$1.log
}
laneB 0 20 0 & laneB 1 21 0 & laneB 2 22 1 & laneB 3 23 1 &
wait
echo "=== B GRID END $(date -Is) ==="
echo CHAIN_B_DONE
