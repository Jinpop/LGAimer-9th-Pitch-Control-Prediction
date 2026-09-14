#!/bin/bash
# exp_097 stageA: run the deployed package once per dtype variant, pinned to 4 cores.
# Sequential on purpose -- the point is a clean per-variant delta, not throughput.
B=/home/<user>/SungjinPark/exp097
PY=/home/<user>/SungjinPark/venv/bin/python
CPUS=32-35
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for V in V0 VD V1 V2 V3 V4; do
  D=$B/runs/$V
  [ -d "$D" ] || continue
  echo "=== START $V $(date -Is) ==="
  ( cd "$D" && taskset -c $CPUS $PY script.py > "$B/logs/run_$V.log" 2>&1 )
  rc=$?
  echo "=== END $V rc=$rc $(date -Is) ==="
  if [ $rc -ne 0 ]; then echo "FAILED $V"; tail -25 "$B/logs/run_$V.log"; fi
  cp "$D/output/submission.csv" "$B/out/sub_$V.csv" 2>/dev/null || echo "no submission for $V"
done
echo CHAIN_DONE
