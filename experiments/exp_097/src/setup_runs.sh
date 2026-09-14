#!/bin/bash
set -euo pipefail
B=/home/<user>/SungjinPark/exp097
SRC=$B/pkg/CH
FR=/home/<user>/SungjinPark/build_BX/frames/cov
PY=/home/<user>/SungjinPark/venv/bin/python
for V in V0 V1 V2 V3 V4; do
  D=$B/runs/$V
  rm -rf "$D"; mkdir -p "$D"
  cp -r "$SRC"/. "$D"/
  mkdir -p "$D/data" "$D/output"
  cp "$FR/test.csv" "$D/data/test.csv"
  cp "$FR/sample_submission.csv" "$D/data/sample_submission.csv"
done
echo "V1"; $PY $B/src/patch.py $B/runs/V1 nn
echo "V2"; $PY $B/src/patch.py $B/runs/V2 deep
echo "V3"; $PY $B/src/patch.py $B/runs/V3 ins
echo "V4"; $PY $B/src/patch.py $B/runs/V4 nn deep ins
echo "=== EXP097 marker counts (V0 must be all 0) ==="
for V in V0 V1 V2 V3 V4; do
  n_nn=$(grep -c EXP097 $B/runs/$V/mate/nn_infer.py || true)
  n_dp=$(grep -c EXP097 $B/runs/$V/mate/deep_infer.py || true)
  n_i1=$(grep -c EXP097 $B/runs/$V/inseason.py || true)
  n_i2=$(grep -c EXP097 $B/runs/$V/mate/inseason.py || true)
  echo "$V nn=$n_nn deep=$n_dp ins_root=$n_i1 ins_mate=$n_i2"
done
echo SETUP_DONE
