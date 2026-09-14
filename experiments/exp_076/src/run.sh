#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
       NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
H=/home/<user>/SungjinPark
PY=$H/venv/bin/python
D=$H/exp076_rg
cd $D
mkdir -p logs
echo "=== s1 $(date) ==="; $PY $D/src/s1_anchors.py
echo "=== s0 $(date) ==="; $PY $D/src/s0_ship.py
echo "=== s2 $(date) ==="; $PY $D/src/s2_screen.py
echo "=== DONE $(date) ==="
