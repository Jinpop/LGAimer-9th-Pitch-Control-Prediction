#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
       NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
H=/home/<user>/SungjinPark
PY=$H/venv/bin/python
cd $H/exp074_fm
mkdir -p logs
echo "=== s1 $(date) ==="; $PY src/s1_tables.py
echo "=== s0 $(date) ==="; $PY src/s0_ship.py
echo "=== s2 $(date) ==="; $PY src/s2_screen.py
echo "=== DONE $(date) ==="
