#!/usr/bin/env bash
# exp_079 적합 드라이버.  fit 당 thread_count 6 고정 (S17), 동시 P 개.
# 컨테이너 CPU 쿼터 20코어 → 6×3=18 이 상한.
set -uo pipefail
E=/home/<user>/SungjinPark/exp079
J="$1"; P="${2:-3}"
echo "=== 시작 $(date +%F' '%T) · 작업표 $J · 동시 $P ==="
xargs -a "$J" -P "$P" -L 1 bash -c '
  script=$1; shift
  tag=$(echo "$@" | tr " " "_")
  /home/<user>/SungjinPark/venv/bin/python \
    /home/<user>/SungjinPark/exp079/src/${script} "$@" \
    >> /home/<user>/SungjinPark/exp079/logs/${script%.py}_${tag}.log 2>&1 \
    && echo "[ok]   $script $* $(date +%T)" \
    || echo "[FAIL] $script $* $(date +%T)"
' _
echo "=== 종료 $(date +%F' '%T) ==="
