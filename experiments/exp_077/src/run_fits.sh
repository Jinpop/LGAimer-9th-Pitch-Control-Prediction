#!/usr/bin/env bash
# exp_077 적합 드라이버 — 컨테이너 CPU 쿼터가 20 코어(cfs 2000000/100000)이므로
# thread_count 6 × 동시 3개 = 18 스레드.  쿼터를 넘기면 초선형으로 느려진다
# (exp_064-V 사고).  동시 수만 조절하고 fit 당 스레드는 6 고정 — 스레드를 바꾸면
# 히스토그램 누산 순서가 달라져 기존 수치와 비교가 안 된다.
set -uo pipefail
E=/home/<user>/SungjinPark/exp077
J="$1"
P="${2:-3}"
echo "=== 적합 시작 $(date +%F' '%T) · 작업표 $J · 동시 $P ==="
xargs -a "$J" -P "$P" -L 1 bash -c '
  seed=$1; mask=$2; th=$3; sv=$4
  /home/<user>/SungjinPark/venv/bin/python \
    /home/<user>/SungjinPark/exp077/src/fit_one.py "$seed" "$mask" "$th" "$sv" \
    >> /home/<user>/SungjinPark/exp077/logs/fit_${mask}_s${seed}_t${th}.log 2>&1 \
    && echo "[ok]   $mask s$seed t$th  $(date +%T)" \
    || echo "[FAIL] $mask s$seed t$th  $(date +%T)"
' _
echo "=== 적합 종료 $(date +%F' '%T) ==="
ls "$E"/cache/ | grep -c "^_cat_.*\.npy$" || true
