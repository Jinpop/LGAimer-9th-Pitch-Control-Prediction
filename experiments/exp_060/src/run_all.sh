#!/usr/bin/env bash
# exp_060 driver — stage0 tm 매핑 → stage1 (ins/sp ∥ v6 연도 유닛 P3) → 조립 → GPU 학습 → 변환 검증
set -euo pipefail
B=~/SungjinPark/exp060_nn
PY=~/SungjinPark/venv/bin/python
cd "$B"
mkdir -p logs work cache out
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[$(date '+%F %T')] stage0 prep_tm"
$PY src/prep_tm.py > logs/prep_tm.log 2>&1

echo "[$(date '+%F %T')] stage1 prep_inssp (bg) + prep_v6 x6 (P3)"
$PY src/prep_inssp.py > logs/prep_inssp.log 2>&1 &
INSPID=$!
printf "2019\n2020\n2021\n2022\n2023\n2024\n" \
  | xargs -P 3 -n 1 -I{} sh -c "$PY $B/src/prep_v6.py {} > $B/logs/prep_v6_{}.log 2>&1"
wait "$INSPID"

echo "[$(date '+%F %T')] stage2 assemble"
$PY src/assemble.py > logs/assemble.log 2>&1

echo "[$(date '+%F %T')] stage3 train_nn (GPU ${CUDA_VISIBLE_DEVICES:-all})"
$PY src/train_nn.py > logs/train.log 2>&1

echo "[$(date '+%F %T')] stage4 convert_verify"
$PY src/convert_verify.py > logs/convert.log 2>&1

echo "[$(date '+%F %T')] ALL_DONE"
