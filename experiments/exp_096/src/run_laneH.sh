#!/bin/bash
# A1(attn) 재발사 — laneA 는 예측 청크 32768 CUDA OOM 으로 fit 0건이었다.
# 바뀐 것은 PRED_CHUNK=8192 하나. 학습 경로는 한 글자도 안 바뀐다.
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
export ATT_D=192 ATT_BLK=3 ATT_HEAD=8 ATT_MULT=2 MICRO=2048 PRED_CHUNK=8192
for S in 42 777 2024 101 7; do
  $PY src/train096.py attn $S cuda:1 attn_oot23_s$S >> logs/laneH.log 2>&1
done
$PY src/eval096.py s1_attn.json attn > logs/eval_h.log 2>&1
$PY src/mk_metrics.py > logs/mk_metrics2.log 2>&1
echo "LANE_H_DONE" >> logs/laneH.log
