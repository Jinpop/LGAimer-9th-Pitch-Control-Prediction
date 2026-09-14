#!/bin/bash
# 진단 전용 — 사전 등록 판정(1632스텝)에는 안 들어간다.
# T=2448·3264 가 급격히 나빠졌으므로 아래쪽을 훑어 봉우리 위치를 잡는다.
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
until grep -q LANE_E_DONE logs/laneE.log 2>/dev/null; do sleep 10; done
for T in 816 1224; do
  for S in 42 777 2024 101 7; do
    STEPS096=$T $PY src/train096.py rank $S cuda:1 rankT${T}_oot23_s$S >> logs/laneF.log 2>&1
  done
done
$PY src/eval096.py s1d_ranksteps_low.json rankT816 rankT1224 > logs/eval_f.log 2>&1
echo "LANE_F_DONE" >> logs/laneF.log
