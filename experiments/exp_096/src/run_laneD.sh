#!/bin/bash
# 진단 전용 — 사전 등록 판정에는 안 들어간다.  이긴 팔(rank)의 학습량 민감도.
# 근거: 딥 팔에서 학습량 축이 LB +4.83 이었다(S111). 순위변환은 입력 조건수를 바꾸므로
#       최적 스텝이 1632 와 다를 수 있다.  1632 판본이 사전 등록 정본이다.
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
until grep -q LANE_C_DONE logs/laneC.log 2>/dev/null; do sleep 10; done
for T in 2448 3264; do
  for S in 42 777 2024 101 7; do
    STEPS096=$T $PY src/train096.py rank $S cuda:1 rankT${T}_oot23_s$S >> logs/laneD.log 2>&1
  done
done
echo "LANE_D_DONE" >> logs/laneD.log
