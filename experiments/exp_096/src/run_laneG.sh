#!/bin/bash
# 🔴 감사 §5(b)(c) 대조. 같은 하니스(train096.py)에서:
#    mlp96 = exp_089 의 float32 표준화 X  (딥 기준선의 하니스 동일 재현)
#    mlp64 = float64 표준화 X            (float32 결함만 고친 판본)
# rank 는 열별 단조 사상에 불변이라 두 X 에서 같은 팔이 된다 -> 이 둘이 결함 몫을 가른다.
set -uo pipefail
cd /home/<user>/SungjinPark/exp096
PY=/home/<user>/SungjinPark/venv/bin/python
for A in mlp96 mlp64; do
  for S in 42 777 2024 101 7; do
    $PY src/train096.py $A $S cuda:1 ${A}_oot23_s$S >> logs/laneG.log 2>&1
  done
done
$PY src/eval096.py s1e_ctrl.json mlp96 mlp64 > logs/eval_g.log 2>&1
echo "LANE_G_DONE" >> logs/laneG.log
