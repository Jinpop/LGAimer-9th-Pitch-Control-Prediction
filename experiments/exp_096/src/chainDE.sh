#!/bin/bash
cd /home/<user>/SungjinPark/exp096
until grep -q LANE_D_DONE logs/laneD.log 2>/dev/null; do sleep 15; done
/home/<user>/SungjinPark/venv/bin/python src/eval096.py s1c_ranksteps.json rankT2448 rankT3264 > logs/eval_d.log 2>&1
