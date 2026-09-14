#!/bin/bash
cd /home/<user>/SungjinPark/exp096
until grep -q LANE_B_DONE logs/laneB.log 2>/dev/null; do sleep 10; done
/home/<user>/SungjinPark/venv/bin/python src/eval096.py s1_arch_partial.json cos rank ctrl35 > logs/eval_b.log 2>&1
