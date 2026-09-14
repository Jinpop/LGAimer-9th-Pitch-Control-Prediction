#!/bin/bash
cd /home/<user>/SungjinPark/exp096
until grep -q LANE_A_DONE logs/laneA.log 2>/dev/null; do sleep 30; done
/home/<user>/SungjinPark/venv/bin/python src/eval096.py s1_attn.json attn > logs/eval_a.log 2>&1
