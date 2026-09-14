#!/bin/bash
cd /home/<user>/SungjinPark/exp096
until grep -q LANE_C_DONE logs/laneC.log 2>/dev/null; do sleep 15; done
/home/<user>/SungjinPark/venv/bin/python src/eval096.py s2_seq.json seqA seqB seqN > logs/eval_c.log 2>&1
