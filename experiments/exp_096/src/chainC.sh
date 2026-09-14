#!/bin/bash
cd /home/<user>/SungjinPark/exp096
until grep -q LANE_B_DONE logs/laneB.log 2>/dev/null; do sleep 10; done
bash src/run_laneC.sh
