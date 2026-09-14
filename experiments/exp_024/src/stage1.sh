#!/usr/bin/env bash
# exp_024 stage-1 driver.  Seven B3 runs, at most THREE at a time (efficiency
# rule #2: concurrency 3, num_threads 6 per fit, never changed).  Longest unit
# first so the three fold_2024 runs share the first wave.
#
# The seed-42 fold_2024 run is a CODE-PATH CHECK, not a replacement: the arm
# evaluation still reuses exp_020's stored vector byte-for-byte.  exp_024's
# b3_block.py has never produced a fold_2024 unit before, so without this run
# the seed-7/2024 bases would sit on an ungated code path while seed 42 sat on
# exp_020's — and the seed comparison would be confounded by that difference
# rather than by the seed.  It costs no wall clock (it fills the first wave).
set -uo pipefail
cd ~/SungjinPark/experiments/exp_024
mkdir -p logs blocks pred
: > logs/stage1_status.txt
echo "$(date +%H:%M:%S) STAGE1 LAUNCH" >> logs/stage1_status.txt
printf '%s\n' 2024 7 2024 2024 2024 42 2023 7 2023 2024 2022 7 2022 2024 \
  | xargs -P 3 -n 2 bash src/run_one.sh
echo "$(date +%H:%M:%S) STAGE1 ALL DONE" >> logs/stage1_status.txt
