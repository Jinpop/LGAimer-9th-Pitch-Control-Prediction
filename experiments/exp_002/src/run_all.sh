#!/bin/bash
# exp_002 전체 파이프라인 (서버에서 nohup으로 실행)
set -euo pipefail
cd ~/SungjinPark/experiments/exp_002
PY=~/SungjinPark/venv/bin/python
$PY src/01_build_mapping.py
$PY src/02_build_profile.py
$PY src/03_validate.py
echo "ALL_DONE"
