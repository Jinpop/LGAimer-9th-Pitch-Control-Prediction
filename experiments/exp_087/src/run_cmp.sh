#!/usr/bin/env bash
# exp_087 부가 측정 — 3시드 F 브랜치와 5시드 F 브랜치의 최종 제출 예측을 비교한다.
# 두 쪽 다 출하 script.py 를 같은 프레임에 돌려서 얻으므로, 브랜치 산술을 다시 구현하지 않고
# 배포 경로 그대로 잰다.  F 브랜치는 game_type=='F' 행에만 가중 0.5 로 닿으므로 R 행은 대조군이다.
set -uo pipefail
R=/home/<user>/SungjinPark/exp087
PY=/home/<user>/SungjinPark/venv/bin/python
FR=/home/<user>/SungjinPark/build_CA/frames/cov
mkdir -p "$R/runs" "$R/logs" "$R/work"
for tag in f3 real5; do
  d="$R/work/cmp_$tag"
  rm -rf "$d"; cp -r "$R/pkg/$tag" "$d"
  rm -rf "$d/data" "$d/output" "$d/mate/data" "$d/mate/output"; mkdir -p "$d/data"
  cp "$FR/test.csv" "$FR/sample_submission.csv" "$d/data/"
  ( cd "$d" && OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      NUMEXPR_NUM_THREADS=6 "$PY" script.py ) > "$R/logs/cmp_$tag.log" 2>&1
  rc=$?
  cp "$d/output/submission.csv" "$R/runs/cmp_$tag.csv"
  echo "[$(date +%H:%M:%S)] $tag rc=$rc rows=$(( $(wc -l < "$R/runs/cmp_$tag.csv") - 1 ))"
  rm -rf "$d"
done
"$PY" "$R/src/compare_pred.py" "$R/runs/cmp_f3.csv" "$R/runs/cmp_real5.csv" \
      "$FR/test.csv" "$R/res/pred_move_3_vs_5.json"
echo "CMP_DONE $(date +%F' '%H:%M:%S)"
