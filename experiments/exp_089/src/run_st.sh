set -uo pipefail
cd /home/<user>/SungjinPark/exp089
PY=/home/<user>/SungjinPark/venv/bin/python
I=0
for N in 720 1088; do
for SEED in 42 777 2024 101 7; do
  TAG="S${N}_oot23_s${SEED}"
  [ -f "out/fit_${TAG}.json" ] && continue
  DEV="cuda:$((I % 2))"; I=$((I+1))
  STEPS089=$N $PY src/train089.py oot23 "$SEED" "$DEV" ftt 0.0 adamw0.01 "$TAG" > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
done; done
# 배포 구간의 항등 대조: le2024 에서 STEPS089=720 이 기본(360x2)과 같은 결과여야 한다
STEPS089=720 $PY src/train089.py le2024 42 cuda:0 ftt 0.0 adamw0.01 IDENT_le2024_s42 > logs/IDENT.log 2>&1 || echo "FAILED IDENT"
echo ST_DONE
