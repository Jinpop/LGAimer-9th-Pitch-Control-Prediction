set -uo pipefail
cd /home/<user>/SungjinPark/exp089
PY=/home/<user>/SungjinPark/venv/bin/python
I=0
for SEED in 42 777 2024 101 7; do
  TAG="E3_oot23_s${SEED}"
  [ -f "out/fit_${TAG}.json" ] && continue
  DEV="cuda:$((I % 2))"; I=$((I+1))
  EP089=3 $PY src/train089.py oot23 "$SEED" "$DEV" ftt 0.0 adamw0.01 "$TAG" > "logs/${TAG}.log" 2>&1 || echo "FAILED $TAG"
done
echo EP3_DONE
