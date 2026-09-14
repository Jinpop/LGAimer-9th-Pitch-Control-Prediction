#!/bin/bash
# exp_104 축A 후속 — 중간 판독에서 크기 문턱(+2.5)을 넘은 유일한 지점이 **무작위 47열 마스크**
# `D5h3_rand47` 였다(+7.41, t 1.11). 기댓값이 같은 시드 묶음 2개를 더 구워
# 그 값이 되풀이되는지, 그리고 이 계열의 **적합-간 sd** 가 얼마인지 직접 잰다(S178).
# 같은 대조를 위해 형제 귀무 D5h1·D5h2 도 같은 두 묶음으로 굽는다.
B=/home/<user>/SungjinPark/exp104
PY=/home/<user>/SungjinPark/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p $B/logs $B/out/preds
J=$B/logs/jobsC.txt
: > $J
$PY - "$J" <<'EOF'
import json, sys
meta = json.load(open("/home/<user>/SungjinPark/exp104/out/jobsA_meta.json"))
G2 = [11, 13, 17, 19, 23]
G3 = [29, 31, 37, 41, 43]
with open(sys.argv[1], "w") as f:
    for nm in ("D5h1_rand47", "D5h2_rand47", "D5h3_rand47"):
        m = meta["points"][nm]
        for s in G2 + G3:
            f.write("|".join([f"{nm}__k1632_s{s}", ",".join(map(str, m["hid"])),
                              str(m["drop"]), m["opt"],
                              ",".join(map(str, m["mask"])), "1632", str(s)]) + "\n")
EOF
N=$(wc -l < $J); echo "=== C GRID START $N jobs $(date -Is) ==="
lane () {
  i=0
  while IFS='|' read -r TAG HID DR OPT MASK STEP SEED; do
    if [ $(( i % 4 )) -eq $1 ]; then
      if [ ! -f "$B/out/preds/z_$TAG.npy" ]; then
        HID092="$HID" MASK092="$MASK" LOSS092="bce" STEPS092="$STEP" NT092=1 \
          taskset -c $2 $PY $B/src/train104.py f64o $SEED cuda:$3 ftt $DR $OPT $TAG \
          >> $B/logs/laneC$1.log 2>&1 || echo "FAIL $TAG" >> $B/logs/laneC$1.log
      fi
    fi
    i=$((i+1))
  done < $J
  echo "LANE_C$1_DONE" >> $B/logs/laneC$1.log
}
lane 0 24 0 & lane 1 25 0 & lane 2 26 1 & lane 3 27 1 &
wait
echo "=== C GRID END $(date -Is) ==="
grep -c FAIL $B/logs/laneC*.log
echo CHAIN_C_DONE
