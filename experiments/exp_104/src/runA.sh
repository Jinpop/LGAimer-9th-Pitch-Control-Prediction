#!/bin/bash
# exp_104 축A — exp_091/092/093 이 결함면에서 기각한 30지점을 **고친 f64 입력**에서 다시 굽는다.
# 하니스는 exp_102 train102b.py / exp_093 train093.py 를 기준 디렉터리만 바꿔 승계했다.
# 레인 4개 = CPU 4코어 상한(config 자원_규율). fit 당 torch 스레드 1 (S102 — z 에 안 닿는다).
B=/home/<user>/SungjinPark/exp104
PY=/home/<user>/SungjinPark/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p $B/logs $B/out/preds

echo "=== REPRO GATES $(date -Is) ==="
# G1: train104 기본 경로 = exp_097 z_F1632_f64o_s42 와 바이트 동일해야 한다.
STEPS092=1632 NT092=1 taskset -c 24 $PY $B/src/train104.py f64o 42 cuda:0 ftt 0.0 \
  adamw0.01 GATE_base_s42 > $B/logs/gate_base.log 2>&1
echo "gate_base rc=$?"
# G2: train104m TARGET093=bce 도 같은 z 를 내야 한다(출력층 교체가 딸고 오는 것만이 변인).
TARGET093=bce STEPS093=1632 NT093=1 taskset -c 25 $PY $B/src/train104m.py f64o 42 \
  cuda:1 ftt 0.0 adamw0.01 GATE_mhbce_s42 > $B/logs/gate_mhbce.log 2>&1
echo "gate_mhbce rc=$?"
$PY - <<'EOF'
import hashlib, numpy as np
H = "/home/<user>/SungjinPark"
ref = np.load(f"{H}/exp097/out/preds/z_F1632_f64o_s42.npy")
sh = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
ok = True
for t in ("GATE_base_s42", "GATE_mhbce_s42"):
    z = np.load(f"{H}/exp104/out/preds/z_{t}.npy")
    m = sh(z) == sh(ref)
    ok &= m
    print(f"REPRO {t}: sha_match={m} max_abs_diff={np.abs(z-ref).max():.3e}")
print("REPRO_ALL_PASS" if ok else "REPRO_FAIL")
EOF

# 1632 먼저 — 중간 판독을 위해 스텝 순서를 재배열한다.
J=$B/logs/jobsA_ord.txt
: > $J
for K in 1632 816 2720; do awk -F'|' -v k=$K '$11==k' $B/logs/jobsA.txt >> $J; done
N=$(wc -l < $J); echo "=== GRID START $N jobs $(date -Is) ==="

lane () {   # $1 lane idx  $2 cpu  $3 gpu
  i=0
  while IFS='|' read -r TAG RUN HID DR OPT MASK LOSS ROWS OFF TGT STEP SEED; do
    if [ $(( i % 4 )) -eq $1 ]; then
      if [ ! -f "$B/out/preds/z_$TAG.npy" ]; then
        if [ "$RUN" = "train104m" ]; then
          TARGET093="$TGT" HID093="$HID" STEPS093="$STEP" NT093=1 \
            taskset -c $2 $PY $B/src/train104m.py f64o $SEED cuda:$3 ftt $DR $OPT $TAG \
            >> $B/logs/lane$1.log 2>&1 || echo "FAIL $TAG" >> $B/logs/lane$1.log
        else
          HID092="$HID" MASK092="$MASK" LOSS092="$LOSS" ROWS092="$ROWS" OFF092="$OFF" \
          STEPS092="$STEP" NT092=1 \
            taskset -c $2 $PY $B/src/train104.py f64o $SEED cuda:$3 ftt $DR $OPT $TAG \
            >> $B/logs/lane$1.log 2>&1 || echo "FAIL $TAG" >> $B/logs/lane$1.log
        fi
      fi
    fi
    i=$((i+1))
  done < $J
  echo "LANE_$1_DONE" >> $B/logs/lane$1.log
}
lane 0 24 0 & lane 1 25 0 & lane 2 26 1 & lane 3 27 1 &
wait
echo "=== GRID END $(date -Is) ==="
ls $B/out/preds/ | wc -l
grep -c FAIL $B/logs/lane*.log
echo CHAIN_A_DONE
