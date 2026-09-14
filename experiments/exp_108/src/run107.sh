#!/bin/bash
# exp_107 — 타깃 축 격자. 레인 3개(= CPU 3코어) · fit 당 torch 스레드 1(S102).
# 🔴 CPU 6코어 상한: 레인 3 + 리크 게이트 2 + exp_106 잔여 1 = 6.
set -u
B=/home/<user>/SungjinPark/exp107
PY=/home/<user>/SungjinPark/venv/bin/python
PHASE=${1:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p $B/logs $B/out/preds

if [ "$PHASE" = "1" ]; then
echo "=== REPRO GATES $(date -Is) ==="
# G1: TGT107 없음 = exp_104/097 기본 경로. z 가 exp097 z_F1632_f64o_s42 와 바이트 동일해야 한다.
STEPS092=1632 NT092=1 $PY $B/src/train107.py f64o 42 cuda:0 ftt 0.0 adamw0.01 \
  GATE_default_s42 > $B/logs/gate_default.log 2>&1
echo "gate_default rc=$?"
# G2: TGT107=N3 항등 벡터. 같은 z 를 내야 한다 = 통로 자체가 무해함의 증거(S135).
TGT107=$B/work/tgt_N3_identity.npy STEPS092=1632 NT092=1 $PY $B/src/train107.py f64o 42 \
  cuda:1 ftt 0.0 adamw0.01 GATE_ident_s42 > $B/logs/gate_ident.log 2>&1
echo "gate_ident rc=$?"
$PY - <<'EOF'
import hashlib, json, numpy as np
H = "/home/<user>/SungjinPark"
ref = np.load(f"{H}/exp097/out/preds/z_F1632_f64o_s42.npy")
sh = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
out, ok = {}, True
for t in ("GATE_default_s42", "GATE_ident_s42"):
    z = np.load(f"{H}/exp107/out/preds/z_{t}.npy")
    m = bool(sh(z) == sh(ref)); ok &= m
    out[t] = {"sha_match_vs_exp097_z_F1632_f64o_s42": m,
              "max_abs_diff": float(np.abs(z - ref).max())}
    print(f"REPRO {t}: sha_match={m} max_abs_diff={np.abs(z-ref).max():.3e}")
out["ALL_PASS"] = bool(ok)
json.dump(out, open(f"{H}/exp107/out/gates_repro.json", "w"), indent=1)
print("REPRO_ALL_PASS" if ok else "REPRO_FAIL")
EOF
fi

J=$B/logs/jobs${PHASE}.txt
N=$(wc -l < $J); echo "=== GRID PHASE $PHASE START $N fits $(date -Is) ==="

lane () {   # $1 lane idx  $2 gpu
  i=0
  while IFS='|' read -r TAG TGT STEP SEED; do
    if [ $(( i % 3 )) -eq $1 ]; then
      if [ ! -f "$B/out/preds/z_$TAG.npy" ]; then
        TGT107="$TGT" STEPS092="$STEP" NT092=1 \
          $PY $B/src/train107.py f64o $SEED cuda:$2 ftt 0.0 adamw0.01 $TAG \
          >> $B/logs/lane$1.log 2>&1 || echo "FAIL $TAG" >> $B/logs/lane$1.log
      fi
    fi
    i=$((i+1))
  done < $J
  echo "LANE_$1_DONE" >> $B/logs/lane$1.log
}
lane 0 0 & lane 1 0 & lane 2 1 &
wait
echo "=== GRID PHASE $PHASE END $(date -Is) ==="
ls $B/out/preds/ | wc -l
grep -h FAIL $B/logs/lane*.log | sort | uniq -c
echo CHAIN_${PHASE}_DONE
