# exp_108 — 딥 팔 초기화 스케일. 🔴 CPU 5코어 상한: 레인 3 + 여유 2.
#   레인당 NT092=1 (S102: 스레드 수는 z 에 안 닿는다).
set -u
B=/home/<user>/SungjinPark/exp108
PY=/home/<user>/SungjinPark/venv/bin/python
PHASE=${1:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p $B/logs $B/out/preds

if [ "$PHASE" = "1" ]; then
echo "=== REPRO GATES $(date -Is) ==="
# G1: INIT=ftt = exp_104/107 기본 경로. exp097 z_F1632_f64o_s42 와 바이트 동일해야 한다.
STEPS092=1632 NT092=1 $PY $B/src/train108.py f64o 42 cuda:0 ftt 0.0 adamw0.01 \
  GATE_ftt_s42 > $B/logs/gate_ftt.log 2>&1
echo "gate_ftt rc=$?"
# G2: INIT=u1 — **초기화 스케일 1.0 판본**. 새 파싱 분기를 타지만 난수 호출이 ftt 와
#     글자 그대로 같으므로 z 가 바이트 동일해야 한다 = "초기화만 바꿨다" 의 증거(config 게이트).
STEPS092=1632 NT092=1 $PY $B/src/train108.py f64o 42 cuda:1 u1 0.0 adamw0.01 \
  GATE_u1_s42 > $B/logs/gate_u1.log 2>&1
echo "gate_u1 rc=$?"
# G3: INIT=ftt@ce1 — 귀무 N4 배관을 배율 1.0 으로. mul_(1.0) 은 정확하므로 역시 바이트 동일.
STEPS092=1632 NT092=1 $PY $B/src/train108.py f64o 42 cuda:0 ftt@ce1 0.0 adamw0.01 \
  GATE_ce1_s42 > $B/logs/gate_ce1.log 2>&1
echo "gate_ce1 rc=$?"
$PY - <<'EOF'
import hashlib, json, numpy as np
H = "/home/<user>/SungjinPark"
ref = np.load(f"{H}/exp097/out/preds/z_F1632_f64o_s42.npy")
sh = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
out, ok = {}, True
for t in ("GATE_ftt_s42", "GATE_u1_s42", "GATE_ce1_s42"):
    z = np.load(f"{H}/exp108/out/preds/z_{t}.npy")
    m = bool(sh(z) == sh(ref)); ok &= m
    out[t] = {"sha_match_vs_exp097_z_F1632_f64o_s42": m,
              "max_abs_diff": float(np.abs(z - ref).max())}
    print(f"REPRO {t}: sha_match={m} max_abs_diff={np.abs(z-ref).max():.3e}")
out["ALL_PASS"] = bool(ok)
json.dump(out, open(f"{H}/exp108/out/gates_repro.json", "w"), indent=1)
print("REPRO_ALL_PASS" if ok else "REPRO_FAIL")
EOF
fi

J=$B/logs/jobs${PHASE}.txt
N=$(wc -l < $J); echo "=== GRID PHASE $PHASE START $N fits $(date -Is) ==="

lane () {   # $1 lane idx  $2 gpu
  i=0
  while IFS='|' read -r TAG INIT STEP SEED; do
    if [ $(( i % 3 )) -eq $1 ]; then
      if [ ! -f "$B/out/preds/z_$TAG.npy" ]; then
        STEPS092="$STEP" NT092=1 \
          $PY $B/src/train108.py f64o $SEED cuda:$2 "$INIT" 0.0 adamw0.01 $TAG \
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
