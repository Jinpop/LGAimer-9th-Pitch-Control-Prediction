#!/usr/bin/env bash
# exp_083 캡처 — 실행당 6스레드 고정.  파일 삭제 없음(exp083 아래 새로 만드는 것만).
set -uo pipefail
H=/home/<user>/SungjinPark
O=$H/exp083
PY=$H/venv/bin/python
T=6
mkdir -p "$O/work" "$O/logs" "$O/out"

# ── cap_aa : 우리 AA 팔 (배포 부스터 = ≤2023 학습, model_cal 과 바이트 동일)
A=$O/work/cap_aa
mkdir -p "$A"
for f in aa_infer.py form_decomp.py form_offset.py inseason.py; do
  cp -f "$H/build_BO/pkg_BO/$f" "$A/$f"
done
ln -sfn "$H/build_BO/pkg_BO/model"      "$A/model"
ln -sfn "$H/build_BO/pkg_BO/model_form" "$A/model_form"
ln -sfn "$H/build_AC/work/tr2024"       "$A/data"
mkdir -p "$A/output"

# ── cap_mate : 파트너 v6 + EBM (arms/mate_cap 레이아웃 = lt2024 보정 앵커)
M=$O/work/cap_mate
mkdir -p "$M"
for f in ebm_core.py infer_ebm6.py inseason.py nn_infer.py pipeline.py splits.py v6_infer.py; do
  cp -f "$H/arms/mate_cap/$f" "$M/$f"
done
ln -sfn "$H/arms/mate_cap/model" "$M/model"
ln -sfn "$H/build_AC/work/tr2024" "$M/data"
mkdir -p "$M/output"

cp -f "$O/src/cap_aa.py" "$O/src/cap_mate.py" "$O/work/"

run() {  # <tag> <script> <workdir> <outnpz>
  local t0=$(date +%s) rc=0
  OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
    NUMEXPR_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T \
    "$PY" "$O/work/$2" "$3" "$4" > "$O/logs/$1.log" 2>&1 || rc=$?
  echo "[$(date +%H:%M:%S)] $1 rc=$rc $(( $(date +%s) - t0 ))s"
}

echo "=== exp083 캡처 시작 $(date +%F' '%H:%M:%S) ==="
run cap_aa   cap_aa.py   "$A" "$O/out/cap_aa.npz"   &
run cap_mate cap_mate.py "$M" "$O/out/cap_mate.npz" &
wait
echo "=== EXP083_CAP_DONE $(date +%F' '%H:%M:%S) ==="
