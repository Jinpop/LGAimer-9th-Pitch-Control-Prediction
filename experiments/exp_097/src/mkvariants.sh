#!/bin/bash
set -euo pipefail
B=/home/<user>/SungjinPark/exp097
SRC=$B/pkg/CH
FR=/home/<user>/SungjinPark/build_BX/frames/cov
for V in V0 V1 V2 V3 V4; do
  D=$B/runs/$V
  rm -rf "$D"; mkdir -p "$D"
  cp -r "$SRC"/. "$D"/
  mkdir -p "$D/data" "$D/output"
  cp "$FR/test.csv" "$D/data/test.csv"
  cp "$FR/sample_submission.csv" "$D/data/sample_submission.csv"
done
# ---- patch definitions (sed, exact-line anchored by unique text) ----
patch_nn () {  # $1 = pkg dir
  f="$1/mate/nn_infer.py"
  sed -i "s/mu = np.array(M\[mu\], dtype=np.float32)/mu = np.array(M[mu], dtype=np.float64)  # EXP097 PATCH/" "$f"
  sed -i "s/sd = np.array(M\[sd\], dtype=np.float32)/sd = np.array(M[sd], dtype=np.float64)  # EXP097 PATCH/" "$f"
  sed -i "s/X = df\[DENSE\].astype(float32).values/X = df[DENSE].astype(float64).values  # EXP097 PATCH/" "$f"
  sed -i "s/X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)/X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)  # EXP097 PATCH/" "$f"
  sed -i "s/embs = \[W\[femb{i}\] for i in range(len(CATS))\]/embs = [W[f\"emb{i}\"].astype(np.float64) for i in range(len(CATS))]  # EXP097 PATCH/" "$f"
  grep -c "EXP097 PATCH" "$f"
}
patch_deep () {
  f="$1/mate/deep_infer.py"
  sed -i "s/mu = np.array(P\[mu\], np.float32); sd = np.array(P\[sd\], np.float32)/mu = np.array(P['mu'], np.float64); sd = np.array(P['sd'], np.float64)  # EXP097 PATCH/" "$f"
  sed -i "s/X = df\[NUM\].astype(float32).values/X = df[NUM].astype(float64).values  # EXP097 PATCH/" "$f"
  sed -i "s/X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)/X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)  # EXP097 PATCH/" "$f"
  grep -c "EXP097 PATCH" "$f"
}
patch_ins () {
  for f in "$1/inseason.py" "$1/mate/inseason.py"; do
    sed -i "s/return out.astype(float32)/return out.astype(float64)  # EXP097 PATCH/" "$f"
  done
  grep -c "EXP097 PATCH" "$1/inseason.py" "$1/mate/inseason.py"
}
echo "V1 nn:"; patch_nn  $B/runs/V1
echo "V2 deep:"; patch_deep $B/runs/V2
echo "V3 ins:"; patch_ins  $B/runs/V3
echo "V4 all:"; patch_nn $B/runs/V4; patch_deep $B/runs/V4; patch_ins $B/runs/V4
echo MKVARIANTS_DONE
