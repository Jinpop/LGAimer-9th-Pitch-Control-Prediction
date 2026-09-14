#!/usr/bin/env bash
# exp_087 G4 — 1-thread worst-case (shuffled 245,789-row frame) inference time
# as a function of the number of F-branch seeds.  STRICTLY SEQUENTIAL and the
# only load on the box: a time measured under contention cannot be compared to
# the 374.10 s the packager recorded for CA2.
set -uo pipefail
H=/home/<user>/SungjinPark
PY=$H/venv/bin/python
FRAME=$H/build_CA/frames/cov_shuf_al
ROOT=$H/exp087
WORK=$ROOT/work/g4
mkdir -p "$WORK" "$ROOT/runs" "$ROOT/logs"
MAN=$ROOT/runs/g4_manifest.tsv
[ -f "$MAN" ] || printf "tag\tn_F_seeds\trc\tsec\tsha256\trows\n" > "$MAN"

for n in 3 6 9 12 15 18; do
  tag="t1_f${n}"
  d="$WORK/$tag"
  rm -rf "$d"; cp -r "$ROOT/pkg/f${n}" "$d"
  rm -rf "$d/data" "$d/output" "$d/mate/data" "$d/mate/output"
  mkdir -p "$d/data"
  cp "$FRAME/test.csv" "$FRAME/sample_submission.csv" "$d/data/"
  sync
  t0=$(date +%s.%N); rc=0
  ( cd "$d" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 OMP_THREAD_LIMIT=1 "$PY" script.py ) \
      > "$ROOT/logs/$tag.log" 2>&1 || rc=$?
  t1=$(date +%s.%N)
  sec=$(echo "$t1 - $t0" | bc)
  sha="-"; rows="-"
  if [ -f "$d/output/submission.csv" ]; then
    cp "$d/output/submission.csv" "$ROOT/runs/$tag.csv"
    sha=$(sha256sum "$ROOT/runs/$tag.csv" | cut -d' ' -f1)
    rows=$(( $(wc -l < "$ROOT/runs/$tag.csv") - 1 ))
  fi
  printf "%s\t%d\t%d\t%s\t%s\t%s\n" "$tag" "$n" "$rc" "$sec" "$sha" "$rows" >> "$MAN"
  echo "[$(date +%H:%M:%S)] $tag rc=$rc ${sec}s rows=$rows"
  rm -rf "$d"
done
echo "G4_TIMING_DONE $(date +%F' '%H:%M:%S)"
