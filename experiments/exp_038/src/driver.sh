#!/usr/bin/env bash
# exp_038 driver — the ordering lives here, on the server, not in an agent.
#
#   setsid nohup bash driver.sh </dev/null > logs/driver.log 2>&1 &
#
# CONTRACT (exp_037's, unchanged)
#   * `set -uo pipefail`, deliberately NOT `-e`: a failed unit must not take
#     the run down.  Every exit code is recorded and the driver continues.
#   * idempotent resume — a unit is skipped iff state/<name>.done exists, and
#     that marker is written only on exit code 0.
#   * PROGRESS.jsonl gets one append-only line per unit event.
#   * HEARTBEAT is rewritten every 30 s by a detached writer, so liveness is
#     decidable with nothing attached.
#   * a stage-1 unit whose argmin lands on the round cap exits 3 and is retried
#     once at RETRY_CAP.  An argmin standing on the cap is not an argmin.
#
# GATE
#   Nothing trains until inherit_gate.py passes.  This experiment refits none
#   of the control and regenerates none of the features, so the whole run rests
#   on those artefacts being provably the ones exp_037 produced.

set -uo pipefail

EXP=exp_038
ROOT="$HOME/SungjinPark/experiments/$EXP"
SRC37="$HOME/SungjinPark/experiments/exp_037"
SRC="$ROOT/src"
LOGS="$ROOT/logs"
STATE="$ROOT/state"
PY="$HOME/SungjinPark/venv/bin/python"
PROGRESS="$ROOT/PROGRESS.jsonl"
HB="$ROOT/HEARTBEAT"

MAXPAR=3
CAP=1650
RETRY_CAP=3300
FEATPQ="$SRC37/featdir/feat/feat_main.parquet"
PROMOTED=A_idea            # fixed before stage A ran, not chosen from it

mkdir -p "$LOGS" "$STATE" "$ROOT/res" "$ROOT/pred" "$ROOT/curves"

prog() {
  printf '{"ts":"%s","event":"%s","unit":"%s","rc":%s,"note":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-}" >> "$PROGRESS"
}

if [ ! -f "$STATE/hb.pid" ] || ! kill -0 "$(cat "$STATE/hb.pid" 2>/dev/null)" 2>/dev/null; then
  setsid bash -c 'while :; do date -u +%FT%TZ > "'"$HB"'"; sleep 30; done' \
    </dev/null >/dev/null 2>&1 &
  echo $! > "$STATE/hb.pid"
fi

PIDS=()
wait_slot() {
  while :; do
    local alive=() p
    if [ "${#PIDS[@]}" -gt 0 ]; then
      for p in "${PIDS[@]}"; do
        if kill -0 "$p" 2>/dev/null; then alive+=("$p"); fi
      done
    fi
    PIDS=()
    if [ "${#alive[@]}" -gt 0 ]; then PIDS=("${alive[@]}"); fi
    if [ "${#PIDS[@]}" -lt "$MAXPAR" ]; then break; fi
    sleep 5
  done
}
wait_all() {
  local p
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null; done
  fi
  PIDS=()
}

run_fg() {
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then prog SKIP "$name" 0 "marker present"; return 0; fi
  prog START "$name" -1 ""
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$STATE/$name.done"; fi
  prog END "$name" "$rc" ""
  return $rc
}

run_bg() {
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then prog SKIP "$name" 0 "marker present"; return 0; fi
  wait_slot
  prog START "$name" -1 ""
  (
    "$@" > "$LOGS/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then
      prog RETRY "$name" 3 "argmin hit cap $CAP, retrying at $RETRY_CAP"
      "$@" --cap "$RETRY_CAP" >> "$LOGS/$name.log" 2>&1
      rc=$?
    fi
    if [ $rc -eq 0 ]; then touch "$STATE/$name.done"; fi
    prog END "$name" "$rc" ""
  ) &
  PIDS+=($!)
}

prog DRIVER_START all 0 "$(date -u +%FT%TZ)"
cd "$SRC" || exit 1

# ======================================================== GATE: inheritance
run_fg GATE_inherit "$PY" "$SRC/inherit_gate.py" --src-exp "$SRC37" --out "$ROOT"
if [ ! -f "$STATE/GATE_inherit.done" ]; then
  prog DRIVER_ABORT gate 1 "inheritance gate failed; nothing was trained"
  echo "INHERIT GATE FAILED — see $ROOT/inherit_gate.json" >&2
  exit 1
fi
prog GATE_PASS inherit 0 "feat parquet sha + R0 control sha + recomputed Brier"

# ============================================ STAGE A: fold_2023 decomposition
for A in A_idea A_cnt A_match; do
  run_bg "A_${A}" "$PY" "$SRC/b3_feat.py" --arm "$A" --unit 2023 --seed 42 \
    --stage 1 --cap "$CAP" --feat-parquet "$FEATPQ" --out "$ROOT"
done
wait_all
prog STAGE_DONE A 0 "three arms, fold_2023, seed 42"

# ================================================ STAGE B: fold_2024 promotion
BASE="$ROOT/res/res_${PROMOTED}_u2023_seed42_st1.json"
if [ ! -f "$BASE" ]; then
  prog SKIP "B_${PROMOTED}" 1 "stage-1 base json missing; stage B cannot run"
else
  for S in 42 7 2024; do
    run_bg "B_${PROMOTED}_s${S}" "$PY" "$SRC/b3_feat.py" --arm "$PROMOTED" \
      --unit 2024 --seed "$S" --stage 2 --base-json "$BASE" \
      --feat-parquet "$FEATPQ" --out "$ROOT"
  done
  # queued into the same pool of 3, so it starts only when a seed finishes and
  # concurrency never exceeds MAXPAR
  run_bg "GAIN_${PROMOTED}" "$PY" "$SRC/gain_probe.py" --arm "$PROMOTED" \
    --seed 42 --base-json "$BASE" --feat-parquet "$FEATPQ" --out "$ROOT"
fi
wait_all
prog STAGE_DONE B 0 "promoted arm was fixed in advance"

# ================================================================= AGGREGATE
rm -f "$STATE/AGG.done"
run_fg AGG "$PY" "$SRC/aggregate038.py" --out "$ROOT" --src-exp "$SRC37"

prog DRIVER_END all 0 "$(date -u +%FT%TZ)"
if [ -f "$STATE/hb.pid" ]; then kill "$(cat "$STATE/hb.pid")" 2>/dev/null; fi
echo "DRIVER COMPLETE"
