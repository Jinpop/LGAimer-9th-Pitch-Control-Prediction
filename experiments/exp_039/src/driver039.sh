#!/usr/bin/env bash
# exp_039 driver — measure the denominator of the adoption rule.
#
# The whole experiment is: run the exp_037 R0 / fold_2024 / stage-2 unit six
# more times, changing nothing but --seed, and inherit the other three from
# exp_037 without refitting.  The ordering lives here rather than in an
# agent's head, so an API error or a session restart costs nothing.
#
# CONTRACT (exp_037's, unchanged)
#   * `set -uo pipefail`, deliberately NOT `-e`: a failed unit must not take
#     the run down.  A failed unit REDUCES n and is reported as such — it is
#     never replaced by a substitute seed.  That would be seed cherry-picking.
#   * idempotent resume — a unit is skipped iff state/<name>.done exists, and
#     that marker is written only on exit code 0.
#   * PROGRESS.jsonl: one append-only line per unit event.
#   * HEARTBEAT: a timestamp every 30 s from a detached writer.
#   * MAXPAR=3 with 6 threads per fit = 18 threads.  exp_038 is running the
#     same budget on the same 40-core box.  DO NOT RAISE THIS, and do not
#     touch any process that is not ours.
#
# NO RETRY-ON-CAP HERE.  These are stage-2 units: their iteration counts are
# handed down from exp_037's stage-1 argmin, so there is no cap to stand on.
# The gate checks that base was cap-free before anything fits.

set -uo pipefail

EXP=exp_039
ROOT="$HOME/SungjinPark/experiments/$EXP"
SRC="$ROOT/src"
LOGS="$ROOT/logs"
STATE="$ROOT/state"
PY="$HOME/SungjinPark/venv/bin/python"
PROGRESS="$ROOT/PROGRESS.jsonl"
HB="$ROOT/HEARTBEAT"

E37="$HOME/SungjinPark/experiments/exp_037"
BASEJSON="$E37/res/res_R0_u2023_seed42_st1.json"
FEATPQ="$E37/featdir/feat/feat_main.parquet"   # R0 has 0 extra cols; the arg
                                               # is required but never opened
MAXPAR=3
SEEDS_NEW="1 13 101 777 2025 31337"            # pre-registered.  Frozen.

mkdir -p "$LOGS" "$STATE" "$ROOT/res" "$ROOT/pred" "$ROOT/curves"

prog() {  # event unit rc note
  printf '{"ts":"%s","event":"%s","unit":"%s","rc":%s,"note":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-}" >> "$PROGRESS"
}

# ---- heartbeat: detached, survives this shell, exactly one writer ----------
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
    if [ $rc -eq 0 ]; then touch "$STATE/$name.done"; fi
    prog END "$name" "$rc" ""
  ) &
  PIDS+=($!)
}

prog DRIVER_START all 0 "$(date -u +%FT%TZ)"
cd "$SRC" || exit 1

# ============================================================= INHERITANCE GATE
run_fg GATE_inherit "$PY" "$SRC/inherit_gate.py"
if [ ! -f "$STATE/GATE_inherit.done" ]; then
  prog DRIVER_ABORT gate 1 "inheritance gate failed — no fits started"
  echo "INHERITANCE GATE FAILED — see $ROOT/inherit_gate.json" >&2
  exit 1
fi
prog GATE_PASS inherit 0 "exp_037 R0 u2024 st2 seeds 42/7/2024 inherited by sha"

# ================================================================== SIX SEEDS
for S in $SEEDS_NEW; do
  run_bg "U_seed${S}" "$PY" "$SRC/b3_feat.py" --arm R0 --unit 2024 \
    --seed "$S" --stage 2 --base-json "$BASEJSON" \
    --feat-parquet "$FEATPQ" --out "$ROOT"
done
wait_all
prog STAGE_DONE seeds 0 ""

# =================================================================== AGGREGATE
rm -f "$STATE/AGG.done"
run_fg AGG "$PY" "$SRC/aggregate039.py" --out "$ROOT"

prog DRIVER_END all 0 "$(date -u +%FT%TZ)"
if [ -f "$STATE/hb.pid" ]; then kill "$(cat "$STATE/hb.pid")" 2>/dev/null; fi
echo "DRIVER COMPLETE"
