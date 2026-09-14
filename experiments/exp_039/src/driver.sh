#!/usr/bin/env bash
# exp_037 driver — the whole experiment, held by the SERVER, not by an agent.
#
# WHY THIS FILE EXISTS
#   Last night six API errors and one session restart cost real time even though
#   no server job ever died: what died each time was the agent's memory of "what
#   runs next".  That ordering now lives here.  Launch it once with
#       setsid nohup bash driver.sh </dev/null > logs/driver.log 2>&1 &
#   and it runs A -> B -> C -> select -> D -> aggregate to completion, alone.
#
# CONTRACT
#   * `set -uo pipefail` and deliberately NOT `-e`: one failed unit must not
#     take the run down.  Every unit's exit code is recorded and the driver
#     continues; the summary at the end lists the failures.
#   * idempotent resume — a unit is skipped iff state/<name>.done exists, and
#     that file is written ONLY on exit code 0.  Re-launching after a crash
#     redoes nothing that succeeded.
#   * PROGRESS.jsonl gets one append-only line per unit event.
#   * HEARTBEAT gets a timestamp every 30 s from a detached writer, so liveness
#     is decidable with no agent attached and no process table access.
#   * stage-1 units that hit their round cap exit 3 and are retried ONCE at
#     RETRY_CAP; an argmin standing on the cap is not an argmin.
#
# GATES
#   B/C/D do not start unless verify_asof, verify_groupconst and gate_2024 all
#   report PASS.  That is the experiment's stated precondition and it is
#   enforced here rather than remembered.

set -uo pipefail

EXP=exp_037
ROOT="$HOME/SungjinPark/experiments/$EXP"
SRC="$ROOT/src"
FEAT="$ROOT/featdir"
GATEDIR="$ROOT/gatedir"
LOGS="$ROOT/logs"
STATE="$ROOT/state"
PY="$HOME/SungjinPark/venv/bin/python"
PROGRESS="$ROOT/PROGRESS.jsonl"
HB="$ROOT/HEARTBEAT"

MAXPAR=3
CAP=1650
RETRY_CAP=3300
FEATPQ="$FEAT/feat/feat_main.parquet"

mkdir -p "$LOGS" "$STATE" "$FEAT" "$GATEDIR" "$ROOT/res" "$ROOT/pred" "$ROOT/curves"

prog() {  # event name rc note
  printf '{"ts":"%s","event":"%s","unit":"%s","rc":%s,"note":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-}" >> "$PROGRESS"
}

# ---- heartbeat: detached, survives this shell, one writer only -------------
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

# run a unit in the FOREGROUND (stage A: strictly ordered, each feeds the next)
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

# run a unit in the BACKGROUND pool (stages B/C/D: mutually independent fits)
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

json_pass() {  # file  -> 0 if top-level "PASS": true
  "$PY" - "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    sys.exit(0 if d.get("PASS") is True else 1)
except Exception as e:
    print("gate read failed:", e); sys.exit(1)
EOF
}

prog DRIVER_START all 0 "$(date -u +%FT%TZ)"
cd "$SRC" || exit 1

# =========================================================== STAGE A: features
run_fg A1_mapgen   "$PY" "$SRC/mapgen.py" --out "$FEAT"
run_fg A2_featgen  "$PY" "$SRC/featgen.py" --out "$FEAT" --map-dir "$FEAT" \
                   --seasons 2019,2020,2021,2022,2023,2024 --tag main
run_fg A3_verify_asof      "$PY" "$SRC/verify_asof.py" --out "$FEAT" --tag main
run_fg A4_verify_groupconst "$PY" "$SRC/verify_groupconst.py" --out "$FEAT" --tag main
run_fg A5a_featgen_gate2024 "$PY" "$SRC/featgen.py" --out "$GATEDIR" \
                   --map-dir "$FEAT" --seasons 2024 --tag gate
run_fg A5b_gate_compare    "$PY" "$SRC/gate_compare.py" --main "$FEAT" --gate "$GATEDIR"

GATE_OK=1
for g in "$FEAT/verify_asof.json" "$FEAT/verify_groupconst.json" "$FEAT/gate_2024.json"; do
  if [ ! -f "$g" ] || ! json_pass "$g"; then
    GATE_OK=0
    prog GATE_FAIL "$(basename "$g")" 1 "training is blocked"
  fi
done

if [ "$GATE_OK" -ne 1 ]; then
  prog DRIVER_ABORT gates 1 "leak gates did not all pass; B/C/D skipped"
  echo "GATES FAILED — see $FEAT/verify_*.json and $FEAT/gate_2024.json" >&2
  exit 1
fi
prog GATE_PASS all 0 "verify_asof + verify_groupconst + gate_2024"

# ============================================================ STAGE B: ladder
for A in R0 R1 R2 R3 R4; do
  run_bg "B_${A}" "$PY" "$SRC/b3_feat.py" --arm "$A" --unit 2023 --seed 42 \
    --stage 1 --cap "$CAP" --feat-parquet "$FEATPQ" --out "$ROOT"
done
wait_all
prog STAGE_DONE B 0 ""

# =============================================================== STAGE C: LOO
# L_no_i3 has the same column set as R3 by construction and is aliased by
# select_best.py rather than refitted.
for A in L_no_p6 L_no_sit L_no_idea L_no_p6ALL; do
  run_bg "C_${A}" "$PY" "$SRC/b3_feat.py" --arm "$A" --unit 2023 --seed 42 \
    --stage 1 --cap "$CAP" --feat-parquet "$FEATPQ" --out "$ROOT"
done
wait_all
prog STAGE_DONE C 0 ""

# ============================================================ SELECTION (2023)
run_fg SELECT "$PY" "$SRC/select_best.py" --out "$ROOT"

PROMOTE=$("$PY" - "$ROOT/selection.json" <<'EOF'
import json, sys
try:
    print(" ".join(json.load(open(sys.argv[1]))["promote_arms"]))
except Exception:
    print("R0")
EOF
)
prog SELECTED "$PROMOTE" 0 "chosen on fold_2023 only"

# =========================================================== STAGE D: promote
for A in $PROMOTE; do
  BASE="$ROOT/res/res_${A}_u2023_seed42_st1.json"
  if [ ! -f "$BASE" ]; then
    prog SKIP "D_${A}" 1 "stage-1 base json missing"
    continue
  fi
  for S in 42 7 2024; do
    run_bg "D_${A}_s${S}" "$PY" "$SRC/b3_feat.py" --arm "$A" --unit 2024 \
      --seed "$S" --stage 2 --base-json "$BASE" \
      --feat-parquet "$FEATPQ" --out "$ROOT"
  done
done
wait_all
prog STAGE_DONE D 0 ""

# ================================================================= AGGREGATE
rm -f "$STATE/AGG.done"
run_fg AGG "$PY" "$SRC/aggregate.py" --out "$ROOT" --feat "$FEAT"

prog DRIVER_END all 0 "$(date -u +%FT%TZ)"
if [ -f "$STATE/hb.pid" ]; then kill "$(cat "$STATE/hb.pid")" 2>/dev/null; fi
echo "DRIVER COMPLETE"
