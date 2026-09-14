"""exp_052 stage 0 / G0a — train row  <->  trackman row correspondence.

=============================================================================
PRE-REGISTERED METHOD.  Written and shipped BEFORE any match rate was seen.
The pass/fail threshold is config exp_052's: unique match rate >= 60% of all
1,475,092 train rows.  Nothing below is tuned on the answer.
=============================================================================

WHY NOT exp_002's METHOD AS-IS
------------------------------
exp_002 matched games by an EXACT md5 of the whole game's
(inning, top_bottom, B/S/O, pitcher_hand) sequence.  One missing tracked pitch
anywhere in a game destroys the whole game's fingerprint, and it reached 49.27%
of rows -- below this experiment's 60% gate.  This file keeps exp_002's exact
fingerprint as the SEED stage (it is high precision) and adds two things:

  * a game-identity stage that does not need the pitch sequence at all
    (Jaccard of the game's PITCHER SET, translated through the seed stage's
    id map), so a game with a few untracked pitches is still identifiable;
  * an ORDER-PRESERVING sequence alignment inside a matched game, so the
    pitches that ARE present line up 1:1 even when some are missing.

REGISTERED PIPELINE (fixed constants marked [K])
------------------------------------------------
S1  train game blocks: cut where (season, month, dow, game_type, unordered
    team-id pair) changes, OR where `inning` regresses inside a block
    (double-header split).  This is exp_002's rule; it reproduces its 4,868.
S2  hand encoding: train {1,2} -> trackman {Right,Left} by global frequency
    rank, then ASSERTED against the seed-stage matches.
S3  SEED (exp_002 stage A): key = (season, month, dow, n_pitches, md5 of the
    (inning, top_bottom, balls, strikes, outs, pitcher_hand) sequence).
    Accept only keys that are unique on BOTH sides.
S4  id maps: from the seed games' positional row correspondence, count
    (pitcher_id, pitcher_trackman_id) pairs; build a global 1:1 map greedily
    by descending evidence, skipping a pair if either side is already used.
    Same for batters.
S5  EXTEND (one pass, no iteration): for each still-unmatched train block,
    candidates = trackman games with the same (season, month, dow) that no
    accepted match already uses.  score = Jaccard(train block's pitcher ids
    translated by S4, candidate's pitcher_trackman_id set).
    Accept iff  top >= 0.50 [K]  AND  top - second >= 0.20 [K].
    If >= 2 candidates survive the top>=0.50 test without the margin, the
    block's rows are counted as MULTI (ambiguous), not as unmatched.
S6  ROW ALIGNMENT: inside each accepted (block, game) pair, run
    difflib.SequenceMatcher (autojunk=False) over the integer-coded key
    sequence (inning, top_bottom, balls, strikes, outs, pitcher_hand,
    batter_hand); every position inside a matching block is a 1:1 row pair.
S7  ROW ACCEPTANCE: a pair is kept iff
      (a) both hands agree, AND
      (b) if the train pitcher_id is in the S4 map, the paired
          pitcher_trackman_id equals the mapped one (same for batter).
    Everything else is unmatched.  Contradictions are DROPPED, never coerced.

OUTPUT
------
  res/g0a_match.json    the rates, coverage and selection-bias table
  work/tm_join.npz      `tm_row` int64 (len = 1,475,092), -1 where unmatched
                        -- the row-wise join stage 1 would consume.

ROW INDEPENDENCE / RULE 2-4
---------------------------
This file touches train.csv and trackman_history.csv only.  It never reads
test.csv.  The join it emits is a per-row integer index; the consumer looks up
one trackman row per train row.  No test-frame statistic exists here.
"""
import difflib
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data")
BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
REF002 = os.path.join(BASE, "ref_exp002")

JACC_MIN = 0.50          # [K] registered
JACC_MARGIN = 0.20       # [K] registered
GATE_UNIQUE_PCT = 60.0   # config exp_052

TR_COLS = ["row_id", "season", "game_month", "game_dayofweek", "inning",
           "top_bottom", "game_type", "balls_before", "strikes_before",
           "outs_before", "pitcher_id", "batter_id", "pitcher_hand",
           "batter_hand", "pitcher_team_id", "batter_team_id",
           "control_success"]
TM_COLS = ["season", "game_month", "game_dayofweek", "trackman_game_id",
           "pitch_no", "inning", "top_bottom", "balls_before",
           "strikes_before", "outs_before", "pitcher_trackman_id",
           "batter_trackman_id", "pitcher_hand", "batter_hand"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    os.makedirs(os.path.join(BASE, "res"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "work"), exist_ok=True)
    out = {"method": "pre-registered, see module docstring",
           "constants": {"JACC_MIN": JACC_MIN, "JACC_MARGIN": JACC_MARGIN,
                         "GATE_UNIQUE_PCT": GATE_UNIQUE_PCT}}

    log("load")
    tr = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=TR_COLS)
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                     usecols=TM_COLS)
    n_tr = len(tr)
    out["n_train"] = int(n_tr)
    out["n_tm"] = int(len(tm))

    # ---- S1: train game blocks -------------------------------------------
    pmin = np.minimum(tr.pitcher_team_id.values, tr.batter_team_id.values)
    pmax = np.maximum(tr.pitcher_team_id.values, tr.batter_team_id.values)
    key = (tr.season.astype(str) + "|" + tr.game_month.astype(str) + "|"
           + tr.game_dayofweek.astype(str) + "|" + tr.game_type.astype(str)
           + "|" + pd.Series(pmin).astype(str) + "|"
           + pd.Series(pmax).astype(str)).values
    inn = tr.inning.values
    cut = np.zeros(n_tr, bool)
    cut[0] = True
    cut[1:] = (key[1:] != key[:-1]) | (inn[1:] < inn[:-1])
    blk = np.cumsum(cut) - 1
    tr["blk"] = blk
    n_blocks = int(blk.max() + 1)
    out["S1_n_train_blocks"] = n_blocks
    log("blocks", n_blocks)

    # ---- S2: hand encoding ------------------------------------------------
    tr_hand_order = tr.pitcher_hand.value_counts().index.tolist()
    tm_hand_order = tm.pitcher_hand.value_counts().index.tolist()
    hand_map = {int(a): str(b) for a, b in zip(tr_hand_order, tm_hand_order)}
    out["S2_hand_map"] = {str(k): v for k, v in hand_map.items()}
    tr_ph = tr.pitcher_hand.map(hand_map).values
    tr_bh = tr.batter_hand.map(hand_map).values

    # ---- integer coding of the alignment key ------------------------------
    def code(inning, tb, b, s, o, ph, bh):
        tbv = (np.asarray(tb) == "T") | (np.asarray(tb) == "Top")
        phv = np.asarray(ph) == "Right"
        bhv = np.asarray(bh) == "Right"
        return (np.asarray(inning).astype(np.int64) * 2 + tbv) * 4 * 3 * 3 \
            * 2 * 2 + (np.asarray(b).astype(np.int64) * 3 * 3 + np.asarray(s)
                       .astype(np.int64) * 3 + np.asarray(o).astype(np.int64)
                       ) * 4 + phv * 2 + bhv

    tr_code = code(tr.inning, tr.top_bottom, tr.balls_before,
                   tr.strikes_before, tr.outs_before, tr_ph, tr_bh)
    # exp_002's fingerprint key omits batter_hand (switch hitters disagree)
    tr_fp = tr_code // 2

    tm = tm.sort_values(["trackman_game_id", "pitch_no"], kind="mergesort")
    tm_pos = tm.index.to_numpy()          # original row index in file order
    tm = tm.reset_index(drop=True)
    tm_code = code(tm.inning, tm.top_bottom, tm.balls_before,
                   tm.strikes_before, tm.outs_before, tm.pitcher_hand,
                   tm.batter_hand)
    tm_fp = tm_code // 2

    # ---- group boundaries -------------------------------------------------
    def spans(labels):
        chg = np.empty(len(labels), bool)
        chg[0] = True
        chg[1:] = labels[1:] != labels[:-1]
        starts = np.flatnonzero(chg)
        ends = np.append(starts[1:], len(labels))
        return starts, ends

    tr_s, tr_e = spans(blk)
    tm_gid = tm.trackman_game_id.values
    tm_s, tm_e = spans(tm_gid)
    n_games = len(tm_s)
    out["S1_n_tm_games"] = int(n_games)

    tr_meta = [(int(tr.season.values[a]), int(tr.game_month.values[a]),
                int(tr.game_dayofweek.values[a])) for a in tr_s]
    tm_meta = [(int(tm.season.values[a]), int(tm.game_month.values[a]),
                int(tm.game_dayofweek.values[a])) for a in tm_s]

    # ---- S3: seed via exact full-sequence fingerprint ----------------------
    def fp(arr):
        return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()

    trk, tmk = {}, {}
    for i in range(n_blocks):
        a, b = tr_s[i], tr_e[i]
        k = tr_meta[i] + (b - a, fp(tr_fp[a:b]))
        trk.setdefault(k, []).append(i)
    for j in range(n_games):
        a, b = tm_s[j], tm_e[j]
        k = tm_meta[j] + (b - a, fp(tm_fp[a:b]))
        tmk.setdefault(k, []).append(j)
    seed = {}
    for k, v in trk.items():
        w = tmk.get(k)
        if len(v) == 1 and w is not None and len(w) == 1:
            seed[v[0]] = w[0]
    out["S3_seed_games"] = len(seed)
    log("seed games", len(seed))

    # ---- S4: id maps from the seed games ----------------------------------
    from collections import Counter
    pc, bc = Counter(), Counter()
    for i, j in seed.items():
        a, b = tr_s[i], tr_e[i]
        c, d = tm_s[j], tm_e[j]
        assert b - a == d - c
        for p, q in zip(tr.pitcher_id.values[a:b],
                        tm.pitcher_trackman_id.values[c:d]):
            pc[(int(p), int(q))] += 1
        for p, q in zip(tr.batter_id.values[a:b],
                        tm.batter_trackman_id.values[c:d]):
            bc[(int(p), int(q))] += 1

    def bijection(cnt):
        used_l, used_r, m = set(), set(), {}
        for (l, r), n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            if l in used_l or r in used_r:
                continue
            used_l.add(l)
            used_r.add(r)
            m[l] = r
        return m

    pmap = bijection(pc)
    bmap = bijection(bc)
    out["S4_pitcher_map_size"] = len(pmap)
    out["S4_batter_map_size"] = len(bmap)

    # cross-check against exp_002's independently built map (report only)
    try:
        e2 = pd.read_csv(os.path.join(REF002, "pitcher_id_map.csv"))
        common = [(int(r.pitcher_id), int(r.pitcher_trackman_id))
                  for r in e2.itertuples() if int(r.pitcher_id) in pmap]
        agree = sum(1 for l, r in common if pmap[l] == r)
        out["S4_vs_exp002"] = {"exp002_rows": int(len(e2)),
                              "overlap": len(common), "agree": agree,
                              "agree_pct": round(100 * agree / max(len(common), 1), 3)}
    except Exception as e:                                    # noqa: BLE001
        out["S4_vs_exp002"] = {"error": repr(e)}

    # ---- S5: extend by pitcher-set Jaccard --------------------------------
    tm_pset = [set(tm.pitcher_trackman_id.values[tm_s[j]:tm_e[j]].tolist())
               for j in range(n_games)]
    by_meta = {}
    for j in range(n_games):
        by_meta.setdefault(tm_meta[j], []).append(j)
    taken = set(seed.values())
    matched = dict(seed)
    ambiguous = set()
    n_no_cand = 0
    for i in range(n_blocks):
        if i in matched:
            continue
        a, b = tr_s[i], tr_e[i]
        want = {pmap[int(p)] for p in set(tr.pitcher_id.values[a:b].tolist())
                if int(p) in pmap}
        cands = [j for j in by_meta.get(tr_meta[i], []) if j not in taken]
        if not want or not cands:
            n_no_cand += 1
            continue
        sc = []
        for j in cands:
            s = tm_pset[j]
            inter = len(want & s)
            sc.append((inter / max(len(want | s), 1), j))
        sc.sort(reverse=True)
        top = sc[0][0]
        second = sc[1][0] if len(sc) > 1 else 0.0
        if top >= JACC_MIN and top - second >= JACC_MARGIN:
            matched[i] = sc[0][1]
            taken.add(sc[0][1])
        elif top >= JACC_MIN:
            ambiguous.add(i)
    out["S5_matched_games_total"] = len(matched)
    out["S5_added_by_extend"] = len(matched) - len(seed)
    out["S5_ambiguous_blocks"] = len(ambiguous)
    out["S5_blocks_no_candidate"] = n_no_cand
    log("matched games", len(matched), "ambiguous", len(ambiguous))

    # ---- S6/S7: row alignment + acceptance --------------------------------
    tm_row = np.full(n_tr, -1, np.int64)
    n_pairs = n_drop_hand = n_drop_map = 0
    for i, j in matched.items():
        a, b = tr_s[i], tr_e[i]
        c, d = tm_s[j], tm_e[j]
        A = tr_code[a:b].tolist()
        B = tm_code[c:d].tolist()
        sm = difflib.SequenceMatcher(a=A, b=B, autojunk=False)
        for ia, ib, size in sm.get_matching_blocks():
            for k in range(size):
                ra, rb = a + ia + k, c + ib + k
                n_pairs += 1
                pid, bid = int(tr.pitcher_id.values[ra]), int(tr.batter_id.values[ra])
                tp = int(tm.pitcher_trackman_id.values[rb])
                tb = int(tm.batter_trackman_id.values[rb])
                if pid in pmap and pmap[pid] != tp:
                    n_drop_map += 1
                    continue
                if bid in bmap and bmap[bid] != tb:
                    n_drop_map += 1
                    continue
                tm_row[ra] = tm_pos[rb]
    out["S6_aligned_pairs"] = int(n_pairs)
    out["S7_dropped_by_idmap"] = int(n_drop_map)
    out["S7_dropped_by_hand"] = int(n_drop_hand)

    # ---- the gate ---------------------------------------------------------
    uniq = tm_row >= 0
    amb_rows = np.zeros(n_tr, bool)
    for i in ambiguous:
        amb_rows[tr_s[i]:tr_e[i]] = True
    amb_rows &= ~uniq
    out["gate"] = {
        "unique_pct": round(100 * float(uniq.mean()), 4),
        "multi_ambiguous_pct": round(100 * float(amb_rows.mean()), 4),
        "unmatched_pct": round(100 * float((~uniq & ~amb_rows).mean()), 4),
        "threshold_pct": GATE_UNIQUE_PCT,
        "pass": bool(100 * uniq.mean() >= GATE_UNIQUE_PCT),
    }

    # ---- coverage + selection bias ---------------------------------------
    tr["uniq"] = uniq
    cov_season = tr.groupby("season").uniq.agg(["mean", "size"])
    cov_gt = tr.groupby("game_type").uniq.agg(["mean", "size"])
    cov_sg = tr.groupby(["season", "game_type"]).uniq.mean()
    out["coverage_by_season"] = {str(k): {"pct": round(100 * v["mean"], 3),
                                          "n": int(v["size"])}
                                 for k, v in cov_season.iterrows()}
    out["coverage_by_game_type"] = {str(k): {"pct": round(100 * v["mean"], 3),
                                             "n": int(v["size"])}
                                    for k, v in cov_gt.iterrows()}
    out["coverage_by_season_game_type"] = {f"{a}_{b}": round(100 * v, 3)
                                           for (a, b), v in cov_sg.items()}
    y = tr.control_success.values
    out["selection_bias"] = {
        "base_rate_all": round(float(y.mean()), 6),
        "base_rate_matched": round(float(y[uniq].mean()), 6),
        "base_rate_unmatched": round(float(y[~uniq].mean()), 6),
        "delta_matched_minus_all": round(float(y[uniq].mean() - y.mean()), 6),
        "game_type_F_share_all": round(float((tr.game_type == "F").mean()), 6),
        "game_type_F_share_matched":
            round(float((tr.game_type[uniq] == "F").mean()), 6),
        "base_rate_by_season_matched": {
            str(k): round(float(v), 6) for k, v in
            tr[uniq].groupby("season").control_success.mean().items()},
        "base_rate_by_season_all": {
            str(k): round(float(v), 6) for k, v in
            tr.groupby("season").control_success.mean().items()},
    }
    # hand agreement on the accepted pairs (S2 assertion, reported)
    sel = np.flatnonzero(uniq)
    agree_p = float((tr_ph[sel] == tm.pitcher_hand.values[
        pd.Index(tm_pos).get_indexer(tm_row[sel])]).mean())
    out["S2_hand_agreement_on_matched"] = round(agree_p, 6)

    np.savez_compressed(os.path.join(BASE, "work", "tm_join.npz"),
                        tm_row=tm_row)
    out["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(BASE, "res", "g0a_match.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(json.dumps(out["gate"], ensure_ascii=False))
    log("done", out["elapsed_min"], "min")


if __name__ == "__main__":
    main()
