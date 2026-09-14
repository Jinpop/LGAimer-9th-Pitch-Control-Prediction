"""exp_052 stage 0 / G0a — STRUCTURAL DIAGNOSTICS ONLY (no matching decision).

Purpose: check the assumptions the registered matching method rests on, BEFORE
the method is registered and run.  This file makes no accept/reject decision
and computes no match rate.

Checked here:
  D1  row counts and per-season counts on both sides
  D2  trackman: is `pitch_no` a contiguous 1..N ordering inside a game?
  D3  trackman: is `trackman_game_id` one game (one date, one team pair)?
  D4  can train `pitcher_team_id` be mapped 1:1 to trackman `pitcher_team`?
  D5  is train row order chronological inside a game block?
  D6  game_type distribution in train; does trackman carry an equivalent?
"""
import json
import os
import time

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_052/res")

TR_COLS = ["row_id", "season", "game_month", "game_dayofweek", "inning",
           "top_bottom", "game_type", "balls_before", "strikes_before",
           "outs_before", "pitcher_id", "batter_id", "pitcher_hand",
           "batter_hand", "pitcher_team_id", "batter_team_id",
           "control_success"]
TM_COLS = ["trackman_id", "season", "game_date", "game_month",
           "game_dayofweek", "trackman_game_id", "pitch_no", "inning",
           "top_bottom", "balls_before", "strikes_before", "outs_before",
           "pitch_of_pa", "pitcher_trackman_id", "batter_trackman_id",
           "pitcher_hand", "batter_hand", "pitcher_team", "batter_team"]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    out = {}

    tr = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=TR_COLS)
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                     usecols=TM_COLS)
    out["D1"] = {
        "train_rows": int(len(tr)), "tm_rows": int(len(tm)),
        "train_by_season": {int(k): int(v) for k, v in
                            tr.season.value_counts().sort_index().items()},
        "tm_by_season": {int(k): int(v) for k, v in
                         tm.season.value_counts().sort_index().items()},
        "train_cols": list(tr.columns), "tm_cols": list(tm.columns),
    }

    # ---- D2: pitch_no contiguity inside a trackman game --------------------
    g = tm.groupby("trackman_game_id")["pitch_no"]
    stats = pd.DataFrame({"n": g.size(), "mn": g.min(), "mx": g.max(),
                          "nuniq": g.nunique()})
    out["D2"] = {
        "n_games": int(len(stats)),
        "pct_min_is_1": float((stats.mn == 1).mean() * 100),
        "pct_max_eq_n": float((stats.mx == stats.n).mean() * 100),
        "pct_all_unique": float((stats.nuniq == stats.n).mean() * 100),
        "pitches_per_game_describe": {
            k: float(v) for k, v in stats.n.describe().items()},
    }

    # ---- D3: is a trackman_game_id one date + one team pair? ---------------
    gg = tm.groupby("trackman_game_id")
    out["D3"] = {
        "pct_single_date": float((gg.game_date.nunique() == 1).mean() * 100),
        "pct_single_season": float((gg.season.nunique() == 1).mean() * 100),
        "n_distinct_pitcher_team": int(tm.pitcher_team.nunique()),
        "n_distinct_batter_team": int(tm.batter_team.nunique()),
        "pct_two_pitcher_teams": float(
            (gg.pitcher_team.nunique() == 2).mean() * 100),
    }

    # ---- D4: team id <-> team code, via (season, month, dow) signatures ----
    tr["bucket"] = (tr.season.astype(str) + "-" + tr.game_month.astype(str)
                    + "-" + tr.game_dayofweek.astype(str))
    tm["bucket"] = (tm.season.astype(str) + "-" + tm.game_month.astype(str)
                    + "-" + tm.game_dayofweek.astype(str))
    a = pd.crosstab(tr.pitcher_team_id, tr.bucket)
    b = pd.crosstab(tm.pitcher_team, tm.bucket)
    cols = sorted(set(a.columns) & set(b.columns))
    A = a[cols].values.astype(float)
    B = b[cols].values.astype(float)
    A = A / np.maximum(A.sum(1, keepdims=True), 1)
    B = B / np.maximum(B.sum(1, keepdims=True), 1)
    # cosine similarity between every train team id and every tm team code
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    S = An @ Bn.T
    best = S.argmax(1)
    srt = np.sort(S, axis=1)
    out["D4"] = {
        "n_train_team_ids": int(len(a)), "n_tm_team_codes": int(len(b)),
        "argmax_is_bijection": bool(len(set(best.tolist())) == len(best)),
        "min_best_cos": float(srt[:, -1].min()),
        "max_second_cos": float(srt[:, -2].max()),
        "proposed": {int(a.index[i]): str(b.index[best[i]])
                     for i in range(len(a))},
    }

    # ---- D5: is train row order chronological inside a game block? --------
    # crude block cut: change in (season, month, dow, game_type, team pair)
    pair = np.minimum(tr.pitcher_team_id, tr.batter_team_id).astype(str) + "_" \
        + np.maximum(tr.pitcher_team_id, tr.batter_team_id).astype(str)
    key = (tr.season.astype(str) + "|" + tr.game_month.astype(str) + "|"
           + tr.game_dayofweek.astype(str) + "|" + tr.game_type.astype(str)
           + "|" + pair)
    cut = (key != key.shift(1)).values
    blk = np.cumsum(cut) - 1
    tr["blk0"] = blk
    inn = tr.inning.values
    back = np.zeros(len(tr), bool)
    back[1:] = (inn[1:] < inn[:-1]) & (blk[1:] == blk[:-1])
    out["D5"] = {
        "n_blocks_naive": int(blk.max() + 1),
        "n_inning_regressions_inside_block": int(back.sum()),
        "rows_sorted_by_row_id": bool(tr.row_id.is_monotonic_increasing),
        "season_monotone_in_file": bool(tr.season.is_monotonic_increasing),
    }

    # ---- D6: game_type ----------------------------------------------------
    out["D6"] = {
        "train_game_type": {str(k): int(v) for k, v in
                            tr.game_type.value_counts().items()},
        "train_game_type_by_season": {
            str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in tr.groupby("season").game_type.value_counts()
            .unstack(fill_value=0).iterrows()},
        "label_base_rate_overall": float(tr.control_success.mean()),
    }

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "g0a_diag.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
