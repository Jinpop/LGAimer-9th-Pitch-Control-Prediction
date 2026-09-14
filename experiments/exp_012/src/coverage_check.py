"""exp_012 addendum — TrackMan/profile coverage by league (R vs F).

The primary gauge moved to valid_all (R + F) because test.csv is now believed to
contain Futures-league rows.  TrackMan is plausibly biased toward first-team
stadiums, so a profile that covers 99% of R rows and ~0% of F rows would look
great on valid_R and do nothing (or worse, feed the median imputer garbage) on
the 11% of test that is F.  This script measures that directly.

Reports, per (season, game_type):
  * share of pitch rows whose pitcher has a profile2 row at the fold's as-of
    cutoff (== the has_profile2 flag the model actually sees)
  * the same at the shipped cutoff-2024 static table
  * how far the imputed median sits from the R-league median, i.e. how wrong a
    median-imputed F row is

Read-only; no model is fit.  Deterministic.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile2 import TM_USECOLS, build_profile2, derive_outcomes, to_pitcher_id  # noqa: E402

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_012")
PMAP_PATH = os.path.expanduser("~/SungjinPark/experiments/exp_002/out/pitcher_id_map.csv")

KEYCOL = "cmd_ball_rate"      # the flagship command column
FLAGSRC = "cmd_n"             # what train.py uses to set has_profile2


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                     usecols=["season", "game_type", "pitcher_id"])
    pmap = pd.read_csv(PMAP_PATH)
    tm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"),
                     encoding="utf-8-sig", usecols=TM_USECOLS)
    tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]
    tm, _ = derive_outcomes(tm)

    res = {}

    # ---- coverage at each fold's as-of cutoff -------------------------------
    for cutoff, valid_year in ((2021, 2022), (2022, 2023), (2023, 2024)):
        prof = to_pitcher_id(build_profile2(tm, cutoff), pmap)
        has = prof[FLAGSRC].notna()
        ok = set(has[has].index)
        v = df[df["season"] == valid_year].copy()
        v["has"] = v["pitcher_id"].isin(ok)
        res[f"fold_{valid_year}_cutoff_{cutoff}"] = {
            "row_coverage_by_game_type": {
                g: {"n_rows": int(len(s)), "has_profile2": round(float(s["has"].mean()), 4)}
                for g, s in v.groupby("game_type")
            },
            "pitcher_coverage_by_game_type": {
                g: {
                    "n_pitchers": int(s["pitcher_id"].nunique()),
                    "with_profile2": int(s.loc[s["has"], "pitcher_id"].nunique()),
                }
                for g, s in v.groupby("game_type")
            },
        }

    # ---- shipped static table (cutoff 2024) --------------------------------
    static = to_pitcher_id(build_profile2(tm, 2024), pmap)
    ok = set(static.index[static[FLAGSRC].notna()])
    df["has"] = df["pitcher_id"].isin(ok)
    res["static_2024_row_coverage"] = {
        f"{int(s)}_{g}": round(float(x["has"].mean()), 4)
        for (s, g), x in df.groupby(["season", "game_type"])
    }
    res["static_2024_row_coverage_overall"] = {
        g: round(float(x["has"].mean()), 4) for g, x in df.groupby("game_type")
    }

    # ---- how wrong is a median-imputed F row? ------------------------------
    # SimpleImputer fills from the TRAINING-row distribution, which is dominated
    # by R.  Compare the R-row-weighted median to the F-row-weighted median of
    # the flagship column for pitchers who DO have a profile.
    cmp_ = {}
    for col in (KEYCOL, "cmd_pressure_delta", "cons_release_disp_w", "cons_move_disp_w"):
        j = df.merge(static[[col]], left_on="pitcher_id", right_index=True, how="left")
        cmp_[col] = {
            g: {"median_over_rows": (None if x[col].isna().all()
                                     else round(float(x[col].median()), 5)),
                "nan_share": round(float(x[col].isna().mean()), 4)}
            for g, x in j.groupby("game_type")
        }
    res["imputation_gap_R_vs_F"] = cmp_

    with open(os.path.join(OUT_DIR, "coverage_by_league.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
