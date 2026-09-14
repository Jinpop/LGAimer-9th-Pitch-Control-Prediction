"""exp_052 stage 0.6 / TASK B step 1 — FEATURE distillation: predict the pitch's
physics from context, one physics column per process.

WHY THIS SHAPE
--------------
Stage 0 located the teacher's advantage in MOVEMENT (`induced_vert_break` gain
rank 2, `horz_break` 3; `rel_speed` only 26).  A soft label compresses
everything the teacher knows into ONE number, and movement is not a function of
the student's features, so that channel is shut.  This step does not compress:
it predicts each physics quantity from context and hands the STUDENT the
predicted quantity as a column.  `p6_*` (predicted pitch type) is exactly this
shape, is legal, and has passed audit.

🔴 WHAT A GAIN HERE WOULD AND WOULD NOT MEAN
The predicted physics is a DETERMINISTIC FUNCTION of columns the student
already holds.  It therefore carries NO new information.  If arm E beats arm A
it is a BASIS EXPANSION: a smooth function the trees cannot build for
themselves, precomputed.  Same mechanism that makes as-of rate features useful.
Do not write it up as "trackman information reached the student".

AS-OF DISCIPLINE (forward chaining, exp_037's construction)
----------------------------------------------------------
For season s the predictor is fitted ONLY on matched rows with season < s, then
applied to every season-s row.  So:
    2019 -> NaN (no prior season; no sentinel, same policy as ins_* and OOF)
    2020 -> fitted on 2019          2022 -> fitted on <= 2021
    2021 -> fitted on <= 2020       2023 -> fitted on <= 2022
A season-2023 row is scored by a <= 2022 model.  fold_2024 is never touched.

LEAK SURFACE
  * the target of these regressions is a PHYSICAL MEASUREMENT, never
    `control_success`.  The label is not in the frame at all.
  * the input list CTX is fixed below and contains no trackman column and no
    label-derived column.  Every `asof_*` RATE is excluded; only the two pure
    counters (`asof_pitcher_n`, `asof_batter_n`) are kept, and a counter has no
    label content.
    ⚠️ RECORDED CONSEQUENCE: excluding the as-of rates makes these predictors
    WEAKER than they could be, which biases arm E DOWNWARD -- i.e. toward
    closing the axis.  If arm E fails by a small margin this choice is a
    candidate explanation and must be named.
  * applied per row; the frozen booster is the only cross-row object.

Usage: python s06_phys.py --col induced_vert_break
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
PHYS6 = ["induced_vert_break", "horz_break", "rel_side", "rel_height",
         "spin_rate", "zone_speed"]
CTX = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "game_type", "balls_before", "strikes_before", "outs_before",
       "run_total_before", "score_diff_pitcher_team", "num_runners_on",
       "base_state", "li", "pitcher_id", "batter_id", "pitcher_hand",
       "batter_hand", "pitcher_team_id", "batter_team_id",
       "asof_pitcher_n", "asof_batter_n"]
CTX_CAT = ["top_bottom", "game_type", "base_state"]
CUTOFFS = [2020, 2021, 2022, 2023]
PP = {"objective": "regression", "metric": "l2", "learning_rate": 0.05,
      "num_leaves": 63, "min_data_in_leaf": 300, "feature_fraction": 0.9,
      "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 42, "verbose": -1,
      "num_threads": 6, "deterministic": True, "force_row_wise": True}
ROUNDS = 400


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", required=True, choices=PHYS6)
    a = ap.parse_args()
    col = a.col
    t0 = time.time()
    import lightgbm as lgb

    df = pd.read_parquet(os.path.join(BASE, "work",
                                      "design_fold2023.parquet"))
    assert col not in CTX and "control_success" not in CTX
    season = df["season"].to_numpy()
    has_tm = df["has_tm"].to_numpy(np.float64) > 0.5
    yv = pd.to_numeric(df[col], errors="raise").to_numpy(np.float64)
    ok = has_tm & ~np.isnan(yv)

    X = df[CTX].copy()
    for c in CTX_CAT:
        X[c] = X[c].astype(str)
    for c in [c for c in CTX if c not in CTX_CAT]:
        X[c] = pd.to_numeric(X[c], errors="raise").astype(np.float64)
    del df

    pred = np.full(len(X), np.nan, np.float64)
    info = {"col": col, "ctx_cols": CTX, "n_ctx": len(CTX), "rounds": ROUNDS,
            "params": PP, "cutoffs": {},
            "policy_2019": "NaN -- no prior season, no sentinel"}

    for s in CUTOFFS:
        fit_m = ok & (season < s)
        app_m = season == s
        ev_m = app_m & ok
        # categorical vocabulary comes from the FIT window only
        lv = {c: sorted(X.loc[fit_m, c].unique().tolist()) for c in CTX_CAT}
        Xf = X.loc[fit_m].copy()
        Xa = X.loc[app_m].copy()
        for c in CTX_CAT:
            Xf[c] = pd.Categorical(Xf[c], categories=lv[c])
            Xa[c] = pd.Categorical(Xa[c], categories=lv[c])
        booster = lgb.train(PP, lgb.Dataset(Xf, yv[fit_m],
                                            categorical_feature=CTX_CAT),
                            ROUNDS)
        p = booster.predict(Xa, num_threads=PP["num_threads"])
        pred[app_m] = p
        # holdout R^2: season-s matched rows, never in the fit window
        yy = yv[ev_m]
        pp = pred[ev_m]
        mse = float(np.mean((pp - yy) ** 2))
        var = float(np.var(yy))
        info["cutoffs"][str(s)] = {
            "fit_rows": int(fit_m.sum()), "applied_rows": int(app_m.sum()),
            "holdout_rows": int(ev_m.sum()),
            "holdout_MSE": mse, "holdout_var": var,
            "holdout_R2": 1.0 - mse / var,
            "holdout_RMSE": float(np.sqrt(mse)),
            "target_mean": float(yy.mean()),
            "cat_vocab_sizes": {c: len(v) for c, v in lv.items()}}
        log(col, s, "R2", round(1.0 - mse / var, 5),
            "fit_rows", int(fit_m.sum()))
        del booster, Xf, Xa

    info["pct_nan_pred"] = round(float(np.isnan(pred).mean() * 100), 4)
    np.savez_compressed(os.path.join(BASE, "work", f"phys_{col}.npz"),
                        pred=pred)
    with open(os.path.join(BASE, "res", f"stage06_phys_{col}.json"), "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    info["elapsed_min"] = round((time.time() - t0) / 60, 2)
    open(os.path.join(BASE, "logs", f"phys_{col}.DONE"), "w").write("ok\n")
    log("done", col, info["elapsed_min"], "min")


if __name__ == "__main__":
    main()
