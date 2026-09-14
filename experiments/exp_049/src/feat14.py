"""exp_049 — the 14 target-independent columns of team-mate Q's 77, as ONE module.

WHAT THESE 14 ARE
-----------------
    p6_{curve,cutter,fastball,offspeed,sinker,slider}   6   pitch-type model output
    idea_matchup_pitch                                   1   (p_hand, b_hand, argmax p6)
    i3_anom_{six types}, i3_anom_l1                      7   p6 vs the count's mean mix

They are exactly the subset of Q's `pipeline.apply_features` that reads NO TARGET.
The other eight columns Q's module produces (`sit_*` 6, `idea_cnt_te`, `i3_te48`)
are target encodings and are OUT OF SCOPE here.

WHOSE TABLES
------------
NOT Q's.  Q ships one `pitchmodel6.txt` fitted on trackman through 2024 and one
`anom_cnt.csv` scored by it, and their README says the pair is reused for every
season because "p6 is not target-derived".  For a fold_2024 measurement that is
not usable: the 2024 validation rows are inside that model's training set.

This module therefore consumes the AS-OF tables exp_037 built, in which the
pitch model is refitted at every season boundary from trackman rows with
season <= s-1, joined to main-data `pitcher_id` through a map that was itself
re-aggregated per cutoff.  A 2024 row is scored by a <=2023 model.

The arithmetic below is `mate_pipeline_ORIGINAL.py` / exp_037 `featgen.py`
verbatim for these 14 columns: same PM_FEATS, same PM_CATS, same COMBO
enumeration, same EPS clip, same nanargmax / nansum axis.

ROW INDEPENDENCE
----------------
Every statement here is a per-row lookup or a per-row arithmetic expression.
The only cross-row object is the frozen table read off disk.  `predict` is
LightGBM's per-row tree walk.  s0_gates.py proves this empirically by shuffling
and by scoring single rows in isolation; it does not take this docstring's word.
"""
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

TYPES = ["curve", "cutter", "fastball", "offspeed", "sinker", "slider"]
PC = [f"p6_{t}" for t in TYPES]
IDEA = "idea_matchup_pitch"
ANOM = [f"i3_anom_{t}" for t in TYPES] + ["i3_anom_l1"]
COLS14 = PC + [IDEA] + ANOM
assert len(COLS14) == 14 and len(set(COLS14)) == 14

GROUPS14 = {"p6": PC, "idea": [IDEA], "i3": ANOM}

PM_FEATS = ["inning", "top_bottom", "balls_before", "strikes_before",
            "outs_before", "batter_hand", "pitcher_hand", "pitcher_id"]
PM_CATS = ["top_bottom", "batter_hand", "pitcher_hand", "pitcher_id"]
COMBO = {f"{p}_{b}_{d}": i for i, (p, b, d) in enumerate(
    [(p, b, d) for p in (1, 2) for b in (1, 2) for d in range(-1, 6)])}
EPS = 1e-3

# exp_037's pinned copy of Q's parameters (+ the three threading/determinism
# keys exp_037 added and listed as a deviation).
PM_PARAMS = {"objective": "multiclass", "num_class": 6, "learning_rate": 0.08,
             "num_leaves": 63, "min_data_in_leaf": 300, "feature_fraction": 0.9,
             "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 42,
             "verbose": -1,
             "num_threads": 6, "deterministic": True, "force_row_wise": True}
PM_PREDICT_THREADS = 6
PM_ROUNDS = 250


# --------------------------------------------------------------- builders
def build_pitch_model(tm_hist, outdir):
    """tm_hist: trackman rows with season <= s-1, `pitcher_id` already joined."""
    y = tm_hist.lab.map({t: i for i, t in enumerate(TYPES)}).values
    assert not pd.isna(y).any()
    X = tm_hist[PM_FEATS].copy()
    lv = {c: sorted(X[c].dropna().unique().tolist()) for c in PM_CATS}
    for c in PM_CATS:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    m = lgb.train(PM_PARAMS, lgb.Dataset(X, y, categorical_feature=PM_CATS),
                  PM_ROUNDS)
    m.save_model(f"{outdir}/pitchmodel6.txt")
    json.dump({k: [int(x) if not isinstance(x, str) else x for x in v]
               for k, v in lv.items()}, open(f"{outdir}/pmcat6.json", "w"))
    return int(len(tm_hist))


def build_anom_table(df_rows_with_p6, outdir):
    """count-wise mean of the PREDICTED mix over the same <= s-1 rows."""
    cs = (df_rows_with_p6.balls_before.astype(str) + "_"
          + df_rows_with_p6.strikes_before.astype(str))
    df_rows_with_p6.groupby(cs)[PC].mean().to_csv(f"{outdir}/anom_cnt.csv")


# ----------------------------------------------------------------- apply
def predict_p6(df, tdir):
    m = lgb.Booster(model_file=os.path.join(tdir, "pitchmodel6.txt"))
    lv = json.load(open(os.path.join(tdir, "pmcat6.json")))
    X = df[PM_FEATS].copy()
    for c in PM_CATS:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    return m.predict(X, num_threads=PM_PREDICT_THREADS)


def apply14(df, tdir):
    """The 14 columns for the rows of `df`, from the frozen tables in `tdir`."""
    P = predict_p6(df, tdir)
    out = pd.DataFrame(P, columns=PC, index=df.index)

    cs = df.balls_before.astype(str) + "_" + df.strikes_before.astype(str)

    dom = np.nanargmax(np.nan_to_num(P, nan=-1), axis=1)
    key = (df.pitcher_hand.astype(int).astype(str) + "_"
           + df.batter_hand.astype(int).astype(str) + "_"
           + pd.Series(dom, index=df.index).astype(str))
    out[IDEA] = key.map(COMBO).astype(float).values

    ac = pd.read_csv(os.path.join(tdir, "anom_cnt.csv"), index_col=0)
    ref = ac.reindex(cs)[PC].values
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = P / np.clip(ref, EPS, None)
    for i, t in enumerate(TYPES):
        out[f"i3_anom_{t}"] = ratio[:, i]
    out["i3_anom_l1"] = np.nansum(np.abs(P - ref), axis=1)
    return out[COLS14]
