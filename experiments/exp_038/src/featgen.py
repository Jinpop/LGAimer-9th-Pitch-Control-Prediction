"""exp_037 / A2 — the team-mate's 22 columns, REBUILT ON OUR FOLD BOUNDARY.

WHAT IS INHERITED
-----------------
`mate_pipeline_ORIGINAL.py` (sha e9ccfb66...6890, carried beside this file and
sha-asserted at run time) is the team-mate's exp_032 feature module.  Its four
builders and its `apply_features` are reproduced here ARITHMETICALLY UNCHANGED:
same smoothing constants (K_SIT 80 / K_TE48 300 / K_CNT 500), same COMBO
enumeration, same EPS clip, same LightGBM multiclass parameters, same
serialise-to-disk / read-back-from-disk contract.

WHAT IS DIFFERENT, AND WHY
--------------------------
The team-mate builds ONE set of tables and ONE pitch model from all training
rows before their fold and applies it everywhere.  exp_037's leak boundary is
stricter: a row of season s may consume only tables built from seasons <= s-1.
So this file loops over target seasons and rebuilds EVERYTHING per season:

    season 2019 -> no antecedent season exists  -> all 22 columns stay NaN.
                   A sentinel fill (0 / -1 / mean) is banned by project rule.
    season s>2019 -> tables from train rows with season <= s-1
                     pitch model from trackman rows with season <= s-1,
                     joined to main-data pitcher_id through the AS-OF map
                     `map/pitcher_map_upto_{s-1}.csv` (mapgen.py).

That per-season as-of is strictly tighter than a per-fold as-of and is what
makes ONE feature matrix legitimate for fold_2022, fold_2023 and fold_2024 at
the same time: whichever fold reads it, no row ever saw its own season or any
later one.

THE PITCH MODEL IS RE-FITTED AT EVERY SEASON BOUNDARY.  A single model trained
on all trackman seasons would put 2024 physics inside a 2021 row.  Six target
seasons, five fitted models (2019 has none).

NO CURRENT-ROW TRACKMAN PHYSICS ENTERS ANY DESIGN MATRIX.  The pitch model's
inputs are the eight columns that exist in test.csv
(inning, top_bottom, balls_before, strikes_before, outs_before, batter_hand,
pitcher_hand, pitcher_id); rel_speed / spin_rate / break / extension are used
only as the model's TRAINING LABEL source (the tagged pitch type), never as a
feature of a scored row.  `p6_*` is a prediction, not a measurement.

THE DISK ROUND TRIP IS DELIBERATE.  Tables are written to CSV/txt/json and read
back before being applied, exactly as a submission would, so that A5's
reproduction gate exercises the serialisation path rather than an in-memory
shortcut.

Usage:
    python featgen.py --out FEATDIR --map-dir MAPDIR --seasons 2019,...,2024
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser("~/SungjinPark/data")
MATE_SHA = "e9ccfb66646d516cf79be3be0660c61db1f17cd54fc056284c9f696fe0416890"

TARGET = "control_success"
TYPES = ["curve", "cutter", "fastball", "offspeed", "sinker", "slider"]
PC = [f"p6_{t}" for t in TYPES]
PM_FEATS = ["inning", "top_bottom", "balls_before", "strikes_before",
            "outs_before", "batter_hand", "pitcher_hand", "pitcher_id"]
PM_CATS = ["top_bottom", "batter_hand", "pitcher_hand", "pitcher_id"]
COMBO = {f"{p}_{b}_{d}": i for i, (p, b, d) in enumerate(
    [(p, b, d) for p in (1, 2) for b in (1, 2) for d in range(-1, 6)])}
K_SIT, K_TE48, K_CNT = 80.0, 300.0, 500.0
EPS = 1e-3
# The team-mate's parameters, plus three THREADING/DETERMINISM keys they left
# unset.  Unset, LightGBM takes num_threads = nproc, which on this 40-core box
# put 143 threads on a 254k x 8 problem and spent 101 CPU-minutes in 5 wall
# minutes on the smallest of the six models.  Pinning to 6 matches the project's
# per-fit thread convention, and `deterministic` + `force_row_wise` are what make
# the A5 reproduction gate a real check rather than a coin flip: without them the
# same data can produce a different tree ensemble when the thread schedule moves.
# These three keys do not change the model's learning behaviour, only how it is
# scheduled — but they ARE a listed deviation from the source parameters.
PM_PARAMS = {"objective": "multiclass", "num_class": 6, "learning_rate": 0.08,
             "num_leaves": 63, "min_data_in_leaf": 300, "feature_fraction": 0.9,
             "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 42,
             "verbose": -1,
             "num_threads": 6, "deterministic": True, "force_row_wise": True}
PM_PREDICT_THREADS = 6
PM_ROUNDS = 250

GROUPS = {
    "p6": PC,
    "sit": ["sit_cnt", "sit_cnt_dev", "sit_run", "sit_run_dev",
            "sit_sc", "sit_sc_dev"],
    "idea": ["idea_matchup_pitch", "idea_cnt_te"],
    "i3": [f"i3_anom_{t}" for t in TYPES] + ["i3_anom_l1", "i3_te48"],
}
ALL_COLS = GROUPS["p6"] + GROUPS["sit"] + GROUPS["idea"] + GROUPS["i3"]
assert len(ALL_COLS) == 22 and len(set(ALL_COLS)) == 22

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_list(vals):
    h = hashlib.sha256()
    for v in vals:
        h.update(str(v).encode())
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# builders — arithmetic identical to mate_pipeline_ORIGINAL.py
# --------------------------------------------------------------------------- #
def build_tables(df_hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    T = TARGET
    base = df_hist[T].mean()

    cs = df_hist.balls_before.astype(str) + "_" + df_hist.strikes_before.astype(str)
    mc = (df_hist.pitcher_hand.astype(int).astype(str) + "_"
          + df_hist.batter_hand.astype(int).astype(str) + "_" + cs)

    g = df_hist[T].groupby(cs).agg(["sum", "size"])
    ((g["sum"] + K_CNT * base) / (g["size"] + K_CNT)).rename("v").to_csv(
        f"{outdir}/cnt_te.csv")

    g = df_hist[T].groupby(mc).agg(["sum", "size"])
    ((g["sum"] + K_TE48 * base) / (g["size"] + K_TE48)).rename("v").to_csv(
        f"{outdir}/te48.csv")

    d = df_hist.copy()
    d["runner"] = (d.num_runners_on > 0).astype(int)
    d["scoring"] = (d.num_runners_on >= 2).astype(int)
    overall = d.groupby("pitcher_id")[T].agg(["mean", "size"])
    overall.columns = ["p_all", "p_n"]
    overall.to_csv(f"{outdir}/pitcher_overall.csv")
    for name, keys in [("cnt", ["balls_before", "strikes_before"]),
                       ("run", ["runner"]), ("sc", ["scoring"])]:
        gg = d.groupby(["pitcher_id"] + keys)[T].agg(["mean", "size"]).join(
            overall["p_all"])
        w = gg["size"] / (gg["size"] + K_SIT)
        gg[f"sit_{name}"] = w * gg["mean"] + (1 - w) * gg["p_all"]
        gg[f"sit_{name}_dev"] = gg[f"sit_{name}"] - gg["p_all"]
        gg[[f"sit_{name}", f"sit_{name}_dev"]].reset_index().to_csv(
            f"{outdir}/sit_{name}.csv", index=False)
    json.dump({"base": float(base)}, open(f"{outdir}/meta.json", "w"))


def build_pitch_model(tm_hist, outdir):
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


def build_anom_table(df_rows_with_p6, outdir):
    cs = (df_rows_with_p6.balls_before.astype(str) + "_"
          + df_rows_with_p6.strikes_before.astype(str))
    df_rows_with_p6.groupby(cs)[PC].mean().to_csv(f"{outdir}/anom_cnt.csv")


def predict_p6(df, tdir):
    m = lgb.Booster(model_file=f"{tdir}/pitchmodel6.txt")
    lv = json.load(open(f"{tdir}/pmcat6.json"))
    X = df[PM_FEATS].copy()
    for c in PM_CATS:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    return m.predict(X, num_threads=PM_PREDICT_THREADS)


def apply_features(df, tdir):
    P = predict_p6(df, tdir)
    out = pd.DataFrame(P, columns=PC, index=df.index)

    cs = df.balls_before.astype(str) + "_" + df.strikes_before.astype(str)
    mc = (df.pitcher_hand.astype(int).astype(str) + "_"
          + df.batter_hand.astype(int).astype(str) + "_" + cs)

    cnt = pd.read_csv(f"{tdir}/cnt_te.csv", index_col=0)["v"]
    te48 = pd.read_csv(f"{tdir}/te48.csv", index_col=0)["v"]
    out["idea_cnt_te"] = cs.map(cnt).values
    out["i3_te48"] = mc.map(te48).values

    # exp_037 leak audit, finding N4.  The old line was
    #     dom = np.nanargmax(np.nan_to_num(P, nan=-1), axis=1)
    # which turns a NaN row into all-(-1) and makes argmax return class 0 in
    # silence, i.e. a whole row silently claims "fastball".  LightGBM's
    # predict() cannot emit NaN on this path so the branch has never fired,
    # but a silent wrong default is not something to leave armed.
    assert not np.isnan(P).any(), "pitch-model probabilities contain NaN"
    dom = np.argmax(P, axis=1)
    key = (df.pitcher_hand.astype(int).astype(str) + "_"
           + df.batter_hand.astype(int).astype(str) + "_"
           + pd.Series(dom, index=df.index).astype(str))
    out["idea_matchup_pitch"] = key.map(COMBO).astype(float).values

    ac = pd.read_csv(f"{tdir}/anom_cnt.csv", index_col=0)
    ref = ac.reindex(cs)[PC].values
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = P / np.clip(ref, EPS, None)
    for i, t in enumerate(TYPES):
        out[f"i3_anom_{t}"] = ratio[:, i]
    out["i3_anom_l1"] = np.nansum(np.abs(P - ref), axis=1)

    tmp = df[["pitcher_id"]].copy()
    tmp["runner"] = (df.num_runners_on > 0).astype(int)
    tmp["scoring"] = (df.num_runners_on >= 2).astype(int)
    tmp["balls_before"] = df.balls_before.values
    tmp["strikes_before"] = df.strikes_before.values
    for name, keys in [("cnt", ["balls_before", "strikes_before"]),
                       ("run", ["runner"]), ("sc", ["scoring"])]:
        tab = pd.read_csv(f"{tdir}/sit_{name}.csv")
        s = tmp[["pitcher_id"] + keys].merge(tab, on=["pitcher_id"] + keys,
                                             how="left")
        out[f"sit_{name}"] = s[f"sit_{name}"].values
        out[f"sit_{name}_dev"] = s[f"sit_{name}_dev"].values
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--seasons", default="2019,2020,2021,2022,2023,2024")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    mapdir = os.path.abspath(args.map_dir)
    seasons = [int(s) for s in args.seasons.split(",")]
    for d in ("", "tables", "prov", "feat"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    got = sha256_file(os.path.join(SRC_DIR, "mate_pipeline_ORIGINAL.py"))
    assert got == MATE_SHA, ("mate pipeline.py drifted", got)

    log("loading train ...")
    train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    assert train["row_id"].is_unique
    assert train["season"].is_monotonic_increasing
    log("train:", train.shape)
    tml = pd.read_parquet(os.path.join(mapdir, "tm_lab.parquet"))
    log("tm_lab:", tml.shape)

    prov = {"tag": args.tag, "seasons": seasons, "groups": GROUPS,
            "mate_pipeline_sha256": got, "per_season": {},
            "pitch_model_params": PM_PARAMS, "pitch_model_rounds": PM_ROUNDS,
            "smoothing": {"K_SIT": K_SIT, "K_TE48": K_TE48, "K_CNT": K_CNT},
            "sha256": {"train.csv": sha256_file(
                os.path.join(DATA, "train.csv")),
                "featgen.py": sha256_file(os.path.abspath(__file__))}}

    parts = []
    for s in seasons:
        rows = train["season"] == s
        n = int(rows.sum())
        assert n > 0, s
        if s == 2019:
            F = pd.DataFrame(np.nan, index=train.index[rows], columns=ALL_COLS)
            prov["per_season"]["2019"] = {
                "n_rows": n, "policy": "no antecedent season -> all 22 columns "
                                       "NaN, sentinel fill banned",
                "hist_n": 0, "tm_n": 0}
            log(f"season 2019: {n:,} rows -> all NaN")
        else:
            tdir = os.path.join(out, "tables", f"upto_{s-1}")
            os.makedirs(tdir, exist_ok=True)
            hist = train.loc[train["season"] <= s - 1]
            assert int(hist["season"].max()) <= s - 1
            build_tables(hist, tdir)

            pmap = pd.read_csv(os.path.join(
                mapdir, "map", f"pitcher_map_upto_{s-1}.csv"))
            tm_hist = tml.loc[tml["season"] <= s - 1].merge(
                pmap[["tr", "tm"]], left_on="pitcher_trackman_id",
                right_on="tm", how="inner")
            tm_hist = tm_hist.rename(columns={"tr": "pitcher_id"})
            assert int(tm_hist["season"].max()) <= s - 1
            assert len(tm_hist) > 0, s
            build_pitch_model(tm_hist, tdir)

            # anomaly denominator: the SAME <= s-1 rows, scored by the SAME
            # <= s-1 model.  A count-wise mean of predicted mix; no label.
            hp = hist[["balls_before", "strikes_before"]].copy()
            hp[PC] = predict_p6(hist, tdir)
            build_anom_table(hp, tdir)
            del hp

            F = apply_features(train.loc[rows], tdir)
            assert list(F.columns) == ALL_COLS or set(F.columns) == set(ALL_COLS)
            F = F[ALL_COLS]

            hid = hist["row_id"].tolist()
            tid = sorted(tm_hist["trackman_id"].tolist())
            with gzip.open(os.path.join(out, "prov",
                                        f"hist_rowids_s{s}.txt.gz"), "wt") as f:
                f.write("\n".join(map(str, hid)))
            with gzip.open(os.path.join(out, "prov",
                                        f"tm_ids_s{s}.txt.gz"), "wt") as f:
                f.write("\n".join(map(str, tid)))
            prov["per_season"][str(s)] = {
                "n_rows": n,
                "hist_n": len(hid),
                "hist_season_max_declared": int(hist["season"].max()),
                "hist_rowid_sha256": sha_list(hid),
                "tm_n": len(tid),
                "tm_season_max_declared": int(tm_hist["season"].max()),
                "tm_id_sha256": sha_list(tid),
                "n_pitchers_in_map": int(len(pmap)),
                "table_dir": f"tables/upto_{s-1}",
                "table_sha256": {fn: sha256_file(os.path.join(tdir, fn))
                                 for fn in sorted(os.listdir(tdir))},
                "base_rate_used": json.load(
                    open(os.path.join(tdir, "meta.json")))["base"],
                "na_rate": {c: round(float(F[c].isna().mean()), 6)
                            for c in ALL_COLS},
            }
            log(f"season {s}: {n:,} rows, hist {len(hid):,}, tm {len(tid):,}, "
                f"mean NA {F.isna().mean().mean():.4f}")
        part = pd.DataFrame({"row_id": train.loc[rows, "row_id"].values})
        for c in ALL_COLS:
            part[c] = np.asarray(F[c], dtype=np.float64)
        parts.append(part)

    feat = pd.concat(parts, ignore_index=True)
    if set(seasons) == {2019, 2020, 2021, 2022, 2023, 2024}:
        assert len(feat) == len(train)
        assert (feat["row_id"].to_numpy() == train["row_id"].to_numpy()).all(), \
            "feature rows are not in train file order"
    path = os.path.join(out, "feat", f"feat_{args.tag}.parquet")
    feat.to_parquet(path, index=False)
    prov["feat_file"] = os.path.relpath(path, out)
    prov["feat_sha256"] = sha256_file(path)
    prov["n_feat_rows"] = len(feat)
    prov["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out, f"featgen_{args.tag}.json"), "w") as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    log("DONE", prov["minutes"], "min ->", path)


if __name__ == "__main__":
    main()
