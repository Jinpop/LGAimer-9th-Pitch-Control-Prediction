"""exp_052 stage 0 / G0b — build the fold_2023 design matrix for the 4 teachers.

FOLD (config exp_052): train <= 2022 -> valid 2023, seed 42 only.
fold_2024 IS NOT OPENED anywhere in this experiment.

ROW SET — the deployed student's own recipe
-------------------------------------------
  train rows : season <= 2022 AND game_type != 'F'   (exp_004 F-mask)
  valid rows : season == 2023, ALL of them (valid_all)
Expected 870,752 / 245,525 — the counts exp_044's `stage1_pc.json` recorded for
its unit 2023, asserted below.  Getting them without having read that file's
row logic is the check that this frame is the same object.

COLUMN BLOCKS
-------------
  base47   train.csv minus row_id, control_success (file order preserved)
  oof4     exp_044 `stack_C5B3_pc_u2023_seed42_st1.npz`, the four forward-
           chained cell columns of ITS unit-2023 stage 1 (season 2019 all NaN,
           season 2023 predicted by <=2022 cells).
           DEVIATION, recorded: these were fitted with exp_044's per-cell HPO
           parameters, not the deployed C5 defaults.  They enter ALL FOUR arms
           identically, so they cancel in B-A and D-C, which are the only
           quantities this experiment reports.  Refitting the deployed cells
           for fold_2023 would cost ~15 min and could not change a difference.
  ins16    exp_048/051 `inseason` pack.  The pack's `build_ins_matrix` asserts
           a FROZEN anchor file that only exists for valid_season 2024, so the
           anchor loop is reproduced here for valid_season 2023 with every
           anchor rebuilt from train.csv (`build_anchor(train[season < s])`).
           Same arithmetic, no frozen file.  Season 2019 stays all-NaN.
  f14      exp_037 `featdir/feat/feat_main.parquet`, columns COLS14 only.
           The other 8 columns in that file are TARGET ENCODINGS and are not
           read.  The 14 are as-of: a season-s row is scored by tables built
           from seasons <= s-1.
  tm12     trackman physics 8 + pitch-type labels 3 + `has_tm`.
           Joined ROW-WISE through g0a_match's `tm_row` index.  Unmatched rows
           get NaN (numeric) / "__NOTM__" (categorical) + has_tm = 0.
           No sentinel numeric fill.  No groupby, no aggregation, no ordering
           dependence: one trackman row per train row, by integer index.

TEACHER-ONLY vs STUDENT-AVAILABLE
---------------------------------
tm12 is the ONLY teacher-only block.  base47 + oof4 + ins16 = 67 (arm A) and
+ f14 = 81 (arm C) are both things the deployed student can and does hold.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
DATA = os.path.expanduser("~/SungjinPark/data")
E044_OOF = os.path.expanduser(
    "~/SungjinPark/experiments/exp_044/pred/stack_C5B3_pc_u2023_seed42_st1.npz")
E037_PQ = os.path.expanduser(
    "~/SungjinPark/experiments/exp_037/featdir/feat/feat_main.parquet")
INS_PACK = os.path.expanduser("~/SungjinPark/experiments/exp_049/src/ins")
sys.path.insert(0, INS_PACK)
sys.path.insert(0, os.path.expanduser("~/SungjinPark/experiments/exp_049/src"))
import inseason as ins                                            # noqa: E402
from ins_feats import COLS16                                      # noqa: E402
from feat14 import COLS14                                         # noqa: E402

TARGET = "control_success"
OOF4 = ["p_is_mid_only", "p_is_rev_only", "p_is_both", "p_is_wayoff_true"]
TM_PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]
TM_LAB = ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]
BASE_CAT = ["top_bottom", "game_type", "base_state"]
N_TRAIN_EXPECT = 870752
N_VALID_EXPECT = 245525


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_ins_fold(df_all, valid_season):
    """`ins_feats.build_ins_matrix`'s season loop with the frozen-anchor
    assertion removed (that file exists only for valid_season 2024).  Every
    anchor is rebuilt: A(s) = ins.build_anchor(train rows with season < s)."""
    season = df_all["season"].to_numpy()
    seasons = sorted({int(s) for s in season.tolist()})
    first = seasons[0]
    V = np.full((len(df_all), 16), np.nan, np.float64)
    policy = {}
    for s in seasons:
        m = season == s
        if s == first:
            policy[str(s)] = "NONE -- earliest season, all 16 columns NaN"
            continue
        A = ins.build_anchor(df_all.loc[season < s])
        I = ins.apply(df_all.loc[m], A)
        assert list(I.columns) == COLS16, list(I.columns)
        V[m] = I.to_numpy(np.float64)
        policy[str(s)] = f"build_anchor(train[season < {s}])"
    return V, policy


def main():
    t0 = time.time()
    info = {"fold": "train<=2022 -> valid 2023", "seed_plane": 42,
            "fold_2024_opened": False}
    log("read train.csv")
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    feats47 = [c for c in tr.columns if c not in ("row_id", TARGET)]
    assert len(feats47) == 47, len(feats47)
    info["n_base_cols"] = len(feats47)

    # ---- row set -----------------------------------------------------------
    season = tr.season.values
    is_tr = (season <= 2022) & (tr.game_type.values != "F")
    is_va = season == 2023
    keep = is_tr | is_va
    assert int(is_tr.sum()) == N_TRAIN_EXPECT, int(is_tr.sum())
    assert int(is_va.sum()) == N_VALID_EXPECT, int(is_va.sum())
    info["n_train"] = int(is_tr.sum())
    info["n_valid"] = int(is_va.sum())
    info["exp004_F_mask_rows_removed"] = int(
        ((season <= 2022) & (tr.game_type.values == "F")).sum())

    # ---- ins16 (built on the FULL <=2023 frame, then subset) ---------------
    log("ins16")
    sub23 = tr.loc[season <= 2023]
    V, policy = build_ins_fold(sub23, valid_season=2023)
    ins_df = pd.DataFrame(V, columns=COLS16, index=sub23.index)
    info["ins_season_policy"] = policy
    info["ins_nan_pct_by_season"] = {
        str(int(s)): round(float(np.isnan(V[sub23.season.values == s]).mean()
                                 * 100), 3)
        for s in sorted(set(sub23.season.tolist()))}

    # ---- oof4 --------------------------------------------------------------
    log("oof4")
    z = np.load(E044_OOF)
    assert all(z[k].shape == (len(tr),) for k in OOF4)
    oof_df = pd.DataFrame({c: z[c] for c in OOF4}, index=tr.index)
    info["oof_source"] = E044_OOF
    info["oof_nan_pct_by_season"] = {
        str(int(s)): round(float(np.isnan(z["p_is_both"][season == s]).mean()
                                 * 100), 3)
        for s in sorted(set(season.tolist()))}

    # ---- f14 ---------------------------------------------------------------
    log("f14")
    f14 = pd.read_parquet(E037_PQ, columns=["row_id"] + COLS14)
    assert (f14.row_id.values == tr.row_id.values).all(), "feat_main row order"
    f14 = f14.drop(columns=["row_id"])
    f14.index = tr.index
    info["f14_cols"] = list(COLS14)

    # ---- tm12 (row-wise join) ---------------------------------------------
    log("tm12")
    tm_row = np.load(os.path.join(BASE, "work", "tm_join.npz"))["tm_row"]
    assert tm_row.shape == (len(tr),)
    tmh = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                      usecols=TM_PHYS + TM_LAB)
    has = tm_row >= 0
    safe = np.where(has, tm_row, 0)
    tm_df = pd.DataFrame(index=tr.index)
    for c in TM_PHYS:
        v = tmh[c].to_numpy(np.float64)[safe]
        tm_df[c] = np.where(has, v, np.nan)
    for c in TM_LAB:
        v = tmh[c].astype("object").to_numpy()[safe]
        v = np.where(pd.isna(v), "__NA__", v.astype(str))
        tm_df[c] = np.where(has, v, "__NOTM__")
    tm_df["has_tm"] = has.astype(np.float64)
    info["tm_has_tm_pct_overall"] = round(float(has.mean() * 100), 4)
    info["tm_has_tm_pct_train"] = round(float(has[is_tr].mean() * 100), 4)
    info["tm_has_tm_pct_valid"] = round(float(has[is_va].mean() * 100), 4)
    info["tm_phys_nan_pct_among_matched"] = {
        c: round(float(np.isnan(tmh[c].to_numpy(np.float64)[tm_row[has]])
                       .mean() * 100), 4) for c in TM_PHYS}
    info["tm_label_levels"] = {c: int(tm_df[c].nunique()) for c in TM_LAB}

    # ---- assemble ----------------------------------------------------------
    log("assemble")
    # NOTE: `season` is already inside feats47 -- do not repeat it here.
    out = pd.concat([tr[["row_id", TARGET]], tr[feats47],
                     oof_df, ins_df.reindex(tr.index), f14, tm_df], axis=1)
    out = out.loc[keep].reset_index(drop=True)
    out["_is_train"] = is_tr[keep]
    for c in BASE_CAT + TM_LAB:
        out[c] = out[c].astype(str)
    assert out[BASE_CAT + TM_LAB].notna().all().all()

    cols = {"base47": feats47, "oof4": OOF4, "ins16": list(COLS16),
            "f14": list(COLS14), "tm12": TM_PHYS + TM_LAB + ["has_tm"]}
    info["col_blocks"] = {k: len(v) for k, v in cols.items()}
    info["arm_sizes"] = {"A": 67, "B": 79, "C": 81, "D": 93}
    assert len(cols["base47"]) + len(cols["oof4"]) + len(cols["ins16"]) == 67
    os.makedirs(os.path.join(BASE, "work"), exist_ok=True)
    out.to_parquet(os.path.join(BASE, "work", "design_fold2023.parquet"),
                   index=False)
    with open(os.path.join(BASE, "work", "cols.json"), "w") as f:
        json.dump(cols, f, indent=1)
    info["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(BASE, "res", "g0b_prep.json"), "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    log(json.dumps({k: info[k] for k in
                    ("n_train", "n_valid", "tm_has_tm_pct_train",
                     "tm_has_tm_pct_valid", "arm_sizes", "elapsed_min")}))


if __name__ == "__main__":
    main()
