"""exp_037 — exp_030's C5_B3 with a feature-set switch, and nothing else.

DERIVATION
----------
This file is `b3_cat.py` (exp_030, sha 0984c57e...a76ca, carried byte-identical
beside it and asserted at run time) with ONE change threaded through it: the
design matrix is `base_feats + ARMS[arm]["cols"]` instead of `base_feats`.
Masks, pools, row counts, iteration policy, curve grid, argmin rule, stacking
topology, seeds and CatBoost parameters are exp_030's.

THE FULL VARIABLE LEDGER vs exp_030
-----------------------------------
  1. extra feature columns (0 to 22 of them), read from
     `--feat-parquet`, produced by featgen.py on OUR per-season as-of boundary.
     Arm R0 adds zero columns and therefore MUST reproduce exp_030's stage-1
     fold_2023 result; that comparison is run and reported (`exp030_gate`).
  2. `idea_matchup_pitch` is carried as a FLOAT, not declared categorical.
     The team-mate declares it categorical to LightGBM/CatBoost.  C5's rule is
     "IDs stay numeric, cat_features is exactly the three native string
     columns", and changing that rule would add a second variable to every
     arm.  So the column enters as its COMBO ordinal.  This is a deviation
     from the source and is listed here because an unlisted change is the
     failure mode the project bans.
  3. the columns are added to EVERY sub-model — the four cell models, the
     forward-chained OOF fits and the stacker — not to a single flat model as
     in the team-mate's design.  "22 columns inside B3" is the thing being
     measured; it is not the same object as "22 columns inside a flat blend".

Nothing else differs.  In particular the leak boundary is exp_030's, plus the
extra columns' own boundary which is enforced in featgen.py and checked by
verify_asof.py / verify_groupconst.py / gate_compare.py BEFORE this file is
allowed to run.

Usage:
    python b3_feat.py --arm R4 --unit 2023 --seed 42 --stage 1 --cap 1650 \
        --feat-parquet FEAT/feat/feat_main.parquet --out OUTDIR
    python b3_feat.py --arm R4 --unit 2024 --seed 7 --stage 2 \
        --base-json OUTDIR/res/res_R4_u2023_seed42_st1.json ...
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import exp016_ref as ref                                            # noqa: E402
from labels import (FAIL_CELLS, assert_no_recovered_in_features,     # noqa: E402
                    recover)
from arms import ARMS, ALL_COLS, EXPECTED_N                          # noqa: E402

EXP_ID = "exp_037"
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
EXP027_TRAIN = os.path.expanduser(
    "~/SungjinPark/experiments/exp_027/src/train.py")
EXP027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"
REF016_SHA = "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38a3ece1"
B3_CAT_SHA = "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7bfa76ca"
EXP030_STAGE1 = os.path.expanduser(
    "~/SungjinPark/experiments/exp_030/stage1.json")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]
F_EXCLUDE_MAX_SEASON = 2022
N_THREADS = 6
REF_BRIER = 0.2498069

UNIT_SPEC = {
    2022: {"train_upto": 2021, "n_train": 653728, "n_valid": 247472},
    2023: {"train_upto": 2022, "n_train": 870752, "n_valid": 245525},
    2024: {"train_upto": 2023, "n_train": 1116277, "n_valid": 253507,
           "n_valid_R": 223497},
}
BASE_UNIT = 2023


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_exp027():
    got = sha256_file(EXP027_TRAIN)
    assert got == EXP027_SHA, ("exp_027 train.py changed", got, EXP027_SHA)
    spec = importlib.util.spec_from_file_location("exp027_train", EXP027_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c5 = mod.ARMS["C5"]
    assert c5["family"] == "catboost" and c5["params"] == {}
    assert c5["cat_ids"] is False
    assert mod.CB_BASE["has_time"] is True
    assert mod.CB_BASE["thread_count"] == N_THREADS
    return mod, c5


def score_of(brier):
    return 100000.0 * (1.0 - brier / REF_BRIER)


def assert_time_ordered(tkey, mask, where):
    t = tkey[mask]
    inv = int((np.diff(t) < 0).sum())
    assert inv == 0, (f"has_time=True but pool '{where}' is not in time order",
                      inv)
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--unit", type=int, required=True,
                    choices=[2022, 2023, 2024])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--cap", type=int, default=1650)
    ap.add_argument("--base-json", default="")
    ap.add_argument("--feat-parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ARM = args.arm
    V, SEED, STAGE = args.unit, args.seed, args.stage
    spec_u = UNIT_SPEC[V]
    tu, vy = spec_u["train_upto"], V
    out_dir = os.path.abspath(args.out)
    t_start = time.time()
    np.random.seed(SEED)
    for d in ("", "pred", "curves", "res"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA
    assert sha256_file(os.path.join(SRC_DIR, "b3_cat.py")) == B3_CAT_SHA, \
        "exp_030 b3_cat.py, this file's parent, has drifted"
    e27, C5 = load_exp027()
    build_frame_native = e27.build_frame_native
    chunked_margin_curve = e27.chunked_margin_curve
    fit_catboost = e27.fit_catboost

    stem = f"{ARM}_u{V}_seed{SEED}_st{STAGE}"
    tag = stem
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: arm {ARM} unit {V} "
          f"(train<={tu} -> valid {vy}) seed {SEED} stage {STAGE} "
          f"cap {args.cap}", flush=True)

    # ------------------------------------------------------------------ data
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    # -------------------------------------------------------- extra columns
    extra = list(ARMS[ARM]["cols"])
    feat_sha = None
    if extra:
        F = pd.read_parquet(os.path.abspath(args.feat_parquet))
        feat_sha = sha256_file(os.path.abspath(args.feat_parquet))
        assert len(F) == len(df), (len(F), len(df))
        assert (F["row_id"].to_numpy() == df["row_id"].to_numpy()).all(), \
            "feature parquet is not aligned with train file order"
        assert set(extra) <= set(F.columns), sorted(set(extra) - set(F.columns))
        assert not (set(extra) & set(base_feats)), "extra column shadows a base column"
        for c in extra:
            df[c] = F[c].to_numpy(dtype=np.float64)
        del F
        gc.collect()
    feats_used = base_feats + extra
    assert len(feats_used) == EXPECTED_N[ARM], (len(feats_used), ARM)
    assert len(set(feats_used)) == len(feats_used)
    assert_no_recovered_in_features(feats_used, f"{ARM} feats_used")

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0
    # the recovery adds is_* columns to df AFTER feats_used was frozen; the
    # assert above already proves none of them is in it, and the assert is
    # re-run before every Dataset build below.

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()
    tkey = (season.astype(np.int64) * 100
            + df["game_month"].to_numpy(np.int64))

    tr_full = (season <= tu) & ~f_excl
    tr_mode = tr_full & ok & (succ_season <= tu)
    va = season == vy
    va_ok = ok[va]
    y_va = y_all[va]
    assert int(tr_full.sum()) == spec_u["n_train"], (int(tr_full.sum()), V)
    assert int(va.sum()) == spec_u["n_valid"], (int(va.sum()), V)
    game_type_va = df.loc[va, "game_type"].to_numpy()
    is_r = game_type_va == "R"
    if V == 2024:
        assert int(is_r.sum()) == spec_u["n_valid_R"], int(is_r.sum())
    hand_va = df.loc[va, "pitcher_hand"].to_numpy()

    cat_native = [c for c in feats_used if c in CAT_COLS]
    assert len(cat_native) == 3, cat_native
    assert not any(c in cat_native for c in ID_COLS), "C5 keeps IDs numeric"
    assert not any(c in cat_native for c in extra), \
        "extra columns are numeric by design (see module docstring, item 2)"

    n_mode_self = int(tr_mode.sum())
    n_full_self = int(tr_full.sum())

    res = {
        "exp_id": EXP_ID, "arm": ARM, "arm_role": ARMS[ARM],
        "unit": V, "seed": SEED, "stage": STAGE,
        "architecture": "C5_B3", "train_upto": tu, "valid_season": vy,
        "n_feats_used": len(feats_used), "n_extra_cols": len(extra),
        "extra_cols": extra,
        "n_train_full": n_full_self, "n_train_mode": n_mode_self,
        "n_valid": int(va.sum()), "n_valid_R": int(is_r.sum()),
        "n_valid_F": int((~is_r).sum()),
        "n_rows_masked_exp004": int(f_excl.sum()),
        "cap": args.cap,
        "cat_features": list(cat_native),
        "id_cols_numeric": list(ID_COLS),
        "extra_cols_are_numeric": True,
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "b3_feat.py": sha256_file(os.path.abspath(__file__)),
            "exp030_b3_cat.py": B3_CAT_SHA,
            "feat_parquet": feat_sha,
            "labels.py": LABELS_SHA, "exp016_ref.py": REF016_SHA,
            "exp_027_train.py": EXP027_SHA,
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "time_inversions": {},
    }
    if extra:
        nan_tr = {c: round(float(np.isnan(df.loc[tr_full, c].to_numpy()).mean()), 6)
                  for c in extra}
        nan_va = {c: round(float(np.isnan(df.loc[va, c].to_numpy()).mean()), 6)
                  for c in extra}
        res["extra_nan_rate_train"] = nan_tr
        res["extra_nan_rate_valid"] = nan_va
    import catboost as cb                                            # noqa: F401
    res["versions"]["catboost"] = cb.__version__
    res["time_inversions"]["tr_full"] = assert_time_ordered(tkey, tr_full,
                                                            "tr_full")
    res["time_inversions"]["tr_mode"] = assert_time_ordered(tkey, tr_mode,
                                                            "tr_mode")

    # -------------------------------------------------- iteration resolution
    if STAGE == 1:
        assert V == BASE_UNIT and SEED == 42, "stage 1 is unit 2023 seed 42"
        cell_round = {c: args.cap for c in FAIL_CELLS}
        stack_round = args.cap
        res["iter_source"] = ("stage 1 — trained to cap, reported at own "
                              "fold_2023 valid argmin")
    else:
        with open(args.base_json) as f:
            base = json.load(f)
        assert base["stage"] == 1 and base["unit"] == BASE_UNIT
        assert base["arm"] == ARM, ("stage 2 must inherit its OWN arm's "
                                    "stage 1", base["arm"], ARM)
        assert not base["any_argmin_at_cap"], "stage 1 was invalid (cap hit)"
        s_mode = n_mode_self / base["n_train_mode"]
        s_full = n_full_self / base["n_train_full"]
        cell_round = {c: int(round(base["cell_argmin_iter"][c] * s_mode))
                      for c in FAIL_CELLS}
        stack_round = int(round(base["stack_argmin_iter"] * s_full))
        res["iter_source"] = (
            f"stage 1 (arm {ARM} unit {BASE_UNIT} seed 42) argmin x training-"
            f"row ratio; cell scale {s_mode:.6f}, stacker scale {s_full:.6f}"
            f" — this unit's own valid curve was NOT consulted")
        res["scale_mode"] = s_mode
        res["scale_full"] = s_full
        res["base_cell_argmin"] = base["cell_argmin_iter"]
        res["base_stack_argmin"] = base["stack_argmin_iter"]
        res["base_json"] = os.path.abspath(args.base_json)
    res["cell_round"] = dict(cell_round)
    res["stack_round"] = int(stack_round)

    # ---------------------------------------- cell models on the valid rows
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: cell models, n_mode "
          f"{n_mode_self:,}, rounds {cell_round}", flush=True)
    assert_no_recovered_in_features(feats_used, "cell models")
    F_mode = build_frame_native(df.loc[tr_mode], feats_used, cat_native)
    F_va = build_frame_native(df.loc[va], feats_used, cat_native)
    n_va = int(va.sum())

    stash, cell_info, cell_curves = {}, {}, {}
    cell_argmin = {}
    for c in FAIL_CELLS:
        k = int(cell_round[c])
        yc_tr = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
        assert not np.isnan(yc_tr).any()
        yc_va = df.loc[va, c].to_numpy(dtype=np.float64)[va_ok]
        t0 = time.time()
        pr, params, note = fit_catboost(C5, SEED, F_mode, yc_tr, F_va,
                                        feats_used, cat_native, k)
        fit_min = (time.time() - t0) / 60
        grid = ref.curve_grid(k)
        marg, const = chunked_margin_curve(pr, n_va, grid)
        cv = {g: float(np.mean((ref.sigmoid(marg[g])[va_ok] - yc_va) ** 2))
              for g in grid}
        a_min, a_val = ref.argmin_of(cv)
        k_use = a_min if STAGE == 1 else k
        cell_argmin[c] = a_min
        stash[c] = ref.sigmoid(marg[k_use])
        cell_info[c] = {
            "n_round": k, "argmin_iter": a_min,
            "argmin_own_brier": float(f"{a_val:.10g}"),
            "argmin_at_cap": bool(STAGE == 1 and a_min >= k),
            "iter_used": int(k_use),
            "own_mode_at_used": ref.seg_metrics(
                yc_va, ref.sigmoid(marg[k_use])[va_ok]),
            "pos_rate_train": float(np.mean(yc_tr)),
            "fit_min": round(fit_min, 3),
            "sec_per_round": round(60.0 * fit_min / k, 5),
            "margin_const": float(const),
        }
        cell_curves[c] = {str(g): float(f"{v:.10g}") for g, v in cv.items()}
        print(f"    cell {c}: fit {fit_min:.2f} min, argmin {a_min} "
              f"({a_val:.7f}), used {k_use}", flush=True)
        del marg, pr
        gc.collect()
    res["cells"] = cell_info
    res["cell_argmin_iter"] = cell_argmin
    del F_mode, F_va
    gc.collect()

    cell_iter_used = {c: cell_info[c]["iter_used"] for c in FAIL_CELLS}
    res["cell_iter_used"] = cell_iter_used

    # ------------------------------------------ forward-chaining OOF by season
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    seasons = sorted(set(season[tr_full].tolist()))
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: OOF seasons {seasons[1:]} at "
          f"{cell_iter_used}", flush=True)
    stack = {c: np.full(len(df), np.nan) for c in stack_cols}
    t_oof = time.time()
    oof_log = []
    for s in seasons[1:]:
        fit_rows = (season < s) & ~f_excl & ok & (succ_season < s)
        pred_rows = (season == s) & tr_full
        assert_time_ordered(tkey, fit_rows, f"OOF fit {s}")
        assert_no_recovered_in_features(feats_used, f"B3 OOF {s}")
        Ff = build_frame_native(df.loc[fit_rows], feats_used, cat_native)
        Fp = build_frame_native(df.loc[pred_rows], feats_used, cat_native)
        for c in FAIL_CELLS:
            k = int(cell_iter_used[c])
            yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
            assert not np.isnan(yc).any()
            pr, _, _ = fit_catboost(C5, SEED, Ff, yc, Fp, feats_used,
                                    cat_native, k)
            stack[f"p_{c}"][pred_rows] = ref.sigmoid(pr(0, k))
            del pr
            gc.collect()
        oof_log.append({"season": int(s), "n_fit": int(fit_rows.sum()),
                        "n_pred": int(pred_rows.sum()),
                        "elapsed_s": round(time.time() - t_oof, 1)})
        print(f"    OOF season {s}: fit {int(fit_rows.sum()):,} -> pred "
              f"{int(pred_rows.sum()):,} ({time.time()-t_oof:.0f}s)", flush=True)
        del Ff, Fp
        gc.collect()
    for c in FAIL_CELLS:
        stack[f"p_{c}"][va] = stash[c]
    res["oof_blocks"] = oof_log
    res["oof_min"] = round((time.time() - t_oof) / 60, 2)
    res["nan_stack_rows_in_train_pct"] = round(
        100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3)
    assert not np.isnan(stack[stack_cols[0]][va]).any(), "valid stack has NaN"

    # ------------------------------------------------------------- stacker
    for c in stack_cols:
        df[c] = stack[c]
    feats = feats_used + stack_cols
    assert_no_recovered_in_features(feats, "B3 stacker")
    assert len(feats) == EXPECTED_N[ARM] + 4, (len(feats), ARM)
    Fs_tr = build_frame_native(df.loc[tr_full], feats, cat_native)
    Fs_va = build_frame_native(df.loc[va], feats, cat_native)
    k = int(stack_round)
    t0 = time.time()
    pr, params, note = fit_catboost(C5, SEED, Fs_tr, y_all[tr_full], Fs_va,
                                    feats, cat_native, k)
    fit_min = (time.time() - t0) / 60
    grid = ref.curve_grid(k)
    marg, const = chunked_margin_curve(pr, n_va, grid)
    cv = {g: float(np.mean((ref.sigmoid(marg[g]) - y_va) ** 2)) for g in grid}
    a_min, a_val = ref.argmin_of(cv)
    k_use = a_min if STAGE == 1 else k
    p = ref.sigmoid(marg[k_use])

    res["stacker"] = {
        "n_round": k, "argmin_iter": a_min,
        "argmin_brier": float(f"{a_val:.10g}"),
        "argmin_at_cap": bool(STAGE == 1 and a_min >= k),
        "curve_argmin_is_interior": bool(a_min < k),
        "iter_used": int(k_use),
        "policy_cost_vs_curve_min": float(f"{cv[k_use] - a_val:.6g}"),
        "fit_min": round(fit_min, 3),
        "sec_per_round": round(60.0 * fit_min / k, 5),
        "margin_const": float(const),
        "params": {kk: (vv if isinstance(vv, (int, float, str, bool,
                                              type(None))) else str(vv))
                   for kk, vv in params.items()},
        "family_note": note,
    }
    res["stack_argmin_iter"] = a_min
    res["any_argmin_at_cap"] = bool(
        res["stacker"]["argmin_at_cap"]
        or any(cell_info[c]["argmin_at_cap"] for c in FAIL_CELLS))

    brier = float(np.mean((p - y_va) ** 2))
    res["brier_full"] = brier
    res["score_converted"] = float(f"{score_of(brier):.5f}")
    res["valid_all"] = ref.seg_metrics(y_va, p)
    res["valid_R"] = ref.seg_metrics(y_va[is_r], p[is_r])
    res["valid_F"] = ref.seg_metrics(y_va[~is_r], p[~is_r])
    res["by_pitcher_hand"] = {
        str(h): ref.seg_metrics(y_va[hand_va == h], p[hand_va == h])
        for h in sorted(pd.unique(hand_va).tolist())}
    res["valid_all_brier_full"] = brier

    # -------- R0 is a reproduction gate against exp_030's stage 1 ----------- #
    if ARM == "R0" and STAGE == 1 and V == 2023 and SEED == 42:
        try:
            with open(EXP030_STAGE1) as f:
                r30 = json.load(f)
            d = abs(brier - float(r30["brier_full"]))
            res["exp030_gate"] = {
                "what": ("arm R0 adds zero columns, so this run is exp_030's "
                         "stage-1 fold_2023 unit recomputed by a different "
                         "file; it must land on the same number"),
                "expected_brier": float(r30["brier_full"]),
                "observed_brier": brier, "abs_diff": d, "tol": 5e-6,
                "PASS": bool(d < 5e-6),
                "expected_cell_argmin": r30["cell_argmin_iter"],
                "observed_cell_argmin": cell_argmin,
                "expected_stack_argmin": r30["stack_argmin_iter"],
                "observed_stack_argmin": a_min,
                "argmins_identical": bool(
                    r30["cell_argmin_iter"] == cell_argmin
                    and r30["stack_argmin_iter"] == a_min),
            }
            print(f"    exp030_gate: ours {brier:.17g} vs exp_030 "
                  f"{float(r30['brier_full']):.17g} diff {d:.3e} "
                  f"{'PASS' if d < 5e-6 else 'FAIL'}", flush=True)
        except Exception as e:                                       # noqa: BLE001
            res["exp030_gate"] = {"PASS": None, "error": repr(e)}

    pred_path = os.path.join(out_dir, "pred", f"pred_{stem}.npy")
    np.save(pred_path, p)
    res["pred_file"] = f"pred/pred_{stem}.npy"
    res["pred_sha256"] = sha256_file(pred_path)
    with open(os.path.join(out_dir, "curves", f"curve_{stem}.json"), "w") as f:
        json.dump({"arm": ARM, "unit": V, "seed": SEED, "stage": STAGE,
                   "stacker_grid": grid,
                   "stacker_valid_all_brier": {str(g): float(f"{v:.10g}")
                                               for g, v in cv.items()},
                   "stacker_argmin_iter": a_min,
                   "stacker_argmin_brier": res["stacker"]["argmin_brier"],
                   "cell_own_target_brier": cell_curves,
                   "cell_argmin_iter": cell_argmin}, f)
    res["curve_file"] = f"curves/curve_{stem}.json"
    res["minutes"] = round((time.time() - t_start) / 60, 2)

    with open(os.path.join(out_dir, "res", f"res_{stem}.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("RESULT: " + json.dumps({
        "arm": ARM, "unit": V, "seed": SEED, "stage": STAGE,
        "n_feats": len(feats_used),
        "cell_argmin": cell_argmin, "cell_iter_used": cell_iter_used,
        "stack_argmin": a_min, "stack_iter_used": k_use,
        "any_argmin_at_cap": res["any_argmin_at_cap"],
        "brier": brier, "score": res["score_converted"],
        "minutes": res["minutes"]}), flush=True)
    if STAGE == 1 and res["any_argmin_at_cap"]:
        print("!! STAGE 1 INVALID: an argmin reached the cap. Re-run larger.",
              flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
