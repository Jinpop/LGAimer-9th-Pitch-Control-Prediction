"""exp_030 — the exp_016 B3 mode-stacking architecture rebuilt on CatBoost C5.

ONE PROCESS = ONE UNIT = one (block year, seed, stage).

WHAT IS INHERITED, AND FROM WHERE (sha-gated at run time, not by memory)
-----------------------------------------------------------------------
  experiments/exp_024/src/labels.py       439cd504...a9f2   byte-identical copy
      the recovered per-row mode labels and the machine-checked leak boundary
      `assert_no_recovered_in_features`.
  experiments/exp_024/src/exp016_ref.py   4f24deb7...3ece1   byte-identical copy
      curve_grid · sigmoid · nearest_grid_iter · argmin_of · seg_metrics ·
      lb_score · sha256_file.  (Its LightGBM fit path is NOT used here.)
  experiments/exp_027/src/train.py        e4b308a5...16f0f   IMPORTED, not copied
      CB_BASE · ARMS["C5"] · build_frame_native · chunked_margin_curve ·
      fit_catboost.  The module is loaded by path and its sha256 is asserted
      before anything is read out of it, so "we used exp_027's C5" is a
      checked claim rather than a transcription that could drift.
  experiments/exp_024/src/b3_block.py     cd5c75ee...42b6   ARCHITECTURE source
      the B3 body below (stash -> forward-chaining OOF by season -> stacker)
      is that file's control flow with the LightGBM calls replaced by the
      CatBoost adapter.  Masks, pools and row counts are unchanged.

WHAT IS NEW (the full variable ledger for this experiment)
----------------------------------------------------------
  1. model family of BOTH the four cell models and the stacker:
     LightGBM exp_010-D params  ->  CatBoost C5 (has_time=True, IDs numeric,
     depth 6, l2_leaf_reg 10, lr 0.03, Plain, Bernoulli 0.8, one_hot_max_size
     10, CPU, 6 threads).  Cells and stacker change TOGETHER, so this
     experiment cannot attribute the result to one or the other.
  2. iteration counts.  exp_016/exp_024's 240/1150/170/650 came out of
     LightGBM's learning rate and do not transfer.  Stage 1 measures them on
     fold_2023 (train <=2022 -> valid 2023) with early stopping under a cap;
     Stage 2 fixes them by the training-row ratio of the unit being run and
     never looks at that unit's own validation curve to choose.
  3. the cell iteration is now PER CELL (four numbers).  exp_016 used one
     shared cell iteration taken from arm B2c's argmin.  Here each cell model
     early-stops on its OWN binary target, which is what "each unit stops on
     its own objective" means.  This is a deliberate change and is listed
     because an unlisted change is the failure mode the project bans.

ITERATION POLICY  (fold_2024's validation set never selects anything)
---------------------------------------------------------------------
  Stage 1 : unit 2023, seed 42, --cap C.  Every cell model and the stacker is
            trained to C and reported at its own argmin over curve_grid(C).
            If an argmin lands on C the unit is INVALID (`argmin_at_cap`) and
            must be re-run at a larger cap.
  Stage 2 : iterations come from Stage 1 scaled by the ratio of TRAINING ROWS
            in the pool that actually fits the model —
                cell k   = round(k_base_cell   * n_mode_self / n_mode_base)
                stacker  = round(k_base_stack  * n_full_self / n_full_base)
            The validation curve is still computed, but only as a diagnostic
            (`argmin_iter` / `curve_argmin_is_interior`); `iter_used` is the
            fixed number.

LEAK BOUNDARY
-------------
  * recovered `is_*` columns are TARGETS ONLY; `assert_no_recovered_in_features`
    fires before every single design matrix is built.
  * the stacker's four extra columns are out-of-fold and forward-chaining by
    season: rows of season s are predicted by cell models fitted on seasons
    < s only (exp_016 design decision D4).  No random KFold, no train-segment
    self-prediction.  The earliest season keeps NaN — CatBoost handles NaN in
    a float feature natively; a sentinel fill is banned.
  * `has_time=True` uses the pool's row order as its time axis, so every fit
    pool is re-checked at run time for (season, game_month) inversions.
  * nothing here reads the test file or any other row's prediction.

Usage:
    python b3_cat.py --unit 2023 --seed 42 --stage 1 --cap 1650
    python b3_cat.py --unit 2024 --seed 7  --stage 2 --base-json ../stage1.json
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

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")
EXP027_TRAIN = os.path.expanduser(
    "~/SungjinPark/experiments/exp_027/src/train.py")
EXP027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"
REF016_SHA = "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38a3ece1"

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]        # C5: NUMERIC, never cat_features
F_EXCLUDE_MAX_SEASON = 2022                  # exp_004 mask
N_THREADS = 6
REF_BRIER = 0.2498069                        # 2025-test conversion reference

# exp_004 masks and fold definitions, byte-equal in value to exp_024 BLOCK_SPEC
UNIT_SPEC = {
    2022: {"train_upto": 2021, "n_train": 653728, "n_valid": 247472},
    2023: {"train_upto": 2022, "n_train": 870752, "n_valid": 245525},
    2024: {"train_upto": 2023, "n_train": 1116277, "n_valid": 253507,
           "n_valid_R": 223497},
}
BASE_UNIT = 2023                             # the unit Stage 1 runs on


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
    """Import exp_027/src/train.py by path, but only after its sha matches."""
    got = sha256_file(EXP027_TRAIN)
    assert got == EXP027_SHA, ("exp_027 train.py changed", got, EXP027_SHA)
    spec = importlib.util.spec_from_file_location("exp027_train", EXP027_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # main() is __main__-guarded
    c5 = mod.ARMS["C5"]
    assert c5["family"] == "catboost", c5
    assert c5["params"] == {}, c5                    # C5 == CB_BASE exactly
    assert c5["cat_ids"] is False, c5                # IDs stay numeric
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
    ap.add_argument("--unit", type=int, required=True, choices=[2022, 2023, 2024])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--cap", type=int, default=1650)
    ap.add_argument("--base-json", default="",
                    help="stage 2: path to the stage-1 result json")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    V, SEED, STAGE = args.unit, args.seed, args.stage
    spec_u = UNIT_SPEC[V]
    tu, vy = spec_u["train_upto"], V
    out_dir = args.out
    t_start = time.time()
    np.random.seed(SEED)
    for d in ("", "pred", "curves", "res"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA
    e27, C5 = load_exp027()
    build_frame_native = e27.build_frame_native
    chunked_margin_curve = e27.chunked_margin_curve
    fit_catboost = e27.fit_catboost

    tag = f"u{V}_s{SEED}_st{STAGE}"
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: unit {V} (train<={tu} -> "
          f"valid {vy}) seed {SEED} stage {STAGE} cap {args.cap}", flush=True)

    # ------------------------------------------------------------------ data
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

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

    cat_native = [c for c in base_feats if c in CAT_COLS]
    assert len(cat_native) == 3, cat_native
    assert not any(c in cat_native for c in ID_COLS), "C5 keeps IDs numeric"

    n_mode_self = int(tr_mode.sum())
    n_full_self = int(tr_full.sum())

    res = {
        "exp_id": "exp_030", "unit": V, "seed": SEED, "stage": STAGE,
        "arm": "C5_B3", "train_upto": tu, "valid_season": vy,
        "n_train_full": n_full_self, "n_train_mode": n_mode_self,
        "n_valid": int(va.sum()), "n_valid_R": int(is_r.sum()),
        "n_valid_F": int((~is_r).sum()),
        "n_rows_masked_exp004": int(f_excl.sum()),
        "cap": args.cap,
        "cat_features": list(cat_native),
        "id_cols_numeric": list(ID_COLS),
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "b3_cat.py": sha256_file(os.path.abspath(__file__)),
            "labels.py": LABELS_SHA, "exp016_ref.py": REF016_SHA,
            "exp_027_train.py": EXP027_SHA,
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "time_inversions": {},
    }
    import catboost as cb                                            # noqa: F401
    res["versions"]["catboost"] = cb.__version__
    res["time_inversions"]["tr_full"] = assert_time_ordered(tkey, tr_full, "tr_full")
    res["time_inversions"]["tr_mode"] = assert_time_ordered(tkey, tr_mode, "tr_mode")

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
        assert not base["any_argmin_at_cap"], "stage 1 was invalid (cap hit)"
        s_mode = n_mode_self / base["n_train_mode"]
        s_full = n_full_self / base["n_train_full"]
        cell_round = {c: int(round(base["cell_argmin_iter"][c] * s_mode))
                      for c in FAIL_CELLS}
        stack_round = int(round(base["stack_argmin_iter"] * s_full))
        res["iter_source"] = (
            f"stage 1 (unit {BASE_UNIT} seed 42) argmin x training-row ratio; "
            f"cell scale {s_mode:.6f} = {n_mode_self}/{base['n_train_mode']}, "
            f"stacker scale {s_full:.6f} = {n_full_self}/{base['n_train_full']}"
            f" — this unit's own valid curve was NOT consulted")
        res["scale_mode"] = s_mode
        res["scale_full"] = s_full
        res["base_cell_argmin"] = base["cell_argmin_iter"]
        res["base_stack_argmin"] = base["stack_argmin_iter"]
    res["cell_round"] = dict(cell_round)
    res["stack_round"] = int(stack_round)

    # ---------------------------------------- cell models on the valid rows
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: cell models, n_mode "
          f"{n_mode_self:,}, rounds {cell_round}", flush=True)
    assert_no_recovered_in_features(base_feats, "cell models")
    F_mode = build_frame_native(df.loc[tr_mode], base_feats, cat_native)
    F_va = build_frame_native(df.loc[va], base_feats, cat_native)
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
                                        base_feats, cat_native, k)
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
            "own_mode_at_used": ref.seg_metrics(yc_va,
                                                ref.sigmoid(marg[k_use])[va_ok]),
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
        assert_no_recovered_in_features(base_feats, f"B3 OOF {s}")
        Ff = build_frame_native(df.loc[fit_rows], base_feats, cat_native)
        Fp = build_frame_native(df.loc[pred_rows], base_feats, cat_native)
        for c in FAIL_CELLS:
            k = int(cell_iter_used[c])
            yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
            assert not np.isnan(yc).any()
            pr, _, _ = fit_catboost(C5, SEED, Ff, yc, Fp, base_feats,
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
    feats = base_feats + stack_cols
    assert_no_recovered_in_features(feats, "B3 stacker")
    assert len(feats) == 51, len(feats)
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
        "params": {kk: (vv if isinstance(vv, (int, float, str, bool, type(None)))
                        else str(vv)) for kk, vv in params.items()},
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

    stem = f"C5B3_u{V}_seed{SEED}_st{STAGE}"
    pred_path = os.path.join(out_dir, "pred", f"pred_{stem}.npy")
    np.save(pred_path, p)
    res["pred_file"] = f"pred/pred_{stem}.npy"
    res["pred_sha256"] = sha256_file(pred_path)
    with open(os.path.join(out_dir, "curves", f"curve_{stem}.json"), "w") as f:
        json.dump({"unit": V, "seed": SEED, "stage": STAGE,
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
    if STAGE == 1:
        with open(os.path.join(out_dir, "stage1.json"), "w") as f:
            json.dump(res, f, indent=1)

    print("RESULT: " + json.dumps({
        "unit": V, "seed": SEED, "stage": STAGE,
        "cell_argmin": cell_argmin, "cell_iter_used": cell_iter_used,
        "stack_argmin": a_min, "stack_iter_used": k_use,
        "any_argmin_at_cap": res["any_argmin_at_cap"],
        "brier": brier, "score": res["score_converted"],
        "minutes": res["minutes"]}), flush=True)
    if STAGE == 1 and res["any_argmin_at_cap"]:
        print("!! STAGE 1 INVALID: an argmin reached the cap. Re-run larger.",
              flush=True)


if __name__ == "__main__":
    main()
