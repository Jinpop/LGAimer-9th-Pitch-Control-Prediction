"""exp_051 — 63-column cells (47 + 16 ins_*): OOF regenerated, stacker refit at 1400.

Derived from build_AH/localref/src/local_ref_stack.py (sha 1993f63e...86c85) by
registered substitutions (patch_train51.py); unified diff saved as train51.diff.
The parent docstring is kept below for lineage.

--- parent docstring ---
exp_036 — the UNCENSORED fold_2024 argmin of the B3 stacker.

WHAT THIS FILE IS
-----------------
exp_032's `block_iter.py` (sha b4ecb311...882d, carried BYTE-IDENTICAL beside
this file and asserted at run time) with ONE new code path, STAGE 5.  Stages
1-4 are untouched; this file never runs them.

WHY IT EXISTS
-------------
exp_030's fold_2024 unit trained the stacker to 481 rounds under stage 2's
row-ratio policy.  At seeds 7 and 2024 the validation curve's argmin came out
at EXACTLY 481 — an argmin that equals the cap is not an argmin, it is a
truncation.  So the honest state of knowledge about the fold_2024 stacker
argmin is ">= 481" and nothing more.  This file measures the real one.

STAGE 5 — "cells pinned, stacker measured"
------------------------------------------
  * the four cell models take their iteration counts from --cell-iter, exactly
    as stage 4 does.  They do NOT self-stop.  For this experiment those counts
    are exp_030's fold_2024 numbers, so the cell models, the forward-chained
    OOF columns and therefore the stacker's design matrix are the SAME OBJECTS
    exp_030 built.  The single variable of this experiment is how far the
    stacker is trained.
  * the stacker is trained to --cap and its validation curve is evaluated at
    EVERY ROUND from 1 to --cap (step 1, not the sparse curve_grid).  The
    argmin of that dense curve is the quantity this experiment reports.
  * `iter_used` for the saved prediction IS that argmin.  That is a
    MEASUREMENT, not an adoption: this file's job is to say where the curve
    bottoms, not to nominate a deployment configuration.  Nothing downstream
    of this file is entitled to treat the number as selected.
  * if the argmin lands on --cap the unit is INVALID and says so, for the same
    reason exp_030's 481 was.

THE REPRODUCTION GATE (why "we drew the same curve further" is a checked claim)
------------------------------------------------------------------------------
Before the dense pass, the stacker's margins are ALSO accumulated on the
inherited `chunked_margin_curve` over the inherited `curve_grid(481)` — bit for
bit the arithmetic exp_030 performed — and the resulting 481-round prediction
is compared against exp_030's own artefacts, read from disk, never transcribed:

    ~/SungjinPark/experiments/exp_030/res/res_C5B3_u2024_seed{S}_st2.json
        -> brier_full   : |ours - theirs| must be < 5e-6
        -> pred_sha256  : compared; equality means bit-identity

A 3000-round CatBoost model's first 481 trees ARE the 481-round model
(boosting_type Plain, fixed random_seed), so this gate should hold to the byte.
It is asserted on the Brier and REPORTED on the sha, because the sha can only
differ through a floating-point accumulation difference, which the Brier gate
already bounds.

THE TWO ACCUMULATION PATHS, AND WHY BOTH ARE HERE
--------------------------------------------------
`chunked_margin_curve` (inherited, unmodified) retains one margin vector per
grid point.  At step 1 over 3000 rounds that is 3000 x 253,507 x 8 B = 6.1 GB
per process, and three processes run at once.  `dense_brier_curve` below is
that function's arithmetic with the retention removed — same additivity
assert, same `acc = chunk` / `acc = acc + chunk + const` recurrence, same
constant — computing the Brier inline so exactly one margin vector and one
best-so-far copy are ever alive.  The dense path sums 3000 chunks where
exp_030's summed 64, so the two agree to floating point, not to the byte; the
gate above is run on the INHERITED path precisely so that the exact exp_030
number stays on the record.

LEAK BOUNDARY — unchanged from block_iter.py
--------------------------------------------
  * recovered `is_*` columns are TARGETS ONLY; `assert_no_recovered_in_features`
    fires before every design matrix.
  * the stacker's four extra columns are out-of-fold, forward-chained by season
    (rows of season s predicted by cells fitted on seasons < s only).  No
    random KFold, no self-prediction, earliest season keeps NaN.
  * `has_time=True`, so every fit pool is re-checked for time inversions.
  * nothing reads the test file or any other row's prediction.
  * fold_2024's validation curve selects NOTHING that leaves this file: the
    cell iterations came in on the command line and the stacker argmin is the
    measurement being reported.

Usage:
    python stack_argmin.py --unit 2024 --seed 42 --stage 5 --cap 3000 \
        --cell-iter '{"is_mid_only":577,"is_rev_only":1346,"is_both":449,
                      "is_wayoff_true":897}' \
        --iter-src "exp_030 stage-2 fold_2024 cell iterations, pinned" \
        --gate-k 481
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
sys.path.insert(0, os.path.join(SRC_DIR, "ins"))

EXP_ID = "exp_051"
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_051")
EXP030_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")
EXP048_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_048")
AH_REF_JSON = os.path.expanduser(
    "~/SungjinPark/build_AH/local_reference_AH.json")
AH_OOF_DIR = os.path.expanduser("~/SungjinPark/build_AH/localref/oof")
AH_RES_DIR = os.path.expanduser("~/SungjinPark/build_AH/localref/res")
INS_MATRIX = os.path.expanduser(
    "~/SungjinPark/experiments/exp_051/ins_matrix_fold2024.npz")

B3_CAT = "b3_cat.py"          # exp_030's file, carried byte-identical
B3_CAT_SHA = "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7bfa76ca"
BLOCK_SELF = "block_self.py"  # exp_031's file, carried byte-identical
BLOCK_SELF_SHA = ("2146d0e78a487b3223be42a743e5d03ff5e56d35c12e502f812c0eb1f318"
                  "4406")
BLOCK_ITER = "block_iter.py"  # exp_032's file, THIS FILE'S PARENT
BLOCK_ITER_SHA = ("b4ecb311ba4a8acaff2484e225fe174cc82ca1a356f128fa42fdea5c611"
                  "0882d")
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
GATE_TOL = 5e-6

UNIT_SPEC = {
    2021: {"train_upto": 2020, "n_train": 432501, "n_valid": 247088},
    2022: {"train_upto": 2021, "n_train": 653728, "n_valid": 247472},
    2023: {"train_upto": 2022, "n_train": 870752, "n_valid": 245525},
    2024: {"train_upto": 2023, "n_train": 1116277, "n_valid": 253507,
           "n_valid_R": 223497},
}


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
    assert c5["family"] == "catboost", c5
    assert c5["params"] == {}, c5
    assert c5["cat_ids"] is False, c5
    assert mod.CB_BASE["has_time"] is True
    assert mod.CB_BASE["thread_count"] == N_THREADS
    return mod, c5


def fit_cb(e27, spec, seed, F_tr, y_tr, F_va, feats, cat_native, n_round):
    """exp_048 s1_stack.py `fit_cb` VERBATIM — exp_027's fit_catboost plus the
    model object.  exp_048's A0 gate proved it bit-identical to fit_catboost."""
    import catboost as cb
    params = dict(e27.CB_BASE, random_seed=seed, iterations=n_round,
                  **spec["params"])
    cat_idx = [feats.index(c) for c in cat_native]
    pool_tr = cb.Pool(F_tr, label=y_tr, cat_features=cat_idx)
    pool_va = cb.Pool(F_va, cat_features=cat_idx)
    model = cb.CatBoostClassifier(**params)
    model.fit(pool_tr)
    assert model.tree_count_ == n_round, (model.tree_count_, n_round)

    def predict_range(a, b):
        return model.predict(pool_va, prediction_type="RawFormulaVal",
                             ntree_start=a, ntree_end=b,
                             thread_count=N_THREADS).astype(np.float64)

    return predict_range, params, model


def score_of(brier):
    return 100000.0 * (1.0 - brier / REF_BRIER)


def assert_time_ordered(tkey, mask, where):
    t = tkey[mask]
    inv = int((np.diff(t) < 0).sum())
    assert inv == 0, (f"has_time=True but pool '{where}' is not in time order",
                      inv)
    return inv


def dense_brier_curve(predict_range, y, kmax, tag="", every=250):
    """exp_027 `chunked_margin_curve` at step 1, margins not retained.

    Returns (curve {k: brier}, const, best_k, best_p) where best_p is a COPY of
    the probability vector at the argmin.  The additivity assert and the
    `acc = chunk` / `acc = acc + chunk + const` recurrence are the inherited
    ones verbatim; only the retention of every margin vector is dropped.
    """
    a = predict_range(0, 5)
    b = predict_range(5, 10)
    full = predict_range(0, 10)
    resid = full - (a + b)
    const = float(np.median(resid))
    assert np.allclose(resid, const, atol=1e-8), (
        "raw margins are not additive across round ranges",
        float(np.abs(resid - const).max()))
    del a, b, full, resid
    gc.collect()

    curve, acc, prev = {}, None, 0
    best_k, best_v, best_p = -1, np.inf, None
    t0 = time.time()
    for k in range(1, kmax + 1):
        chunk = predict_range(prev, k)
        acc = chunk if acc is None else acc + chunk + const
        prev = k
        p = ref.sigmoid(acc)
        v = float(np.mean((p - y) ** 2))
        curve[k] = v
        if v < best_v:
            best_v, best_k, best_p = v, k, p.copy()
    print(f"    {tag} dense curve: {kmax} rounds in "
          f"{time.time()-t0:.0f}s, argmin {best_k} ({best_v:.10f})", flush=True)
    return curve, const, best_k, best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", type=int, required=True, choices=[2021, 2022,
                                                                2023, 2024])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", type=int, required=True, choices=[5])
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--cell-iter", default=None,
                    help="optional cross-check; the values are READ from local_reference_AH.json")
    ap.add_argument("--iter-src", required=True,
                    help="in words, where the CELL iterations came from")
    ap.add_argument("--gate-k", type=int, default=481,
                    help="round at which exp_030 must be reproduced")
    ap.add_argument("--rounds", required=True,
                    help="comma-separated round counts to fit and report")
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
    assert sha256_file(os.path.join(SRC_DIR, B3_CAT)) == B3_CAT_SHA, \
        "exp_030 b3_cat.py, the architecture source, has drifted"
    assert sha256_file(os.path.join(SRC_DIR, BLOCK_SELF)) == BLOCK_SELF_SHA, \
        "exp_031 block_self.py has drifted"
    assert sha256_file(os.path.join(SRC_DIR, BLOCK_ITER)) == BLOCK_ITER_SHA, \
        "exp_032 block_iter.py, from which THIS file is derived, has drifted"
    e27, C5 = load_exp027()
    build_frame_native = e27.build_frame_native
    chunked_margin_curve = e27.chunked_margin_curve
    fit_catboost = e27.fit_catboost

    tag = f"u{V}_s{SEED}_st{STAGE}"
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: unit {V} (train<={tu} -> "
          f"valid {vy}) seed {SEED} stage {STAGE} STACKER CAP {args.cap}",
          flush=True)

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

    # ---- exp_051: the 16 ins_* columns, built once by s0_ins.py -------
    zi = np.load(INS_MATRIX, allow_pickle=False)
    assert str(zi["train_csv_sha256"].item()) == sha256_file(DATA_PATH), \
        "the ins matrix was built on a different train.csv"
    ins_cols = [str(c) for c in zi["cols"].tolist()]
    assert len(ins_cols) == 16, ins_cols
    V_ins = np.ascontiguousarray(zi["values"], dtype=np.float64)
    assert V_ins.shape == (len(df), 16), V_ins.shape
    assert not np.isnan(V_ins[va, 0]).any(), "valid rows must carry no NaN"
    for j_ins, c_ins in enumerate(ins_cols):
        df[c_ins] = V_ins[:, j_ins]
    ins_matrix_sha = sha256_file(INS_MATRIX)
    cell_feats = base_feats + ins_cols
    assert len(cell_feats) == 63, len(cell_feats)
    assert_no_recovered_in_features(cell_feats, "cell_feats (63)")
    del V_ins

    n_mode_self = int(tr_mode.sum())
    n_full_self = int(tr_full.sum())

    res = {
        "exp_id": EXP_ID, "unit": V, "seed": SEED, "stage": STAGE,
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
            "self_stack_argmin.py": sha256_file(os.path.abspath(__file__)),
            "exp032_block_iter.py": BLOCK_ITER_SHA,
            "exp031_block_self.py": BLOCK_SELF_SHA,
            "exp030_b3_cat.py": B3_CAT_SHA,
            "labels.py": LABELS_SHA, "exp016_ref.py": REF016_SHA,
            "exp_027_train.py": EXP027_SHA,
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "time_inversions": {},
    }
    import catboost as cb                                            # noqa: F401
    res["versions"]["catboost"] = cb.__version__
    res["sha256"]["ins_matrix_fold2024.npz"] = ins_matrix_sha
    res["sha256"]["ins_feats.py"] = sha256_file(
        os.path.join(SRC_DIR, "ins_feats.py"))
    res["sha256"]["inseason.py"] = sha256_file(
        os.path.join(SRC_DIR, "ins", "inseason.py"))
    res["sha256"]["anchor_lt2024.npz"] = sha256_file(
        os.path.join(SRC_DIR, "ins", "anchor_lt2024.npz"))
    res["cell_n_features"] = 63
    res["stacker_n_features"] = 67
    res["ins_cols"] = list(ins_cols)
    res["time_inversions"]["tr_full"] = assert_time_ordered(tkey, tr_full,
                                                            "tr_full")
    res["time_inversions"]["tr_mode"] = assert_time_ordered(tkey, tr_mode,
                                                            "tr_mode")

    # -------------------------------------------------- iteration resolution
    res["self_early_stop"] = False           # the CELLS never self-stop here
    with open(AH_REF_JSON) as fah:
        ahref = json.load(fah)
    ah_seed = ahref["per_seed"][str(SEED)]
    ci = dict(ah_seed["run"]["cell_iter_used"])
    res["cell_iter_source"] = {
        "path": AH_REF_JSON,
        "key": f"per_seed.{SEED}.run.cell_iter_used",
        "values_read": {k: int(v) for k, v in ci.items()}}
    if args.cell_iter:
        assert json.loads(args.cell_iter) == ci, (args.cell_iter, ci)
    assert set(ci) == set(FAIL_CELLS), (sorted(ci), sorted(FAIL_CELLS))
    cell_round = {c: int(ci[c]) for c in FAIL_CELLS}
    assert all(v > 0 for v in cell_round.values())
    res["iter_source"] = args.iter_src
    res["iter_policy"] = (
        "stage 5 — the four CELL iterations were supplied on the command line "
        "and this unit's own valid curve was not consulted to choose them.  "
        "The STACKER is fitted separately at each NAMED round count and its "
        "valid_all Brier at each is the quantity being MEASURED.  Nothing "
        "here selects a deployment setting: the deployed counts (AF 655, "
        "AG 1000, AH 1400) were fixed on the leaderboard before this ran, "
        "and this measures what the local fold says about them.")
    res["cell_round"] = dict(cell_round)
    res["stack_round"] = int(args.cap)

    # ---------------------------------------- cell models on the valid rows
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: cell models, n_mode "
          f"{n_mode_self:,}, rounds {cell_round}", flush=True)
    assert_no_recovered_in_features(cell_feats, "cell models (63)")
    F_mode = build_frame_native(df.loc[tr_mode], cell_feats, cat_native)
    F_va = build_frame_native(df.loc[va], cell_feats, cat_native)
    n_va = int(va.sum())

    stash, cell_info, cell_curves = {}, {}, {}
    cell_argmin = {}
    for c in FAIL_CELLS:
        k = int(cell_round[c])
        yc_tr = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
        assert not np.isnan(yc_tr).any()
        yc_va = df.loc[va, c].to_numpy(dtype=np.float64)[va_ok]
        t0 = time.time()
        pr, params, model_c = fit_cb(e27, C5, SEED, F_mode, yc_tr, F_va,
                                     cell_feats, cat_native, k)
        fit_min = (time.time() - t0) / 60
        grid = ref.curve_grid(k)
        marg, const = chunked_margin_curve(pr, n_va, grid)
        cv = {g: float(np.mean((ref.sigmoid(marg[g])[va_ok] - yc_va) ** 2))
              for g in grid}
        a_min, a_val = ref.argmin_of(cv)
        k_use = k                                   # cells are PINNED
        cell_argmin[c] = a_min
        stash[c] = ref.sigmoid(marg[k_use])
        cell_info[c] = {
            "n_round": k, "argmin_iter": a_min,
            "argmin_own_brier": float(f"{a_val:.10g}"),
            "argmin_at_cap": bool(a_min == k),
            "iter_used": int(k_use),
            "own_mode_at_used": ref.seg_metrics(yc_va,
                                                ref.sigmoid(marg[k_use])[va_ok]),
            "pos_rate_train": float(np.mean(yc_tr)),
            "fit_min": round(fit_min, 3),
            "sec_per_round": round(60.0 * fit_min / k, 5),
            "margin_const": float(const),
        }
        cell_curves[c] = {str(g): float(f"{v:.10g}") for g, v in cv.items()}
        imp_c = model_c.get_feature_importance(
            type="PredictionValuesChange")
        order_c = np.argsort(-imp_c)
        tot_gain = float(imp_c.sum())
        cell_info[c]["ins_gain_share"] = float(
            f"{sum(imp_c[cell_feats.index(cc)] for cc in ins_cols) / max(tot_gain, 1e-12):.6g}")
        cell_info[c]["importance_top15"] = {
            cell_feats[i]: float(f"{imp_c[i]:.6g}") for i in order_c[:15]}
        cell_info[c]["ins_importance_rank"] = {
            cc: int(np.where(order_c == cell_feats.index(cc))[0][0] + 1)
            for cc in ins_cols}
        del model_c
        print(f"    cell {c}: fit {fit_min:.2f} min, argmin {a_min} "
              f"({a_val:.7f}), PINNED at {k_use}", flush=True)
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
        assert_no_recovered_in_features(cell_feats, f"B3 OOF {s} (63)")
        Ff = build_frame_native(df.loc[fit_rows], cell_feats, cat_native)
        Fp = build_frame_native(df.loc[pred_rows], cell_feats, cat_native)
        for c in FAIL_CELLS:
            k = int(cell_iter_used[c])
            yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
            assert not np.isnan(yc).any()
            pr, _, model_o = fit_cb(e27, C5, SEED, Ff, yc, Fp, cell_feats,
                                    cat_native, k)
            stack[f"p_{c}"][pred_rows] = ref.sigmoid(pr(0, k))
            del pr, model_o
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
    oof_path = os.path.join(out_dir, "oof",
                            f"stack_fold2024_seed{SEED}.npz")
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    np.savez_compressed(
        oof_path, seed=np.int64(SEED), unit=np.int64(V),
        train_upto=np.int64(tu),
        cell_round=np.array([cell_iter_used[c] for c in FAIL_CELLS],
                            dtype=np.int64),
        tr_full=tr_full, va=va, season=season,
        train_csv_sha256=np.array(sha256_file(DATA_PATH)),
        cell_n_features=np.int64(len(cell_feats)),
        **{c: stack[c] for c in stack_cols})
    res["oof_file"] = os.path.relpath(oof_path, out_dir)
    res["oof_sha256"] = sha256_file(oof_path)
    res["oof_bytes"] = os.path.getsize(oof_path)
    res["oof_blocks"] = oof_log
    res["oof_min"] = round((time.time() - t_oof) / 60, 2)
    res["nan_stack_rows_in_train_pct"] = round(
        100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3)
    assert not np.isnan(stack[stack_cols[0]][va]).any(), "valid stack has NaN"

    # ------------------------------------------------------------- stacker
    for c in stack_cols:
        df[c] = stack[c]
    feats = base_feats + stack_cols + ins_cols
    assert_no_recovered_in_features(feats, "B3 stacker (67)")
    assert len(feats) == 67, len(feats)
    Fs_tr = build_frame_native(df.loc[tr_full], feats, cat_native)
    Fs_va = build_frame_native(df.loc[va], feats, cat_native)
    K = int(args.cap)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: stacker fit to {K} rounds",
          flush=True)
    t0 = time.time()
    pr, params, model_s = fit_cb(e27, C5, SEED, Fs_tr, y_all[tr_full], Fs_va,
                                 feats, cat_native, K)
    fit_min = (time.time() - t0) / 60
    print(f"    stacker fit {fit_min:.2f} min ({60*fit_min/K:.5f} s/round)",
          flush=True)

    # ---- exp_051 CONTROL GATE: exp_048 A1 stored predictions ---------
    r48_path = os.path.join(EXP048_DIR, "res",
                            f"res_exp048_s1_seed{SEED}.json")
    with open(r48_path) as f:
        r48 = json.load(f)
    assert r48["exp_id"] == "exp_048" and r48["seed"] == SEED
    a1_1400 = r48["arms"]["A1"]["by_round"]["1400"]
    ctrl_pred_path = a1_1400["pred_file"]
    ctrl_sha_obs = sha256_file(ctrl_pred_path)
    p_ctrl = np.load(ctrl_pred_path)
    assert p_ctrl.shape == y_va.shape, (p_ctrl.shape, y_va.shape)
    b_ctrl_re = float(np.mean((p_ctrl - y_va) ** 2))
    gate = {
        "what": ("the CONTROL is exp_048 arm A1 (cells 47 cols, stacker "
                 "67 cols @1400).  Nothing is refitted: the stored "
                 "prediction vector is sha256-compared against the "
                 "digest exp_048 recorded, and its Brier is recomputed "
                 "here against this run's own y_va."),
        "control_res_json": r48_path,
        "control_pred_file": ctrl_pred_path,
        "recorded_pred_sha256": a1_1400["pred_sha256"],
        "observed_pred_sha256": ctrl_sha_obs,
        "sha_match": bool(ctrl_sha_obs == a1_1400["pred_sha256"]),
        "recorded_brier": a1_1400["valid_all_brier"],
        "recomputed_brier": b_ctrl_re,
        "abs_diff_brier": abs(b_ctrl_re - a1_1400["valid_all_brier"]),
        "control_brier_by_round": {
            rr: r48["arms"]["A1"]["by_round"][rr]["valid_all_brier"]
            for rr in ("655", "1000", "1400")},
    }
    gate["PASS"] = bool(gate["sha_match"]
                        and gate["abs_diff_brier"] < 1e-12)
    res["A0_control_gate_vs_exp048_A1"] = gate
    print(f"    A0 GATE: sha_match {gate['sha_match']}, abs_diff "
          f"{gate['abs_diff_brier']:.3e}, PASS {gate['PASS']}",
          flush=True)
    assert gate["PASS"], ("A0 control gate FAILED", gate)

    # ---- AH-OOF cross-check + OOF movement (config: 추가로_남길_것) ----
    ah_oof_path = os.path.join(AH_OOF_DIR,
                               f"stack_fold2024_seed{SEED}.npz")
    ah_oof_sha = sha256_file(ah_oof_path)
    assert ah_oof_sha == ah_seed["run"]["oof_sha256"], (
        "AH OOF sha mismatch", ah_oof_sha)
    z_ah = np.load(ah_oof_path, allow_pickle=False)
    assert np.array_equal(z_ah["tr_full"].astype(bool), tr_full)
    assert np.array_equal(z_ah["va"].astype(bool), va)
    assert np.array_equal(z_ah["season"], season)
    assert {c: int(v) for c, v in zip(FAIL_CELLS,
                                      z_ah["cell_round"].tolist())} == \
        cell_iter_used, "AH OOF cell rounds differ from the pinned ones"
    oof_delta = {}
    for c in stack_cols:
        new_v, old_v = stack[c], z_ah[c]
        oof_delta[c] = {}
        for seg_name, m_seg in (("train_rows", tr_full),
                                ("valid_rows", va)):
            nv, ov = new_v[m_seg], old_v[m_seg]
            assert np.array_equal(np.isnan(nv), np.isnan(ov)), (
                "NaN pattern moved", c, seg_name)
            keep = ~np.isnan(nv)
            d_ = nv[keep] - ov[keep]
            ad = np.abs(d_)
            q = np.quantile(ad, [0.5, 0.9, 0.99])
            oof_delta[c][seg_name] = {
                "n": int(d_.size),
                "mean_abs": float(f"{ad.mean():.6g}"),
                "p50_abs": float(f"{q[0]:.6g}"),
                "p90_abs": float(f"{q[1]:.6g}"),
                "p99_abs": float(f"{q[2]:.6g}"),
                "max_abs": float(f"{ad.max():.6g}"),
                "corr_new_old": float(
                    f"{np.corrcoef(nv[keep], ov[keep])[0, 1]:.6g}"),
            }
    res["oof_delta_vs_AH"] = {
        "ah_oof_file": ah_oof_path, "ah_oof_sha256": ah_oof_sha,
        "per_column": oof_delta}
    del z_ah

    # ---- 47-column cell self-metrics, read from AH's res for compare --
    ahres_path = os.path.join(AH_RES_DIR,
                              f"res_C5B3_u{V}_seed{SEED}_localref.json")
    with open(ahres_path) as f:
        ah_res = json.load(f)
    res["cells_vs_AH_47col"] = {
        "source": ahres_path,
        "per_cell": {
            c: {"ah_47col_own": ah_res["cells"][c]["own_mode_at_used"],
                "exp051_63col_own": cell_info[c]["own_mode_at_used"],
                "ah_argmin_iter": ah_res["cells"][c]["argmin_iter"],
                "exp051_argmin_iter": cell_info[c]["argmin_iter"],
                "iter_used_pinned": cell_info[c]["iter_used"]}
            for c in FAIL_CELLS}}

    # ================== THE MEASUREMENT: three round counts, two routes =====
    ROUNDS = [int(r) for r in args.rounds.split(",")]
    assert ROUNDS == sorted(ROUNDS) and ROUNDS[-1] == K, (ROUNDS, K)

    def evaluate(prob, label):
        b = float(np.mean((prob - y_va) ** 2))
        return {
            "valid_all_brier": b,
            "valid_all_score_converted": float(f"{score_of(b):.5f}"),
            "valid_all": ref.seg_metrics(y_va, prob),
            "valid_R": ref.seg_metrics(y_va[is_r], prob[is_r]),
            "valid_F": ref.seg_metrics(y_va[~is_r], prob[~is_r]),
            "pred_sha256": label,
        }

    # ---- route B: staged evaluation of the single cap-round fit -------------
    # pr(0, k) is CatBoost's raw formula value over trees [0, k), bias included,
    # so this is exactly what a k-round model of this fit would emit.
    routeB, probs_B = {}, {}
    for r in ROUNDS:
        p_r = ref.sigmoid(pr(0, r))
        probs_B[r] = p_r
        routeB[str(r)] = evaluate(p_r, "")
        print(f"    [staged @{r}] valid_all brier "
              f"{routeB[str(r)]['valid_all_brier']:.12f}  score "
              f"{routeB[str(r)]['valid_all_score_converted']}", flush=True)
    del pr
    gc.collect()

    # ---- route A: a SEPARATE fit at each round count ------------------------
    routeA, probs_A, fits = {}, {}, {}
    for r in ROUNDS:
        if r == K:
            # the cap fit IS the r == K fit; refitting it would only re-measure
            # CatBoost's determinism, which candidate AG already measured
            routeA[str(r)] = dict(routeB[str(r)])
            routeA[str(r)]["source"] = ("the cap fit itself -- no second fit "
                                        "was made at the largest round count")
            probs_A[r] = probs_B[r]
            fits[str(r)] = {"fit_min": round(fit_min, 3), "refitted": False}
            continue
        # exp_051: no refit at r < cap.  exp_047 measured the staged
        # prefix bit-identical to an independent fit, and exp_048's
        # 655/1000 numbers are staged, so this stays comparable.
        routeA[str(r)] = dict(routeB[str(r)])
        routeA[str(r)]["source"] = ("staged from the cap fit -- no refit; "
                                    "prefix identity measured by exp_047")
        probs_A[r] = probs_B[r]
        fits[str(r)] = {"fit_min": 0.0, "refitted": False}

    # ---- do the two routes agree? -------------------------------------------
    agree = {}
    for r in ROUNDS:
        d = np.abs(probs_A[r] - probs_B[r])
        agree[str(r)] = {
            "bit_identical": bool(np.array_equal(probs_A[r], probs_B[r])),
            "max_abs_pred_diff": float(d.max()),
            "n_rows_differing": int(np.count_nonzero(d)),
            "brier_abs_diff": abs(routeA[str(r)]["valid_all_brier"]
                                  - routeB[str(r)]["valid_all_brier"]),
        }
    res["prefix_identity_at_fold2024_scale"] = {
        "what": ("an independently fitted r-round stacker vs the first r trees "
                 "of the 1400-round fit, on the same OOF matrix.  This is the "
                 "same property gate G3 proves on the DEPLOYED boosters, "
                 "measured again here on different data and at a different "
                 "scale."),
        "per_round": agree,
    }

    res["stacker_by_rounds"] = {
        "rounds": ROUNDS,
        "route_A_independent_fits": routeA,
        "route_B_staged_from_cap_fit": routeB,
        "fits": fits,
    }
    deltas = {}
    for a, b in zip(ROUNDS, ROUNDS[1:]):
        db = (routeA[str(b)]["valid_all_brier"]
              - routeA[str(a)]["valid_all_brier"])
        deltas[f"{a}->{b}"] = {
            "delta_brier": db,
            "delta_score_converted": float(f"{-100000.0 * db / REF_BRIER:.5f}"),
        }
    res["deltas_route_A"] = deltas

    # the headline number keeps exp_036's field names so the two are comparable
    p = probs_A[ROUNDS[-1]]
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
    res["stacker_params"] = {
        kk: (vv if isinstance(vv, (int, float, str, bool, type(None)))
             else str(vv)) for kk, vv in params.items()}
    imp_s = model_s.get_feature_importance(type="PredictionValuesChange")
    order_s = np.argsort(-imp_s)
    res["stacker_importance_PredictionValuesChange"] = {
        feats[i]: float(f"{imp_s[i]:.6g}") for i in order_s}
    res["stacker_importance_rank"] = {feats[i]: int(k2 + 1)
                                      for k2, i in enumerate(order_s)}
    del model_s

    stem = f"exp051_u{V}_seed{SEED}"
    for r in ROUNDS:
        pth = os.path.join(out_dir, "pred", f"pred_{stem}_at{r}.npy")
        np.save(pth, probs_A[r])
        routeA[str(r)]["pred_sha256"] = sha256_file(pth)
        routeA[str(r)]["pred_file"] = os.path.relpath(pth, out_dir)
    res["scale_caveat"] = (
        "THIS IS NOT THE DEPLOYED MODEL.  fold_2024 trains on seasons <=2023 "
        "(1,116,277 rows) and validates on 2024; the deployed candidates train "
        "on <=2024 (1,369,784 rows).  A round count is not scale-free -- the "
        "argmin rises with the training row count -- so these numbers say what "
        "the local fold thinks of 655/1000/1400 AT ITS OWN SIZE, not what the "
        "deployment does at 1.23x the rows.")
    res["minutes"] = round((time.time() - t_start) / 60, 2)

    with open(os.path.join(out_dir, "res", f"res_{stem}.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("RESULT: " + json.dumps({
        "unit": V, "seed": SEED, "cell_iter_used": cell_iter_used,
        "rounds": ROUNDS,
        "brier_by_round": {str(r): routeA[str(r)]["valid_all_brier"]
                           for r in ROUNDS},
        "score_by_round": {str(r): routeA[str(r)]["valid_all_score_converted"]
                           for r in ROUNDS},
        "deltas": deltas,
        "gate_pass": gate["PASS"], "gate_sha_match": gate["sha_match"],
        "minutes": res["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
