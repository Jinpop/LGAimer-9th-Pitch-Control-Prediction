"""exp_036 — the UNCENSORED fold_2024 argmin of the B3 stacker.

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

EXP_ID = "exp_036"
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_036")
EXP030_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")

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
    ap.add_argument("--cell-iter", required=True,
                    help="JSON dict {cell: iterations}; pinned from outside")
    ap.add_argument("--iter-src", required=True,
                    help="in words, where the CELL iterations came from")
    ap.add_argument("--gate-k", type=int, default=481,
                    help="round at which exp_030 must be reproduced")
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
    res["time_inversions"]["tr_full"] = assert_time_ordered(tkey, tr_full,
                                                            "tr_full")
    res["time_inversions"]["tr_mode"] = assert_time_ordered(tkey, tr_mode,
                                                            "tr_mode")

    # -------------------------------------------------- iteration resolution
    res["self_early_stop"] = False           # the CELLS never self-stop here
    ci = json.loads(args.cell_iter)
    assert set(ci) == set(FAIL_CELLS), (sorted(ci), sorted(FAIL_CELLS))
    cell_round = {c: int(ci[c]) for c in FAIL_CELLS}
    assert all(v > 0 for v in cell_round.values())
    res["iter_source"] = args.iter_src
    res["iter_policy"] = (
        "stage 5 — the four CELL iterations were supplied on the command line "
        "and this unit's own valid curve was not consulted to choose them.  "
        "The STACKER is trained to the cap and its dense (step-1) valid curve "
        "is the quantity being MEASURED; the argmin of that curve is a "
        "measurement, not a selected deployment setting.")
    res["cell_round"] = dict(cell_round)
    res["stack_round"] = int(args.cap)

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
        k_use = k                                   # cells are PINNED
        cell_argmin[c] = a_min
        stash[c] = ref.sigmoid(marg[k_use])
        cell_info[c] = {
            "n_round": k, "argmin_iter": a_min,
            "argmin_own_brier": float(f"{a_val:.10g}"),
            "argmin_at_cap": False,
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
    K = int(args.cap)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: stacker fit to {K} rounds",
          flush=True)
    t0 = time.time()
    pr, params, note = fit_catboost(C5, SEED, Fs_tr, y_all[tr_full], Fs_va,
                                    feats, cat_native, K)
    fit_min = (time.time() - t0) / 60
    print(f"    stacker fit {fit_min:.2f} min ({60*fit_min/K:.5f} s/round)",
          flush=True)

    # ---- GATE PASS: the INHERITED accumulation on the INHERITED grid ------- #
    # This is exp_030's arithmetic, unmodified, so the number it produces is
    # directly comparable to exp_030's recorded artefacts.
    GK = int(args.gate_k)
    assert GK <= K
    g_grid = ref.curve_grid(GK)
    assert g_grid[-1] == GK, (g_grid[-3:], GK)
    marg_g, const_g = chunked_margin_curve(pr, n_va, g_grid)
    cv_g = {g: float(np.mean((ref.sigmoid(marg_g[g]) - y_va) ** 2))
            for g in g_grid}
    p_gate = ref.sigmoid(marg_g[GK])
    brier_gate = float(np.mean((p_gate - y_va) ** 2))
    gate_path = os.path.join(out_dir, "pred",
                             f"pred_C5B3_u{V}_seed{SEED}_at{GK}_GATE.npy")
    np.save(gate_path, p_gate)
    del marg_g
    gc.collect()

    ref030 = os.path.join(EXP030_DIR, "res",
                          f"res_C5B3_u{V}_seed{SEED}_st2.json")
    with open(ref030) as f:
        r030 = json.load(f)
    assert r030["exp_id"] == "exp_030" and r030["stage"] == 2
    assert r030["unit"] == V and r030["seed"] == SEED
    assert r030["stack_round"] == GK, (r030["stack_round"], GK)
    assert r030["cell_iter_used"] == cell_iter_used, (
        "the pinned cell iterations are NOT exp_030's",
        r030["cell_iter_used"], cell_iter_used)
    exp030_brier = float(r030["brier_full"])
    abs_diff = abs(brier_gate - exp030_brier)
    gate = {
        "what": (f"the stacker trained to {K} rounds, its margins accumulated "
                 f"by the INHERITED chunked_margin_curve over the INHERITED "
                 f"curve_grid({GK}), evaluated at round {GK} — must reproduce "
                 f"exp_030's stage-2 fold_2024 prediction for this seed"),
        "expected_source": ref030,
        "expected_brier": exp030_brier,
        "observed_brier": brier_gate,
        "abs_diff": abs_diff,
        "tol": GATE_TOL,
        "PASS": bool(abs_diff < GATE_TOL),
        "expected_pred_sha256": r030["pred_sha256"],
        "observed_pred_sha256": sha256_file(gate_path),
        "bit_identical": bool(sha256_file(gate_path) == r030["pred_sha256"]),
        "exp030_cell_iter_used": r030["cell_iter_used"],
    }
    res["reproduction_gate"] = gate
    print(f"    GATE @{GK}: ours {brier_gate:.17g} vs exp_030 "
          f"{exp030_brier:.17g}, abs_diff {abs_diff:.3e}, "
          f"{'PASS' if gate['PASS'] else 'FAIL'}, "
          f"bit_identical={gate['bit_identical']}", flush=True)
    assert gate["PASS"], ("reproduction gate FAILED", gate)

    # ---- DENSE PASS: every round 1..cap ------------------------------------ #
    curve, const, a_min, p_best = dense_brier_curve(pr, y_va, K, tag=tag)
    a_val = curve[a_min]
    k_use = a_min                                  # measured, not selected
    p = p_best

    at_cap = bool(a_min >= K)
    res["stacker"] = {
        "n_round": K, "argmin_iter": a_min,
        "argmin_brier": a_val,
        "argmin_at_cap": at_cap,
        "curve_argmin_is_interior": bool(a_min < K),
        "iter_used": int(k_use),
        "brier_at_gate_k": brier_gate,
        "gate_k": GK,
        "gain_from_gate_k_to_argmin_brier": float(cv_g[GK] - a_val),
        "gain_from_gate_k_to_argmin_score": float(
            f"{-100000.0 * (a_val - cv_g[GK]) / REF_BRIER:.5f}"),
        "brier_at_cap": curve[K],
        "rise_argmin_to_cap": float(curve[K] - a_val),
        "fit_min": round(fit_min, 3),
        "sec_per_round": round(60.0 * fit_min / K, 5),
        "margin_const": float(const),
        "margin_const_gate_path": float(const_g),
        "params": {kk: (vv if isinstance(vv, (int, float, str, bool, type(None)))
                        else str(vv)) for kk, vv in params.items()},
        "family_note": note,
    }
    res["stack_argmin_iter"] = a_min
    res["any_argmin_at_cap"] = at_cap

    brier = float(np.mean((p - y_va) ** 2))
    assert abs(brier - a_val) < 1e-15, (brier, a_val)
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
    res["gate_pred_file"] = os.path.basename(gate_path)

    with open(os.path.join(out_dir, "curves", f"curve_{stem}.json"), "w") as f:
        json.dump({"exp_id": EXP_ID, "unit": V, "seed": SEED, "stage": STAGE,
                   "stacker_cap": K,
                   "stacker_curve_step": 1,
                   "stacker_valid_all_brier_dense": {
                       str(k): float(f"{v:.12g}") for k, v in curve.items()},
                   "stacker_argmin_iter": a_min,
                   "stacker_argmin_brier": a_val,
                   "gate_k": GK,
                   "stacker_valid_all_brier_inherited_grid": {
                       str(g): float(f"{v:.12g}") for g, v in cv_g.items()},
                   "cell_own_target_brier": cell_curves,
                   "cell_argmin_iter": cell_argmin,
                   "cell_iter_used": cell_iter_used}, f)
    res["curve_file"] = f"curves/curve_{stem}.json"
    res["minutes"] = round((time.time() - t_start) / 60, 2)

    with open(os.path.join(out_dir, "res", f"res_{stem}.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("RESULT: " + json.dumps({
        "unit": V, "seed": SEED, "stage": STAGE,
        "cell_iter_used": cell_iter_used,
        "stack_cap": K, "stack_argmin": a_min,
        "argmin_at_cap": at_cap,
        "brier_at_argmin": brier, "score_at_argmin": res["score_converted"],
        "brier_at_gate_k": brier_gate,
        "gate_pass": gate["PASS"], "gate_bit_identical": gate["bit_identical"],
        "minutes": res["minutes"]}), flush=True)
    if at_cap:
        print(f"!! UNIT INVALID: the stacker argmin reached the cap {K}. "
              f"Re-run at a larger cap.", flush=True)


if __name__ == "__main__":
    main()
