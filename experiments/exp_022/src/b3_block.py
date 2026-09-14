"""exp_022 stage-1 helper — regenerate the exp_016-B3 prediction for ONE supply block.

WHY THIS EXISTS (it is not a new model)
---------------------------------------
exp_022 fits the offset slope `beta` on the supply blocks 2022 and 2023, the
same closed-form machine exp_019/exp_020 used.  That fit needs the residual
`y - p_B3` ON THOSE BLOCKS.  exp_020 computed those residuals in memory and
saved only `pred/pred_*_seed42_fold_2024.npy`; the block vectors were never
written to disk.  So they have to be produced again.

NOTHING NEW IS TRAINED.  This is exp_020's own code path, re-run:

  * `build_stash` and the B3 body are copied from exp_020/src/train.py, which is
    itself a line-for-line transcription of `exp016_ref.run_b3`'s loop body.
  * the only change is `num_boost_round = <the iteration we need>` instead of
    `N_ROUNDS = 3000`.  Gradient boosting is sequential: the first k trees of a
    3000-round booster and of a k-round booster are the same trees, and
    `curve_grid(3000)` and `curve_grid(k)` agree on every point <= k, so the
    raw-margin accumulation follows the identical chunk boundaries.  The claim
    is not asserted by argument — it is GATED below against the Brier exp_020
    recorded for this block.

REPRODUCTION GATE (per block, abort on failure)
-----------------------------------------------
  block 2022, iter 170 -> valid_all Brier 0.247681   (exp_020 metrics B3_block_2022)
  block 2023, iter 650 -> valid_all Brier 0.248756   (exp_020 metrics B3_exp016_path.fold_2023)

A miss means the truncation argument is wrong, or the data/env drifted, and the
beta fitted downstream would be unreadable.

Usage:  python b3_block.py <2022|2023>
"""

import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import lightgbm as lgb                                            # noqa: E402
import exp016_ref as ref                                          # noqa: E402
from labels import (FAIL_CELLS, assert_no_recovered_in_features,   # noqa: E402
                    recover)

SEED = 42
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
F_EXCLUDE_MAX_SEASON = 2022

# (train_upto, cell-model iter, B3 stacker iter, exp_020 valid_all Brier)
# every constant here is read off experiments/exp_020/metrics.json
BLOCK_SPEC = {
    2022: {"train_upto": 2021, "cell_iter": 240, "stack_iter": 170,
           "gate_brier": 0.247681, "n_valid": 247472, "n_train": 653728,
           "source": "exp_020 metrics.B3_block_2022 (iter_used 170 = own argmin)"},
    2023: {"train_upto": 2022, "cell_iter": 240, "stack_iter": 650,
           "gate_brier": 0.248756, "n_valid": 245525, "n_train": 870752,
           "source": "exp_020 metrics.B3_exp016_path.fold_2023 (iter_used 650)"},
}
TOL = 5e-6


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def prefix_grid(k):
    """curve_grid(3000) truncated at k.  Identical to curve_grid(k) on [1, k],
    which is what makes raw_margin_curve's chunking bit-identical to exp_020."""
    g = [x for x in ref.curve_grid(ref.N_ROUNDS) if x <= k]
    assert g[-1] == k, (k, g[-3:])
    ck = ref.curve_grid(k)
    assert g == [x for x in ck if x <= k], "grid prefix mismatch"
    return g


def main():
    V = int(sys.argv[1])
    spec = BLOCK_SPEC[V]
    tu, vy = spec["train_upto"], V
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(os.path.join(OUT_DIR, "pred"), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] block {V}: loading {DATA_PATH}", flush=True)
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

    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    cat = [c for c in CAT_COLS if c in base_feats]
    params = dict(ref.D_PARAMS, seed=SEED)

    tr_full = (season <= tu) & ~f_excl
    tr_mode = tr_full & ok & (succ_season <= tu)
    va = season == vy
    y_va = y_all[va]
    assert int(va.sum()) == spec["n_valid"], (int(va.sum()), spec["n_valid"])
    assert int(tr_full.sum()) == spec["n_train"], (int(tr_full.sum()), spec["n_train"])

    # ---- stash: cell models on the valid rows (exp_020 build_stash) --------- #
    cell_iter = spec["cell_iter"]
    g_cell = prefix_grid(cell_iter)
    assert_no_recovered_in_features(base_feats, f"block {V} cell models")
    X_mode = df.loc[tr_mode, base_feats]
    X_va = df.loc[va, base_feats]
    stash = {}
    for c in FAIL_CELLS:
        yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
        assert not np.isnan(yc).any()
        _, marg = ref.fit_curve(X_mode, yc, X_va, params, cell_iter, cat, g_cell,
                                f"{c} block_{V}")
        stash[c] = ref.sigmoid(marg[cell_iter])
        del marg
        gc.collect()
    print(f"  [stash block_{V}] cell iter {cell_iter}, n_mode {int(tr_mode.sum()):,}",
          flush=True)
    del X_mode, X_va
    gc.collect()

    # ---- B3 body (exp_020 b3_one_block = exp016_ref.run_b3 unrolled) -------- #
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    seasons = sorted(set(season[tr_full].tolist()))
    print(f"[{time.strftime('%H:%M:%S')}] B3 block_{V}: OOF blocks {seasons[1:]} "
          f"at iter {cell_iter}", flush=True)
    t0 = time.time()
    stack = {c: np.full(len(df), np.nan) for c in stack_cols}
    for s in seasons[1:]:
        fit_rows = ((season < s) & ~f_excl & ok & (succ_season < s))
        pred_rows = (season == s) & tr_full
        Xf = df.loc[fit_rows, base_feats]
        Xp = df.loc[pred_rows, base_feats]
        for c in FAIL_CELLS:
            assert_no_recovered_in_features(base_feats, f"B3 OOF {s} {c}")
            yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
            dtr = lgb.Dataset(Xf, label=yc, categorical_feature=cat)
            bst = lgb.train(params, dtr, num_boost_round=int(cell_iter))
            stack[f"p_{c}"][pred_rows] = bst.predict(Xp)
        print(f"    OOF season {s}: fit {int(fit_rows.sum()):,} -> "
              f"pred {int(pred_rows.sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    for c in FAIL_CELLS:
        stack[f"p_{c}"][va] = stash[c]

    Xs = df[base_feats].copy()
    for c in stack_cols:
        Xs[c] = stack[c]
    feats = base_feats + stack_cols
    assert_no_recovered_in_features(feats, f"B3 stacker block_{V}")
    k = spec["stack_iter"]
    g_stack = prefix_grid(k)
    _, marg = ref.fit_curve(Xs.loc[tr_full], y_all[tr_full], Xs.loc[va], params,
                            k, cat, g_stack, f"B3 stacker block_{V}")
    p = ref.sigmoid(marg[k])

    # ---- reproduction gate ------------------------------------------------- #
    brier = float(np.mean((p - y_va) ** 2))
    gate = {"block": V, "iter": k, "brier": round(brier, 9),
            "expected": spec["gate_brier"],
            "abs_diff": round(abs(round(brier, 6) - spec["gate_brier"]), 9),
            "abs_diff_unrounded": round(abs(brier - spec["gate_brier"]), 9),
            "tolerance": TOL, "source": spec["source"]}
    gate["passed"] = bool(gate["abs_diff_unrounded"] <= TOL)
    print("BLOCK GATE: " + json.dumps(gate), flush=True)

    out_p = os.path.join(OUT_DIR, "pred", f"pred_B3_block_{V}_seed42.npy")
    np.save(out_p, p)
    res = {
        "block": V, "gate": gate,
        "nan_stack_rows_in_train_pct": round(
            100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3),
        "n_train": int(tr_full.sum()), "n_valid": int(va.sum()),
        "pred_path": out_p, "pred_sha256": sha256_file(out_p),
        "minutes": round((time.time() - t_start) / 60, 2),
        "train_csv_sha256": sha256_file(DATA_PATH),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "provenance": "exp_020 b3_one_block / exp016_ref.run_b3 body, "
                      "num_boost_round truncated to the iteration used",
    }
    with open(os.path.join(OUT_DIR, f"block_{V}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] block {V} DONE {res['minutes']} min",
          flush=True)
    assert gate["passed"], "BLOCK REPRODUCTION GATE FAILED: " + json.dumps(gate)


if __name__ == "__main__":
    main()
