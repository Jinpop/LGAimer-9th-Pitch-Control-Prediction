"""exp_024 stage-1 — regenerate the exp_016-B3 prediction for ONE unit at ONE seed.

PROVENANCE, AND WHAT CHANGED FROM exp_022/src/b3_block.py
---------------------------------------------------------
This file is exp_022's `b3_block.py` — sha256
c435175ef4fdc778f8b6bc05783377f80c15a1af28a1887ece827231e304830c as copied —
with FOUR edits, all of them listed here because an unlisted edit is the failure
mode the project bans:

  1. `SEED` is a command-line argument instead of the constant 42, and the
     output file names carry it.  Nothing else about the run depends on the
     seed except `params["seed"]`, exactly as exp_020/exp_022 had it.
  2. `BLOCK_SPEC` gains a 2024 entry — train<=2023, cell iter 1150, stacker
     iter 650 — which is the fold_2024 base of exp_020 arm E0.  Its constants
     are read off experiments/exp_020/metrics.json B3_exp016_path.fold_2024
     the same way the 2022/2023 constants were.
  3. The reproduction gate FIRES ONLY AT SEED 42.  exp_020 measured these Briers
     at seed 42; at seed 7 or 2024 there is no expected value to compare against,
     so the gate is recorded as `applicable: false` and the Brier is reported
     without a pass/fail claim.  This is the honest form — inventing a tolerance
     for an unmeasured quantity would be worse than having no gate.
  4. OUT_DIR points at exp_024.

The body between the gate and the data loading is untouched: it is exp_020's
`b3_one_block`, itself a line-for-line transcription of `exp016_ref.run_b3`'s
per-fold loop, with `num_boost_round` truncated to the iteration actually used.
The truncation argument (the first k trees of a 3000-round booster are the first
k trees of a k-round booster, and `curve_grid(3000)` agrees with `curve_grid(k)`
on every point <= k) was GATED at seed 42 in exp_022 for blocks 2022 and 2023
and is gated here again for all three units at seed 42.

WHY THIS RUNS AT ALL
--------------------
exp_023 was a post-hoc transform of three SAVED seed-42 prediction vectors.  The
adoption threshold is three seeds.  Seeds 7 and 2024 have no such vectors, so
the base predictions must be produced.  They are saved to exp_024/pred/ so that
the next experiment does not pay this cost again — exp_022 failed to save its
block vectors and exp_023 had to regenerate them.

Usage:  python b3_block.py <2022|2023|2024> <seed>
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

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_024")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
F_EXCLUDE_MAX_SEASON = 2022

GATE_SEED = 42          # the only seed for which an expected Brier exists

# (train_upto, cell-model iter, B3 stacker iter, exp_020 valid_all Brier at seed 42)
BLOCK_SPEC = {
    2022: {"train_upto": 2021, "cell_iter": 240, "stack_iter": 170,
           "gate_brier": 0.247681, "n_valid": 247472, "n_train": 653728,
           "source": "exp_020 metrics.B3_block_2022 (iter_used 170 = own argmin)"},
    2023: {"train_upto": 2022, "cell_iter": 240, "stack_iter": 650,
           "gate_brier": 0.248756, "n_valid": 245525, "n_train": 870752,
           "source": "exp_020 metrics.B3_exp016_path.fold_2023 (iter_used 650)"},
    2024: {"train_upto": 2023, "cell_iter": 1150, "stack_iter": 650,
           "gate_brier": 0.248191, "n_valid": 253507, "n_train": 1116277,
           "source": "exp_020 metrics.B3_exp016_path.fold_2024 "
                     "(iter_used 650 = P4 carry from fold_2023; cell iter 1150 "
                     "= exp_016 B2c fold_2024)"},
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
    SEED = int(sys.argv[2])
    spec = BLOCK_SPEC[V]
    tu, vy = spec["train_upto"], V
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(os.path.join(OUT_DIR, "pred"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "blocks"), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] unit {V} seed {SEED}: loading {DATA_PATH}",
          flush=True)
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
    assert params["num_threads"] == 6, params        # efficiency rule: fixed at 6

    tr_full = (season <= tu) & ~f_excl
    tr_mode = tr_full & ok & (succ_season <= tu)
    va = season == vy
    y_va = y_all[va]
    assert int(va.sum()) == spec["n_valid"], (int(va.sum()), spec["n_valid"])
    assert int(tr_full.sum()) == spec["n_train"], (int(tr_full.sum()), spec["n_train"])

    # ---- stash: cell models on the valid rows (exp_020 build_stash) --------- #
    cell_iter = spec["cell_iter"]
    g_cell = prefix_grid(cell_iter)
    assert_no_recovered_in_features(base_feats, f"unit {V} cell models")
    X_mode = df.loc[tr_mode, base_feats]
    X_va = df.loc[va, base_feats]
    stash = {}
    for c in FAIL_CELLS:
        yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
        assert not np.isnan(yc).any()
        _, marg = ref.fit_curve(X_mode, yc, X_va, params, cell_iter, cat, g_cell,
                                f"{c} unit_{V}_s{SEED}")
        stash[c] = ref.sigmoid(marg[cell_iter])
        del marg
        gc.collect()
    print(f"  [stash {V}/{SEED}] cell iter {cell_iter}, "
          f"n_mode {int(tr_mode.sum()):,}", flush=True)
    del X_mode, X_va
    gc.collect()

    # ---- B3 body (exp_020 b3_one_block = exp016_ref.run_b3 unrolled) -------- #
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    seasons = sorted(set(season[tr_full].tolist()))
    print(f"[{time.strftime('%H:%M:%S')}] B3 {V}/{SEED}: OOF blocks {seasons[1:]} "
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
    assert_no_recovered_in_features(feats, f"B3 stacker {V}/{SEED}")
    k = spec["stack_iter"]
    g_stack = prefix_grid(k)
    _, marg = ref.fit_curve(Xs.loc[tr_full], y_all[tr_full], Xs.loc[va], params,
                            k, cat, g_stack, f"B3 stacker {V}/{SEED}")
    p = ref.sigmoid(marg[k])

    # ---- reproduction gate (seed 42 only) ---------------------------------- #
    brier = float(np.mean((p - y_va) ** 2))
    applicable = (SEED == GATE_SEED)
    gate = {"unit": V, "seed": SEED, "iter": k, "brier": round(brier, 9),
            "brier_full": brier,
            "expected_at_seed42": spec["gate_brier"],
            "applicable": applicable,
            "tolerance": TOL, "source": spec["source"]}
    if applicable:
        gate["abs_diff"] = round(abs(round(brier, 6) - spec["gate_brier"]), 9)
        gate["abs_diff_unrounded"] = round(abs(brier - spec["gate_brier"]), 9)
        gate["passed"] = bool(gate["abs_diff_unrounded"] <= TOL)
    else:
        gate["passed"] = None
        gate["note"] = ("no expected Brier exists at this seed; the number above "
                        "is a measurement, not a reproduction claim")
    print("UNIT GATE: " + json.dumps(gate), flush=True)

    out_p = os.path.join(OUT_DIR, "pred", f"pred_B3_block_{V}_seed{SEED}.npy")
    np.save(out_p, p)
    res = {
        "block": V, "seed": SEED, "gate": gate,
        "nan_stack_rows_in_train_pct": round(
            100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3),
        "n_train": int(tr_full.sum()), "n_valid": int(va.sum()),
        "cell_iter": int(cell_iter), "stack_iter": int(k),
        "pred_path": out_p, "pred_sha256": sha256_file(out_p),
        "minutes": round((time.time() - t_start) / 60, 2),
        "train_csv_sha256": sha256_file(DATA_PATH),
        "lgb_params": {kk: vv for kk, vv in params.items()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "provenance": "exp_020 b3_one_block / exp016_ref.run_b3 body, "
                      "num_boost_round truncated to the iteration used",
    }
    with open(os.path.join(OUT_DIR, "blocks", f"block_{V}_seed{SEED}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] unit {V} seed {SEED} DONE "
          f"{res['minutes']} min", flush=True)
    if applicable:
        assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)


if __name__ == "__main__":
    main()
