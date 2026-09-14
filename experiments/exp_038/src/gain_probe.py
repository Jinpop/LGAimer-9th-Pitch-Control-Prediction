"""exp_038 — feature-importance probe for the two `idea` columns.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The B3 stacker's importances are not recoverable after the fact: `b3_feat.py`
is inherited BYTE-IDENTICAL from exp_037 (that byte-identity is what lets this
experiment claim its fits travel exp_037's code path), it returns a staged
predictor rather than a model, and it saves no booster.  Changing it to dump
importances would trade a proven code path for a diagnostic, which is a bad
trade.

So this file fits a SEPARATE, FLAT model and says so loudly:

    rows      = the promoted unit's stacker training pool, exactly
                (season <= 2023, minus the exp_004 F-mask), 1,116,277 rows
    target    = control_success, the stacker's target
    features  = the promoted arm's 49 columns
                MINUS the four p_* out-of-fold stack columns
    seed      = 42, C5 CatBoost parameters, rounds = the promoted unit's
                stack_round (read from the stage-1 json, not typed in)

It therefore answers "where do the two columns rank among the 49, in a model
trained on the same rows and target" and NOT "what is their gain inside the
stacker".  Those are different questions; only the first one is cheap.  No
number produced here is used in any Brier, delta, or adoption decision — it is
supporting evidence for the decomposition and nothing else.
"""
import argparse
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

from arms import ARMS, EXPECTED_N                                    # noqa: E402
from labels import assert_no_recovered_in_features                   # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
EXP027_TRAIN = os.path.expanduser(
    "~/SungjinPark/experiments/exp_027/src/train.py")
EXP027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
F_EXCLUDE_MAX_SEASON = 2022
BASE_UNIT, PROBE_UNIT = 2023, 2024
PROBE_TRAIN_UPTO = 2023


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-json", required=True)
    ap.add_argument("--feat-parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ARM, SEED = args.arm, args.seed
    out = os.path.abspath(args.out)
    t0 = time.time()
    np.random.seed(SEED)

    assert sha256_file(EXP027_TRAIN) == EXP027_SHA
    spec = importlib.util.spec_from_file_location("exp027_train", EXP027_TRAIN)
    e27 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e27)
    C5 = e27.ARMS["C5"]

    with open(args.base_json) as f:
        base = json.load(f)
    assert base["arm"] == ARM and base["stage"] == 1
    assert base["unit"] == BASE_UNIT
    assert not base["any_argmin_at_cap"], "stage 1 was invalid (cap hit)"

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47
    extra = list(ARMS[ARM]["cols"])
    F = pd.read_parquet(os.path.abspath(args.feat_parquet))
    assert (F["row_id"].to_numpy() == df["row_id"].to_numpy()).all()
    for c in extra:
        df[c] = F[c].to_numpy(dtype=np.float64)
    del F
    feats = base_feats + extra
    assert len(feats) == EXPECTED_N[ARM]
    assert_no_recovered_in_features(feats, "gain probe")

    season = df["season"].to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    tr = (season <= PROBE_TRAIN_UPTO) & ~f_excl
    n_tr = int(tr.sum())
    # the same row-count ratio b3_feat.py uses to carry an argmin across units
    n_round = int(round(base["stack_argmin_iter"] * (n_tr / base["n_train_full"])))
    cat_native = [c for c in feats if c in CAT_COLS]
    assert len(cat_native) == 3

    Ftr = e27.build_frame_native(df.loc[tr], feats, cat_native)
    y = df.loc[tr, TARGET].to_numpy(dtype=np.float64)
    # F_va is unused by this probe but fit_catboost wants a valid frame; the
    # first 1000 training rows are handed over precisely so that nothing this
    # file touches can depend on a validation season.
    Fva = e27.build_frame_native(df.loc[tr].iloc[:1000], feats, cat_native)
    pr, params, _ = e27.fit_catboost(C5, SEED, Ftr, y, Fva, feats,
                                     cat_native, n_round)
    fv = pr.__code__.co_freevars
    model = pr.__closure__[fv.index("model")].cell_contents

    imp = model.get_feature_importance(type="PredictionValuesChange")
    assert len(imp) == len(feats)
    order = np.argsort(-np.asarray(imp))
    ranks = {feats[i]: int(np.where(order == i)[0][0]) + 1
             for i in range(len(feats))}

    G = {
        "what": ("flat C5 CatBoost on the promoted unit's stacker training "
                 "pool and target, WITHOUT the four p_* stack columns. Ranks "
                 "the two idea columns among the arm's 49; it is NOT the "
                 "stacker's own importance and no decision uses it."),
        "arm": ARM, "seed": SEED, "n_train": n_tr, "n_feats": len(feats),
        "unit_probed": PROBE_UNIT, "train_upto": PROBE_TRAIN_UPTO,
        "n_round": n_round,
        "n_round_source": (f"stage-1 stacker argmin {base['stack_argmin_iter']}"
                           f" x {n_tr}/{base['n_train_full']}"),
        "importance_type": "PredictionValuesChange",
        "importance_sum": float(np.sum(imp)),
        "extra_cols": extra,
        "extra_col_importance": {c: float(imp[feats.index(c)]) for c in extra},
        "extra_col_rank_of_49": {c: ranks[c] for c in extra},
        "extra_col_share_pct": {c: float(100.0 * imp[feats.index(c)]
                                         / np.sum(imp)) for c in extra},
        "top15": [{"feat": feats[i], "importance": float(imp[i]),
                   "rank": int(k) + 1} for k, i in enumerate(order[:15])],
        "params": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                       else str(v)) for k, v in params.items()},
        "minutes": round((time.time() - t0) / 60, 2),
    }
    os.makedirs(os.path.join(out, "res"), exist_ok=True)
    with open(os.path.join(out, "res", f"gain_probe_{ARM}.json"), "w") as f:
        json.dump(G, f, indent=1)
    print(json.dumps({k: G[k] for k in ("extra_col_rank_of_49",
                                        "extra_col_share_pct", "n_round",
                                        "minutes")}), flush=True)


if __name__ == "__main__":
    main()
