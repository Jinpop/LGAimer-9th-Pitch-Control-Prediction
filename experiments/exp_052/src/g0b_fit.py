"""exp_052 stage 0 / G0b — one teacher arm, fold_2023, seed 42.

  A  67 = base47 + oof4 + ins16                      student-available (deployed)
  B  79 = A + tm12                                   A + trackman
  C  81 = A + f14 (p6_*, idea, i3_anom_*)            student-available (81-col)
  D  93 = C + tm12                                   C + trackman

REPORTED UPPER BOUNDS
  상한_81 = brier(C) - brier(D)   <- the pre-registered 40-point gate rides here
  상한_67 = brier(A) - brier(B)   <- reference only
  C - A                          <- what the 14 columns already carry

PRE-REGISTERED READING RULE (fixed before any fit ran)
------------------------------------------------------
Each arm is trained to CAP rounds and its validation curve is evaluated on ALL
2023 rows (valid_all) at every EVAL_PERIOD rounds in ONE staged pass.  An arm's
Brier is the MINIMUM of its own curve.  Taking each arm's own argmin is the
reading most generous to the trackman arms, which is the correct direction for
a gate whose job is to CLOSE the axis cheaply: if the most optimistic reading
does not clear 40 points, no less optimistic one will.
An argmin landing on CAP makes that arm INVALID (exp_017 rule); reported.

MODEL: exp_027 `CB_BASE` + `ARMS["C5"]` (params {}), random_seed 42,
       thread_count 6, has_time True.  Identical across arms; the ONLY
       difference between arms is the column list.

LEAK SURFACE
  * no column here is derived from the current row's label; the four OOF
    columns are out-of-fold forward-chained cell predictions and appear
    IDENTICALLY in all four arms.
  * trackman columns are the CURRENT pitch's measurements.  That is the point:
    a teacher is a measuring instrument, never shipped.  The config's rule
    amendment covers exactly this.
  * fold_2024 is never read.  test.csv is never read.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
E027 = os.path.expanduser("~/SungjinPark/experiments/exp_027/src/train.py")
E027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
TARGET = "control_success"
BASE_CAT = ["top_bottom", "game_type", "base_state"]
TM_LAB = ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]
REF_BRIER = 0.2498069
N_THREADS = 6
CAP = 3000
EVAL_PERIOD = 25
SEED = 42

ARM_BLOCKS = {"A": ["base47", "oof4", "ins16"],
              "B": ["base47", "oof4", "ins16", "tm12"],
              "C": ["base47", "oof4", "ins16", "f14"],
              "D": ["base47", "oof4", "ins16", "f14", "tm12"]}


def sha256_file(path, block=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_cb_base():
    import importlib.util
    got = sha256_file(E027)
    assert got == E027_SHA, ("exp_027 train.py changed", got)
    spec = importlib.util.spec_from_file_location("e027", E027)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.ARMS["C5"]["params"] == {} and not m.ARMS["C5"]["cat_ids"]
    assert m.CB_BASE["has_time"] is True
    assert m.CB_BASE["thread_count"] == N_THREADS
    return dict(m.CB_BASE)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_BLOCKS))
    args = ap.parse_args()
    arm = args.arm
    t0 = time.time()
    import catboost as cb

    cols = json.load(open(os.path.join(BASE, "work", "cols.json")))
    feats = [c for b in ARM_BLOCKS[arm] for c in cols[b]]
    assert len(feats) == len(set(feats))
    cat_native = [c for c in BASE_CAT + TM_LAB if c in feats]

    df = pd.read_parquet(os.path.join(BASE, "work", "design_fold2023.parquet"))
    is_tr = df["_is_train"].to_numpy(bool)
    y = df[TARGET].to_numpy(np.float64)
    log(arm, "cols", len(feats), "train", int(is_tr.sum()),
        "valid", int((~is_tr).sum()))

    # exp_027 build_frame_native: cat columns keep native values, rest float64
    data = {}
    for c in feats:
        if c in cat_native:
            s = df[c]
            assert s.notna().all(), c
            data[c] = s.astype(str).to_numpy()
        else:
            data[c] = pd.to_numeric(df[c], errors="raise").to_numpy(np.float64)
    F = pd.DataFrame(data, columns=feats)
    del df, data

    cat_idx = [feats.index(c) for c in cat_native]
    pool_tr = cb.Pool(F.loc[is_tr], label=y[is_tr], cat_features=cat_idx)
    pool_va = cb.Pool(F.loc[~is_tr], cat_features=cat_idx)
    y_va = y[~is_tr]
    del F

    params = dict(load_cb_base(), random_seed=SEED, iterations=CAP)
    model = cb.CatBoostClassifier(**params)
    log("fit", params)
    model.fit(pool_tr)
    assert model.tree_count_ == CAP, model.tree_count_
    t_fit = time.time() - t0

    log("staged curve")
    curve = {}
    for k, p in enumerate(model.staged_predict(
            pool_va, prediction_type="Probability", ntree_start=0,
            ntree_end=CAP, eval_period=EVAL_PERIOD, thread_count=N_THREADS)):
        n = (k + 1) * EVAL_PERIOD
        curve[n] = float(np.mean((p[:, 1].astype(np.float64) - y_va) ** 2))
    best_n = min(curve, key=curve.get)
    best = curve[best_n]

    gain = dict(zip(model.feature_names_,
                    model.get_feature_importance(pool_tr).tolist()))
    res = {
        "arm": arm, "n_cols": len(feats), "blocks": ARM_BLOCKS[arm],
        "seed": SEED, "cap": CAP, "eval_period": EVAL_PERIOD,
        "fold": "train<=2022(F-masked) -> valid 2023 (valid_all)",
        "argmin_iter": int(best_n),
        "argmin_at_cap_INVALID": bool(best_n == CAP),
        "brier_at_argmin": best,
        "brier_at_1400": curve.get(1400),
        "brier_at_cap": curve[CAP],
        "curve": {str(k): v for k, v in sorted(curve.items())},
        "gain": gain,
        "params": {k: str(v) for k, v in params.items()},
        "fit_min": round(t_fit / 60, 2),
        "total_min": round((time.time() - t0) / 60, 2),
        "versions": {"catboost": cb.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "python": sys.version.split()[0]},
    }
    os.makedirs(os.path.join(BASE, "res"), exist_ok=True)
    with open(os.path.join(BASE, "res", f"g0b_arm{arm}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(arm, "argmin", best_n, "brier", f"{best:.8f}", "min",
        res["total_min"])


if __name__ == "__main__":
    main()
