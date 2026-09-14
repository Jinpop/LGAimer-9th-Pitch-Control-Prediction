"""exp_052 stage 0.6 / TASK B step 2 — arm E = student 67 + 6 PREDICTED physics
columns (73), against arm A (67) from stage 0.

CONTROL BY CITATION, NOT REFIT
------------------------------
arm A is stage 0's `res/g0b_armA.json` (Brier 0.24822938130212688, argmin 400).
It is not refitted; instead this file ASSERTS that arm A's recorded settings are
the ones used here -- identical CatBoost parameter dict except `iterations`,
identical fold string, identical cap and eval_period, identical column list for
the 67 shared columns and identical order.  If any of those differ the run
aborts, because then `E - A` would not be a one-variable comparison.

THE ONE VARIABLE: the six `pp_*` columns.  Nothing else moves.

🔴 PRE-REGISTERED READING (coordinator, before this ran)
    E - A >= 40 points  -> feature distillation opened the channel; the axis
                           lives; next step would be a 3-seed promotion.
    E - A <  40 points  -> the problem was neither compression nor
                           dimensionality; close the whole "trackman into
                           deployment through a model" axis.
An argmin on cap invalidates the arm (exp_017); cap is raised and it is re-run.

🔴 exp_003 / exp_012 COMPARISON IS REQUIRED IN THE WRITE-UP
Those rejected trackman PITCHER PROFILES (65 columns) and COMMAND metrics (60
columns) -- averages of past aggregates.  These six are per-row CONDITIONAL
EXPECTATIONS, a different object.  But if arm E dies the same way (more columns
eat the split candidates the trees can choose from), say so plainly.
"""
import hashlib
import importlib.util
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
PHYS6 = ["induced_vert_break", "horz_break", "rel_side", "rel_height",
         "spin_rate", "zone_speed"]
REF_BRIER = 0.2498069
N_THREADS = 6
SEED = 42
CAP = int(os.environ.get("ARME_CAP", "3000"))
EVAL_PERIOD = 25
GATE_POINTS = 40.0


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    import catboost as cb
    h = hashlib.sha256()
    with open(E027, "rb") as f:
        h.update(f.read())
    assert h.hexdigest() == E027_SHA
    spec = importlib.util.spec_from_file_location("e027", E027)
    m0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m0)
    CB = dict(m0.CB_BASE)

    armA = json.load(open(os.path.join(BASE, "res", "g0b_armA.json")))
    cols = json.load(open(os.path.join(BASE, "work", "cols.json")))
    student67 = cols["base47"] + cols["oof4"] + cols["ins16"]
    assert armA["n_cols"] == 67 and armA["blocks"] == ["base47", "oof4",
                                                       "ins16"]
    assert armA["seed"] == SEED and armA["eval_period"] == EVAL_PERIOD
    assert armA["cap"] == 3000

    df = pd.read_parquet(os.path.join(BASE, "work",
                                      "design_fold2023.parquet"))
    is_tr = df["_is_train"].to_numpy(bool)
    y = df[TARGET].to_numpy(np.float64)

    ppcols, pp_nan = [], {}
    for c in PHYS6:
        p = np.load(os.path.join(BASE, "work", f"phys_{c}.npz"))["pred"]
        assert p.shape == (len(df),), (c, p.shape)
        name = "pp_" + c
        df[name] = p
        ppcols.append(name)
        pp_nan[name] = {
            "nan_pct_train": round(float(np.isnan(p[is_tr]).mean() * 100), 4),
            "nan_pct_valid": round(float(np.isnan(p[~is_tr]).mean() * 100), 4)}
    assert all(v["nan_pct_valid"] == 0.0 for v in pp_nan.values()), pp_nan

    feats = student67 + ppcols
    assert len(feats) == 73 and len(set(feats)) == 73
    cat_native = [c for c in BASE_CAT if c in feats]

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

    ci = [feats.index(c) for c in cat_native]
    pool_tr = cb.Pool(F.loc[is_tr], label=y[is_tr], cat_features=ci)
    pool_va = cb.Pool(F.loc[~is_tr], cat_features=ci)
    y_va = y[~is_tr]
    del F
    log("arm E cols", len(feats), "train", int(is_tr.sum()),
        "valid", int((~is_tr).sum()), "cap", CAP)

    params = dict(CB, random_seed=SEED, iterations=CAP)
    # one-variable check: every CatBoost key except `iterations` must equal
    # what arm A recorded
    aparams = armA["params"]
    diff = {k: (str(v), aparams.get(k)) for k, v in params.items()
            if k != "iterations" and str(v) != aparams.get(k)}
    assert not diff, ("arm A settings differ", diff)

    mdl = cb.CatBoostClassifier(**params)
    mdl.fit(pool_tr)
    assert mdl.tree_count_ == CAP
    curve = {}
    for k, p in enumerate(mdl.staged_predict(
            pool_va, prediction_type="Probability", ntree_start=0,
            ntree_end=CAP, eval_period=EVAL_PERIOD, thread_count=N_THREADS)):
        curve[(k + 1) * EVAL_PERIOD] = float(
            np.mean((p[:, 1].astype(np.float64) - y_va) ** 2))
    bn = min(curve, key=curve.get)
    bE = curve[bn]
    bA = armA["brier_at_argmin"]
    gain = bA - bE
    g = dict(zip(mdl.feature_names_,
                 mdl.get_feature_importance(pool_tr).tolist()))
    tot = sum(g.values())
    rank = sorted(g, key=g.get, reverse=True)

    res = {
        "arm": "E", "n_cols": 73, "seed": SEED, "cap": CAP,
        "eval_period": EVAL_PERIOD, "fold": armA["fold"],
        "argmin_iter": int(bn),
        "argmin_at_cap_INVALID": bool(bn == CAP),
        "brier_E": bE, "brier_at_1400": curve.get(1400),
        "control_arm_A": {"brier": bA, "argmin_iter": armA["argmin_iter"],
                          "source": "res/g0b_armA.json (stage 0, cited not refit)",
                          "settings_identical_except_iterations": True},
        "E_minus_A": {"brier": gain,
                      "score_points": round(gain / REF_BRIER * 100000.0, 5),
                      "gate_points": GATE_POINTS,
                      "pass": bool(gain / REF_BRIER * 100000.0 >= GATE_POINTS)},
        "pp_nan": pp_nan,
        "pp_gain_pct_of_total": round(
            100 * sum(g[c] for c in ppcols) / tot, 4),
        "pp_gain_ranks": {c: rank.index(c) + 1 for c in ppcols},
        "pp_gain_values": {c: round(g[c], 4) for c in ppcols},
        "top15": [(c, round(g[c], 4)) for c in rank[:15]],
        "curve": {str(k): v for k, v in sorted(curve.items())},
        "params": {k: str(v) for k, v in params.items()},
        "total_min": round((time.time() - t0) / 60, 2),
        "versions": {"catboost": cb.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "python": sys.version.split()[0]},
    }
    with open(os.path.join(BASE, "res", "stage06_armE.json"), "w") as f:
        json.dump(res, f, indent=1)
    open(os.path.join(BASE, "logs", "armE.DONE"), "w").write("ok\n")
    log("E", bE, "A", bA, "E-A pts", res["E_minus_A"]["score_points"],
        "argmin", bn)


if __name__ == "__main__":
    main()
