"""exp_042 step 1 — parameter-acceptance probe + thread scaling calibration.

Two things, both required before the search is launched:

  (a) PROBE.  Does this CatBoost build actually accept all eight parameters the
      config registers?  `min_data_in_leaf` in particular is documented as
      Lossguide/Depthwise-only in some versions, and the default grow policy
      here is SymmetricTree.  A 30-round fit answers it.  If a parameter is
      rejected the search must be told, not silently reshaped.

  (b) CALIBRATION.  The same cell fit (current C5 settings, iterations = cap)
      timed at thread_count 6 / 3 / 2 / 20, run one at a time so nothing but
      the thread count differs.  Container CPU quota is 20 (cfs_quota 2000000 /
      cfs_period 100000); `nproc` 40 is the host's.

  (c) OD CROSS-CHECK.  One fit with the early-stopping detector the search will
      use, at the current settings, so its `best_iteration_` can be compared
      with the argmin exp_030 stage 1 found on the same cell with the coarse
      curve_grid.
"""

import json
import os
import sys
import time

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
import celldata                                                    # noqa: E402

OUT = os.path.expanduser("~/SungjinPark/experiments/exp_042/res/thread_cal.json")
CAP = 1650
OD_WAIT = 250

# the settings in production today (exp_027 CB_BASE via ARMS["C5"]).
# rsm / random_strength / min_data_in_leaf are not set by CB_BASE, so the
# CatBoost defaults apply; they are written out explicitly here so the search's
# trial 0 sits on the same ruler.
CURRENT = {
    "depth": 6, "l2_leaf_reg": 10.0, "learning_rate": 0.03, "subsample": 0.8,
    "rsm": 1.0, "random_strength": 1.0, "min_data_in_leaf": 1,
    "one_hot_max_size": 10,
}
FIXED = {
    "loss_function": "Logloss", "eval_metric": "Logloss", "task_type": "CPU",
    "boosting_type": "Plain", "bootstrap_type": "Bernoulli", "has_time": True,
    "allow_writing_files": False, "verbose": False,
}


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "is_wayoff_true"
    import catboost as cb
    t0 = time.time()
    D = celldata.build(cell)
    build_min = (time.time() - t0) / 60
    print(f"data built in {build_min:.2f} min: n_tr {D['n_train_mode']:,} "
          f"n_va_ok {D['n_valid_ok']:,} pos_rate_tr {D['pos_rate_train']:.6f}",
          flush=True)

    pool_tr = cb.Pool(D["F_tr"], label=D["y_tr"], cat_features=D["cat_idx"])
    pool_va = cb.Pool(D["F_va"], label=D["y_va"], cat_features=D["cat_idx"])

    out = {"cell": cell, "cap": CAP, "od_wait": OD_WAIT,
           "catboost": cb.__version__, "current_params": dict(CURRENT),
           "fixed_params": dict(FIXED), "data_build_min": round(build_min, 3),
           "n_train_mode": D["n_train_mode"], "n_valid_ok": D["n_valid_ok"],
           "train_csv_sha256": D["train_csv_sha256"]}

    # ---------------------------------------------------------------- probe
    probe = {}
    extreme = {"depth": 10, "l2_leaf_reg": 1.0, "learning_rate": 0.1,
               "subsample": 0.6, "rsm": 0.5, "random_strength": 10.0,
               "min_data_in_leaf": 200, "one_hot_max_size": 64}
    for name, ps in (("current", CURRENT), ("space_corner", extreme)):
        try:
            m = cb.CatBoostClassifier(**FIXED, **ps, random_seed=42,
                                      iterations=30, thread_count=6)
            m.fit(pool_tr)
            got = m.get_all_params()
            probe[name] = {"accepted": True, "tree_count": int(m.tree_count_),
                           "echo": {k: got.get(k) for k in list(ps) + ["grow_policy"]}}
        except Exception as e:                                     # noqa: BLE001
            probe[name] = {"accepted": False, "error": f"{type(e).__name__}: {e}"}
        print(f"probe {name}: {json.dumps(probe[name], default=str)}", flush=True)
    out["probe"] = probe

    # ---------------------------------------------------------- calibration
    cal = {}
    for t in (6, 3, 2, 20):
        m = cb.CatBoostClassifier(**FIXED, **CURRENT, random_seed=42,
                                  iterations=CAP, thread_count=t)
        s = time.time()
        m.fit(pool_tr)
        dt = time.time() - s
        assert int(m.tree_count_) == CAP, m.tree_count_
        cal[str(t)] = {"threads": t, "iterations": CAP, "fit_sec": round(dt, 2),
                       "fit_min": round(dt / 60, 3),
                       "sec_per_round": round(dt / CAP, 5)}
        print(f"threads {t:2d}: {dt/60:.3f} min  ({dt/CAP:.5f} s/round)", flush=True)
        del m
    base = cal["6"]["fit_sec"]
    for k in cal:
        cal[k]["speedup_vs_6"] = round(base / cal[k]["fit_sec"], 4)
        cal[k]["throughput_per_cpu"] = round(
            (base / cal[k]["fit_sec"]) / (cal[k]["threads"] / 6.0), 4)
    out["calibration"] = cal

    # ------------------------------------------------------- od cross-check
    m = cb.CatBoostClassifier(**FIXED, **CURRENT, random_seed=42,
                              iterations=CAP, thread_count=6,
                              od_type="Iter", od_wait=OD_WAIT,
                              use_best_model=False)
    s = time.time()
    m.fit(pool_tr, eval_set=pool_va)
    dt = time.time() - s
    bi = int(m.best_iteration_)
    raw = m.predict(pool_va, prediction_type="RawFormulaVal",
                    ntree_start=0, ntree_end=bi + 1, thread_count=6)
    p = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
    brier = float(np.mean((p - D["y_va"]) ** 2))
    ll = float(-np.mean(D["y_va"] * np.log(np.clip(p, 1e-15, 1))
                        + (1 - D["y_va"]) * np.log(np.clip(1 - p, 1e-15, 1))))
    out["od_cross_check"] = {
        "best_iteration": bi, "trees_built": int(m.tree_count_),
        "hit_cap": bool(int(m.tree_count_) >= CAP and bi >= CAP - 1),
        "best_score_eval": {k: float(v) for k, v in
                            m.get_best_score().get("validation", {}).items()},
        "logloss_at_best": ll, "brier_at_best": brier,
        "fit_sec": round(dt, 2), "fit_min": round(dt / 60, 3),
        "note": ("compare best_iteration with exp_030 stage1 (unit 2023 seed 42) "
                 "cell argmin for this cell, which was found on the coarse "
                 "curve_grid with a Brier objective"),
    }
    print("OD: " + json.dumps(out["od_cross_check"]), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE " + OUT, flush=True)


if __name__ == "__main__":
    main()
