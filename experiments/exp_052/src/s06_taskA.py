"""exp_052 stage 0.6 / TASK A — is the matched-row +8.00848 points bigger than
seed noise?  P-B repeated at seeds 42 / 7 / 2024, arms P0 and P2 only.

P1 (pure soft) is NOT repeated: stage 0.5 measured it at -139.6 points overall
and -186.7 on matched rows.  Spending seeds on a dead arm buys nothing.

EVERYTHING except the seed is held fixed to stage 0.5's P-B:
  same season-2023 time-order 60/40 split, same student 67 columns, same
  `CrossEntropy` loss in both arms, same cap 3000 / eval_period 25, each arm
  reading its own argmin on the back 40%'s real labels.

REPRODUCTION GATE: seed 42 is REFITTED here, not copied.  Its two Briers must
match stage 0.5's stored values to < 5e-6, which proves this file's code path
is the same object stage 0.5 measured.

🔴 NO PRE-REGISTERED THRESHOLD.  sigma has never been measured on this plane at
this setting, so a threshold would be invented after the fact.  This file
reports sign agreement across the three seeds and the size in units of
std(ddof=1), and nothing stronger.
"""
import hashlib
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
E027 = os.path.expanduser("~/SungjinPark/experiments/exp_027/src/train.py")
E027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
TARGET = "control_success"
BASE_CAT = ["top_bottom", "game_type", "base_state"]
REF_BRIER = 0.2498069
N_THREADS = 6
SEEDS = (42, 7, 2024)
CAP = 3000
EVAL_PERIOD = 25
FRONT_FRAC = 0.60
GATE_TOL = 5e-6
GATE42 = {"P0_hard": 0.2481909546983687, "P2_mix50": 0.24812048487377822}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def pts(x):
    return round(x / REF_BRIER * 100000.0, 5)


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

    cols = json.load(open(os.path.join(BASE, "work", "cols.json")))
    student67 = cols["base47"] + cols["oof4"] + cols["ins16"]
    df = pd.read_parquet(os.path.join(BASE, "work", "design_fold2023.parquet"))
    d23 = df.loc[~df["_is_train"].to_numpy(bool)].reset_index(drop=True)
    del df
    n23 = len(d23)
    assert n23 == 245525
    y23 = d23[TARGET].to_numpy(np.float64)
    has_tm = d23["has_tm"].to_numpy(np.float64) > 0.5
    n_front = int(round(n23 * FRONT_FRAC))
    front = np.zeros(n23, bool)
    front[:n_front] = True
    back = ~front

    t_prob = np.load(os.path.join(BASE, "work", "teacher_D_2023_at700.npy"))
    assert hashlib.sha256(t_prob.tobytes()).hexdigest() == (
        "a9140747046b6f2e86f2166f45c01963d058fd2fdcbcfc7b3c28edf64cd344a5")

    catS = [c for c in BASE_CAT if c in student67]
    data = {}
    for c in student67:
        data[c] = (d23[c].astype(str).to_numpy() if c in catS
                   else pd.to_numeric(d23[c], errors="raise")
                   .to_numpy(np.float64))
    FS = pd.DataFrame(data, columns=student67)
    cis = [student67.index(c) for c in catS]
    pool_b = cb.Pool(FS.loc[back], cat_features=cis)

    targets = {"P0_hard": y23,
               "P2_mix50": np.where(has_tm, 0.5 * t_prob + 0.5 * y23, y23)}
    msk = has_tm[back]
    w = float(msk.mean())
    res = {"exp_id": "exp_052", "stage": 0.6, "task": "A",
           "seeds": list(SEEDS), "cap": CAP, "has_tm_share_back40": w,
           "no_threshold": ("sigma unmeasured on this plane/setting -- sign "
                            "agreement and std multiples only"),
           "per_seed": {}}

    for seed in SEEDS:
        res["per_seed"][str(seed)] = {}
        for name, tgt in targets.items():
            pool_ft = cb.Pool(FS.loc[front], label=tgt[front],
                              cat_features=cis)
            mdl = cb.CatBoostClassifier(**dict(
                CB, loss_function="CrossEntropy", random_seed=seed,
                iterations=CAP))
            mdl.fit(pool_ft)
            curve, curve_tm, curve_no = {}, {}, {}
            for k, p in enumerate(mdl.staged_predict(
                    pool_b, prediction_type="Probability", ntree_start=0,
                    ntree_end=CAP, eval_period=EVAL_PERIOD,
                    thread_count=N_THREADS)):
                v = p[:, 1].astype(np.float64)
                n = (k + 1) * EVAL_PERIOD
                curve[n] = float(np.mean((v - y23[back]) ** 2))
                curve_tm[n] = float(np.mean((v[msk] - y23[back][msk]) ** 2))
                curve_no[n] = float(
                    np.mean((v[~msk] - y23[back][~msk]) ** 2))
            bn = min(curve, key=curve.get)
            res["per_seed"][str(seed)][name] = {
                "argmin_iter": int(bn),
                "argmin_at_cap_INVALID": bool(bn == CAP),
                "brier_all": curve[bn],
                "brier_has_tm": curve_tm[bn],
                "brier_no_tm": curve_no[bn]}
            log(seed, name, "argmin", bn, f"{curve[bn]:.8f}")
            del mdl, pool_ft

    # seed 42 reproduction gate
    g = {}
    for name, want in GATE42.items():
        got = res["per_seed"]["42"][name]["brier_all"]
        g[name] = {"stage05": want, "here": got, "abs_diff": abs(got - want),
                   "pass": bool(abs(got - want) < GATE_TOL)}
    res["gate_seed42_vs_stage05"] = g
    assert all(v["pass"] for v in g.values()), g
    log("seed-42 reproduction gate ok")

    # deltas P2 - P0, per seed, on three row sets
    dd = {"all": [], "has_tm": [], "no_tm": []}
    for seed in SEEDS:
        r = res["per_seed"][str(seed)]
        for key, col in (("all", "brier_all"), ("has_tm", "brier_has_tm"),
                         ("no_tm", "brier_no_tm")):
            dd[key].append(r["P0_hard"][col] - r["P2_mix50"][col])
    summ = {}
    for key, v in dd.items():
        a = np.asarray(v, np.float64)
        mean = float(a.mean())
        sd = float(a.std(ddof=1))
        summ[key] = {
            "per_seed_brier": {str(s): float(x) for s, x in zip(SEEDS, a)},
            "per_seed_points": {str(s): pts(float(x)) for s, x in zip(SEEDS, a)},
            "mean_brier": mean, "mean_points": pts(mean),
            "std_ddof1_brier": sd, "std_ddof1_points": pts(sd),
            "abs_mean_over_std": (abs(mean) / sd) if sd > 0 else None,
            "signs": [int(np.sign(x)) for x in a],
            "sign_agreement": bool(len(set(np.sign(a).tolist())) == 1),
        }
    res["delta_P2_minus_P0"] = summ
    res["any_argmin_at_cap"] = any(
        res["per_seed"][str(s)][n]["argmin_at_cap_INVALID"]
        for s in SEEDS for n in targets)
    res["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(BASE, "res", "stage06_taskA.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    open(os.path.join(BASE, "logs", "taskA.DONE"), "w").write("ok\n")
    log("done", res["elapsed_min"], "min")


if __name__ == "__main__":
    main()
