"""exp_052 stage 0.5 — re-run ONLY the logit-space projection at a raised cap.

The first pass put the logit projection's argmin exactly on cap 2000.  Under
the cap rule (`실행시간_실측.md` #3, exp_017) an argmin that equals the cap is
not an argmin, it is a truncation, so that fit is INVALID and must be re-run
with a higher cap.  Nothing else in the probe is touched: the probability-space
projection (argmin 1825 < 2000) and all of P-B were uncensored and stand.

The teacher prediction is READ from disk and its sha256 is asserted against the
value the first pass recorded, so this run cannot silently use a different
teacher.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
E027 = os.path.expanduser("~/SungjinPark/experiments/exp_027/src/train.py")
BASE_CAT = ["top_bottom", "game_type", "base_state"]
N_THREADS = 6
SEED = 42
FRONT_FRAC = 0.60
NEW_CAP = 6000
EVAL_PERIOD = 25
EPS = 1e-6


def main():
    t0 = time.time()
    import catboost as cb
    import importlib.util
    spec = importlib.util.spec_from_file_location("e027", E027)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    CB = dict(m.CB_BASE)

    probe_path = os.path.join(BASE, "res", "stage05_probe.json")
    probe = json.load(open(probe_path))
    assert probe["P_A"]["logit"]["argmin_at_cap_INVALID"] is True

    t_prob = np.load(os.path.join(BASE, "work", "teacher_D_2023_at700.npy"))
    got = hashlib.sha256(t_prob.tobytes()).hexdigest()
    assert got == probe["teacher"]["pred_sha256"], ("teacher changed", got)

    cols = json.load(open(os.path.join(BASE, "work", "cols.json")))
    student67 = cols["base47"] + cols["oof4"] + cols["ins16"]
    df = pd.read_parquet(os.path.join(BASE, "work", "design_fold2023.parquet"))
    d23 = df.loc[~df["_is_train"].to_numpy(bool)].reset_index(drop=True)
    has_tm = d23["has_tm"].to_numpy(np.float64) > 0.5
    n23 = len(d23)
    n_front = int(round(n23 * FRONT_FRAC))
    front = np.zeros(n23, bool)
    front[:n_front] = True
    back = ~front

    catS = [c for c in BASE_CAT if c in student67]
    data = {}
    for c in student67:
        if c in catS:
            data[c] = d23[c].astype(str).to_numpy()
        else:
            data[c] = pd.to_numeric(d23[c], errors="raise").to_numpy(np.float64)
    FS = pd.DataFrame(data, columns=student67)
    cis = [student67.index(c) for c in catS]

    c = np.clip(t_prob, EPS, 1 - EPS)
    z = np.log(c / (1 - c))
    pool_ft = cb.Pool(FS.loc[front], label=z[front], cat_features=cis)
    pool_b = cb.Pool(FS.loc[back], cat_features=cis)
    reg = cb.CatBoostRegressor(**dict(CB, loss_function="RMSE",
                                      random_seed=SEED, iterations=NEW_CAP))
    reg.fit(pool_ft)
    curve = {}
    for k, p in enumerate(reg.staged_predict(
            pool_b, ntree_start=0, ntree_end=NEW_CAP,
            eval_period=EVAL_PERIOD, thread_count=N_THREADS)):
        curve[(k + 1) * EVAL_PERIOD] = float(
            np.mean((p.astype(np.float64) - z[back]) ** 2))
    best_n = min(curve, key=curve.get)
    mse = curve[best_n]
    var_b = float(np.var(z[back]))
    p_b = reg.predict(pool_b, ntree_end=best_n).astype(np.float64)
    p_in = reg.predict(pool_ft, ntree_end=best_n).astype(np.float64)
    msk = has_tm[back]

    new = {
        "cap": NEW_CAP, "argmin_iter": int(best_n),
        "argmin_at_cap_INVALID": bool(best_n == NEW_CAP),
        "Var_t_all2023": float(np.var(z)),
        "Var_t_back40": var_b,
        "Var_t_given_x_holdout_MSE": mse,
        "RMSE_holdout": float(np.sqrt(mse)),
        "R2_holdout": 1.0 - mse / var_b,
        "R2_in_sample_reference": 1.0 - float(np.mean(
            (p_in - z[front]) ** 2)) / float(np.var(z[front])),
        "Var_t_given_x_on_has_tm_rows": float(
            np.mean((p_b[msk] - z[back][msk]) ** 2)),
        "supersedes": {"cap": 2000, "reason": "argmin landed on cap (invalid)",
                       "superseded_R2_holdout":
                           probe["P_A"]["logit"]["R2_holdout"]},
    }
    probe["P_A"]["logit_CENSORED_cap2000"] = probe["P_A"].pop("logit")
    probe["P_A"]["logit"] = new
    probe["logit_recap_elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(probe_path, "w") as f:
        json.dump(probe, f, indent=1, ensure_ascii=False)
    print(json.dumps(new, indent=1))


if __name__ == "__main__":
    main()
