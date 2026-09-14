"""exp_021 aggregator — collect res_{arm}_seed{seed}.json into metrics.json.

Pure post-processing: reads only what the fit jobs already wrote, chooses
nothing on validation data, and re-states the pre-registered adoption criterion
verbatim (same sign on all 3 seeds AND |mean delta| >= 3 x control seed std).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/SungjinPark/experiments/exp_021")
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
DENOM = 0.2498069                    # 100000 * (1 - brier/DENOM) = board-scale


def score(b):
    return round(100000.0 * (1.0 - b / DENOM), 2)


def full_precision_2024(res):
    """Recompute fold_2024 Brier from the saved P10 predictions, unrounded.

    res_*.json stores Brier rounded to 1e-6, which is coarser than the seed
    spread we are trying to resolve (~3e-6).  The saved pred_*_fold_2024_P10.npy
    IS the reported prediction vector, so recomputing from it changes nothing
    except the number of digits.  Post-processing only: no model is refitted and
    nothing is selected on validation.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig",
                     usecols=["season", "game_type", "control_success"])
    va = (df["season"].to_numpy() == 2024)
    y = df["control_success"].to_numpy(dtype=np.float64)[va]
    is_r = (df["game_type"].to_numpy()[va] == "R")
    assert len(y) == 253507 and int(is_r.sum()) == 223497, (len(y), int(is_r.sum()))
    out = {}
    for (a, s), r in res.items():
        p = np.load(os.path.join(OUT_DIR, r["pred_files"]["P10"]))
        assert len(p) == len(y), (a, s, len(p))
        rounded = r["fold_2024"]["valid_all"]["brier"]
        b_all = float(np.mean((p - y) ** 2))
        assert abs(b_all - rounded) < 1e-6, (a, s, b_all, rounded)
        out[(a, s)] = {
            "valid_all": b_all,
            "valid_R": float(np.mean((p[is_r] - y[is_r]) ** 2)),
            "valid_F": float(np.mean((p[~is_r] - y[~is_r]) ** 2)),
        }
    return out


def main():
    res = {}
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "res_*.json"))):
        r = json.load(open(p))
        res[(r["arm"], r["seed"])] = r
    if not res:
        print("no res_*.json found", file=sys.stderr)
        sys.exit(1)

    arms = sorted({a for a, _ in res})
    seeds = sorted({s for _, s in res})
    shas = {k: v["sha256"] for k, v in res.items()}
    sha_train = {v["train.csv"] for v in shas.values()}
    sha_src = {v["train.py"] for v in shas.values()}
    assert len(sha_train) == 1, ("train.csv sha differs across jobs", sha_train)
    assert len(sha_src) == 1, ("train.py sha differs across jobs", sha_src)

    m = {
        "exp_id": "exp_021",
        "base_exp": "exp_010 D (47 provided columns, strongly regularised LGBM)",
        "primary_metric": "fold_2024 valid_all (R+F) Brier under P10",
        "score_formula": f"100000 * (1 - brier / {DENOM})",
        "arms_run": arms, "seeds_run": seeds,
        "sha256": {"train.csv": sha_train.pop(), "train.py": sha_src.pop()},
        "versions": next(iter(res.values()))["versions"],
        "lgb_params_seed42": next(iter(res.values()))["lgb_params"],
        "num_boost_round_cap": next(iter(res.values()))["num_boost_round_cap"],
        "mask": next(iter(res.values()))["mask"],
        "n_rows_masked": next(iter(res.values()))["n_rows_masked"],
        "arm_features": {a: res[(a, 42)]["features"] for a in arms
                         if (a, 42) in res},
        "arm_n_features": {a: res[(a, 42)]["n_features"] for a in arms
                           if (a, 42) in res},
        "categorical_declared": {a: res[(a, 42)]["categorical_declared"]
                                 for a in arms if (a, 42) in res},
    }

    if ("D", 42) in res:
        m["reproduction_gate"] = res[("D", 42)].get("reproduction_gate")

    # ---- per (arm, seed) fold table --------------------------------------- #
    per = {}
    for (a, s), r in res.items():
        e = {"n_features": r["n_features"], "elapsed_min": r["elapsed_min"],
             "num_boost_round_cap": r["num_boost_round_cap"],
             "categorical_declared": r["categorical_declared"],
             "oracle_iters": r["oracle_iters"],
             "argmin_at_cap": r["argmin_at_cap"],
             "any_argmin_at_cap": r["any_argmin_at_cap"],
             "unseen_category_rows": r["unseen_category_rows"],
             "constant_in_train_cols": r.get("constant_in_train_cols"),
             "pred_files": r["pred_files"]}
        for f in ("fold_2022", "fold_2023", "fold_2024"):
            for pol in ("", "_P4"):
                k = f + pol
                if k not in r:
                    continue
                v = r[k]
                e[k] = {
                    "iter_used": v["iter_used"], "iter_source": v["iter_source"],
                    "iter_requested": v.get("iter_requested"),
                    "iter_capped_by_num_boost_round":
                        v.get("iter_capped_by_num_boost_round"),
                    "oracle_flag": v["oracle_flag"], "n_train": v["n_train"],
                    "fit_min": v["fit_min"],
                    "argmin_at_cap": v["argmin_at_cap"],
                    "valid_all_brier": v["valid_all"]["brier"],
                    "valid_all_bss": v["valid_all"]["bss"],
                    "valid_all_auc": v["valid_all"].get("auc"),
                    "valid_all_score": score(v["valid_all"]["brier"]),
                    "valid_R_brier": v["valid_R"]["brier"],
                    "valid_R_auc": v["valid_R"].get("auc"),
                    "valid_F_brier": v["valid_F"]["brier"],
                    "valid_F_auc": v["valid_F"].get("auc"),
                    "policy_cost_vs_oracle": v["policy_cost_vs_oracle"],
                }
        per[f"{a}_seed{s}"] = e
    m["per_arm_seed"] = per

    # ---- stage 1 headline: fold_2024 valid_all, P10, seed 42 -------------- #
    main_arms = [a for a in arms if not a.endswith("_DIAG")]
    if all((a, 42) in res for a in main_arms):
        base = res[("D", 42)]["fold_2024"]["valid_all"]["brier"]
        t = {}
        for a in arms:
            if (a, 42) not in res:
                continue
            v = res[(a, 42)]["fold_2024"]["valid_all"]
            t[a] = {
                "n_features": res[(a, 42)]["n_features"],
                "iter_used": res[(a, 42)]["fold_2024"]["iter_used"],
                "fold_2023_argmin": res[(a, 42)]["oracle_iters"]["fold_2023"],
                "argmin_at_cap_any": res[(a, 42)]["any_argmin_at_cap"],
                "valid_all_brier": v["brier"], "valid_all_bss": v["bss"],
                "valid_all_auc": v.get("auc"),
                "score": score(v["brier"]),
                "delta_vs_D": round(v["brier"] - base, 8),
                "valid_R_brier": res[(a, 42)]["fold_2024"]["valid_R"]["brier"],
                "valid_F_brier": res[(a, 42)]["fold_2024"]["valid_F"]["brier"],
            }
        m["summary_stage1_seed42_fold_2024_valid_all_P10"] = t
        cand = [a for a in main_arms if a != "D"]
        if cand:
            m["best_arm_seed42"] = min(cand, key=lambda a: t[a]["valid_all_brier"])

    # ---- stage 2: paired delta vs D across seeds -------------------------- #
    fp = full_precision_2024(res)
    m["fold_2024_brier_full_precision_P10"] = {
        f"{a}_seed{s}": {kk: float(f"{vv:.12g}") for kk, vv in d.items()}
        for (a, s), d in sorted(fp.items(), key=lambda t: (t[0][0], t[0][1]))}
    m["precision_note"] = (
        "seed_summary uses UNROUNDED Brier recomputed from pred/*_fold_2024_"
        "_P10.npy; res_*.json stores 1e-6-rounded values, too coarse for a "
        "~3e-6 seed std.")
    by_arm = {}
    for a in arms:
        d = {str(s): fp[(a, s)]["valid_all"] for s in seeds if (a, s) in res}
        if d:
            by_arm[a] = d
    spread = {}
    for a, d in by_arm.items():
        vals = list(d.values())
        spread[a] = {"per_seed": {k: float(f"{v:.10g}") for k, v in d.items()},
                     "mean": round(float(np.mean(vals)), 8),
                     "mean_score": score(float(np.mean(vals))),
                     "std": round(float(np.std(vals, ddof=1)), 8) if len(vals) > 1
                            else None,
                     "range": round(float(max(vals) - min(vals)), 8),
                     "n_seeds": len(vals)}
    verdicts = {}
    if "D" in by_arm and spread["D"].get("std"):
        d_std = spread["D"]["std"]
        for a, d in by_arm.items():
            common = [s for s in d if s in by_arm["D"]]
            if a == "D" or len(common) < 2:
                continue
            ds = [d[s] - by_arm["D"][s] for s in sorted(common, key=int)]
            mean_d = float(np.mean(ds))
            same = bool(all(x < 0 for x in ds) or all(x > 0 for x in ds))
            verdicts[a] = {
                "vs": "D",
                "per_seed_delta": {s: round(d[s] - by_arm["D"][s], 8)
                                   for s in sorted(common, key=int)},
                "n_seeds": len(common),
                "mean_delta": round(mean_d, 8),
                "mean_delta_score_points": round(-100000.0 * mean_d / DENOM, 2),
                "std_delta": round(float(np.std(ds, ddof=1)), 8),
                "all_seeds_same_sign": same,
                "D_seed_std": d_std,
                "abs_mean_delta_over_D_std": round(abs(mean_d) / d_std, 2)
                                             if d_std > 0 else None,
                "criterion": "PRE-REGISTERED: measured := same sign on all 3 "
                             "seeds AND |mean delta| >= 3 x D seed std",
                "measured": bool(same and d_std > 0
                                 and abs(mean_d) >= 3 * d_std
                                 and len(common) == 3),
            }
    m["seed_summary"] = {"brier_by_seed_fold_2024_valid_all_P10": by_arm,
                         "spread": spread, "paired_delta_vs_D": verdicts}

    m["gain_pct_seed42"] = {a: res[(a, 42)]["fold_2024_gain_pct"]
                            for a in arms if (a, 42) in res}
    m["train_time_min_sum_of_jobs"] = round(
        sum(r["elapsed_min"] for r in res.values()), 2)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: m[k] for k in
                      ("reproduction_gate",
                       "summary_stage1_seed42_fold_2024_valid_all_P10",
                       "best_arm_seed42", "seed_summary") if k in m},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
