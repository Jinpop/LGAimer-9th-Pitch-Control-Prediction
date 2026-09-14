#!/usr/bin/env python
"""exp_039 — aggregation.

Nine Brier values for one configuration (R0, fold_2024, C5_B3, iterations
handed down from exp_037's stage-1 argmin), differing only in the fit seed.
This file turns them into the four things the experiment was registered to
answer, and refuses to say anything it was told not to say.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT
  * The sigma measured here is the spread of the FIT randomness alone.  The
    iteration counts are frozen across all nine seeds by design, so a sigma
    that also absorbed "the seed moved the argmin" would be larger and is NOT
    measured by this experiment.
  * n=9 gives sigma a relative standard error of 1/sqrt(2(n-1)) ~ 26%.  This
    experiment makes the denominator less wrong.  It does not fix it.
  * The adoption rule (3 seeds agreeing in sign AND |mean delta| >= control
    sigma x 3, ddof=1) is NOT touched here.  Only the estimate of sigma is.
  * The new sigma applies FORWARD ONLY.  exp_033 / exp_036 / exp_037 and the
    in-flight exp_038 keep the sigma they were registered against.  The
    counterfactual block below is labelled as such and is not a verdict.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

REF_BRIER = 0.2498069
PTS_PER_BRIER = 100000.0 / REF_BRIER      # 400309.27... points per unit Brier

INHERITED_SEEDS = [42, 7, 2024]
NEW_SEEDS = [1, 13, 101, 777, 2025, 31337]
E37 = os.path.expanduser("~/SungjinPark/experiments/exp_037")


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def r5(x):
    """max 5 decimals, no trailing zeros — the project's reporting rule."""
    return float(f"{round(float(x), 5):.5f}".rstrip("0").rstrip(".") or 0)


def sigma_ci(s, n, conf=0.95):
    """chi-square confidence interval for a normal sigma."""
    df = n - 1
    a = 1.0 - conf
    lo = s * np.sqrt(df / stats.chi2.ppf(1 - a / 2, df))
    hi = s * np.sqrt(df / stats.chi2.ppf(a / 2, df))
    return float(lo), float(hi)


def load_res(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.out)

    M = {"exp_id": "exp_039",
         "title": "the denominator of the adoption threshold, measured at n=9 "
                  "instead of n=3 — R0 / fold_2024 / C5_B3, seed the only "
                  "thing that moves",
         "conversion": {
             "formula": "100000 * (1 - brier / 0.2498069)",
             "reference_brier": REF_BRIER,
             "points_per_unit_brier": PTS_PER_BRIER,
             "sensitivity": "Brier 1e-6 = 0.4003 points, so a 6-digit Brier "
                            "carries +/-0.2 points of slack; the trailing "
                            "decimals of a converted value are a machine "
                            "transform of a rounded input, not a measurement",
         },
         "what_this_sigma_is": (
             "the spread of the FIT randomness only. iteration counts are "
             "frozen across all nine seeds (inherited from exp_037 stage-1), "
             "so the variation that a seed would add by moving the argmin is "
             "NOT included and is NOT measured by this experiment"),
         "scope": (
             "FORWARD ONLY. this sigma is registered for experiments opened "
             "AFTER exp_039. exp_033 / exp_036 / exp_037 and the in-flight "
             "exp_038 stand on the sigma they were registered against"),
         "rule_unchanged": (
             "3 seeds agreeing in sign AND |mean delta| >= control sigma x 3, "
             "ddof=1. exp_039 re-estimates sigma and changes nothing else"),
         }

    with open(os.path.join(root, "inherit_gate.json")) as f:
        gate = json.load(f)
    M["inherit_gate"] = {"PASS": gate["PASS"],
                         "failures": gate["failures"],
                         "file": "inherit_gate.json"}
    contract = gate["checks"].get("iter_contract")
    M["iter_contract"] = contract

    # ------------------------------------------------------------- collect 9
    seeds, missing, drift = {}, [], []
    for s in INHERITED_SEEDS + NEW_SEEDS:
        origin = "inherited (exp_037, not refitted)" if s in INHERITED_SEEDS \
                 else "exp_039 new fit"
        base = E37 if s in INHERITED_SEEDS else root
        p = os.path.join(base, "res", f"res_R0_u2024_seed{s}_st2.json")
        if not os.path.exists(p):
            missing.append(s)
            continue
        r = load_res(p)
        rec = {
            "origin": origin,
            "brier_valid_all": r["brier_full"],
            "score_valid_all": r5(score_of(r["brier_full"])),
            "stack_round": r["stack_round"],
            "cell_round": r["cell_round"],
            "scale_full": r["scale_full"],
            "scale_mode": r["scale_mode"],
            "base_json": r["base_json"],
            "valid_R_brier": r["valid_R"]["brier"],
            "valid_F_brier": r["valid_F"]["brier"],
            "auc": r["valid_all"]["auc"],
            "pred_mean": r["valid_all"]["pred_mean"],
            "minutes": r["minutes"],
            "pred_sha256": r["pred_sha256"],
            "b3_feat_sha256": r["sha256"]["b3_feat.py"],
            "train_csv_sha256": r["sha256"]["train.csv"],
            "n_train_full": r["n_train_full"], "n_valid": r["n_valid"],
            "res_path": p,
        }
        # every unit must have been handed the SAME iterations; that is the
        # premise of "seed is the only thing that varies".
        if contract:
            for k in ("stack_round", "cell_round", "scale_full", "scale_mode",
                      "base_json"):
                if rec[k] != contract[k]:
                    drift.append({"seed": s, "field": k,
                                  "got": rec[k], "expected": contract[k]})
        seeds[str(s)] = rec

    # code path identity: every unit must have run the same b3_feat.py bytes
    shas = sorted({v["b3_feat_sha256"] for v in seeds.values()})
    tsha = sorted({v["train_csv_sha256"] for v in seeds.values()})
    M["code_path_identity"] = {
        "b3_feat_sha256_distinct": shas,
        "train_csv_sha256_distinct": tsha,
        "single_code_path": len(shas) == 1 and len(tsha) == 1,
    }
    if len(shas) != 1 or len(tsha) != 1:
        drift.append({"seed": None, "field": "code/data sha",
                      "got": [shas, tsha], "expected": "one value each"})

    M["units"] = seeds
    M["missing_seeds"] = missing
    M["iter_drift"] = drift
    M["n_effective"] = len(seeds)
    if missing:
        M["missing_note"] = (
            "a failed unit REDUCES n. it was not replaced by a substitute "
            "seed — the six new seeds were frozen before any result was seen")

    order = [s for s in INHERITED_SEEDS + NEW_SEEDS if str(s) in seeds]
    b = np.array([seeds[str(s)]["brier_valid_all"] for s in order], float)
    n = b.size

    # ------------------------------------------------------- Q1: sigma at n=9
    s9 = float(np.std(b, ddof=1))
    m9 = float(b.mean())
    b3 = np.array([seeds[str(s)]["brier_valid_all"] for s in INHERITED_SEEDS
                   if str(s) in seeds], float)
    s3 = float(np.std(b3, ddof=1)) if b3.size >= 2 else None
    m3 = float(b3.mean()) if b3.size else None

    M["Q1_sigma"] = {
        "n": n,
        "seed_order": order,
        "brier_values": [float(x) for x in b],
        "score_values": [r5(score_of(x)) for x in b],
        "mean_brier": m9,
        "mean_score": r5(score_of(m9)),
        "sigma_brier_ddof1": s9,
        "sigma_score": r5(s9 * PTS_PER_BRIER),
        "min_brier": float(b.min()), "max_brier": float(b.max()),
        "range_brier": float(b.max() - b.min()),
        "range_score": r5((b.max() - b.min()) * PTS_PER_BRIER),
        "relative_standard_error_of_sigma": float(1 / np.sqrt(2 * (n - 1))),
        "n3_reference": {
            "seeds": INHERITED_SEEDS,
            "mean_brier": m3,
            "sigma_brier_ddof1": s3,
            "sigma_score": r5(s3 * PTS_PER_BRIER) if s3 else None,
            "relative_standard_error_of_sigma": float(1 / np.sqrt(2 * 2)),
        },
        "ratio_sigma9_over_sigma3": (s9 / s3) if s3 else None,
        "threshold_3sigma_brier": {"n9": 3 * s9,
                                   "n3": (3 * s3) if s3 else None},
        "threshold_3sigma_score": {"n9": r5(3 * s9 * PTS_PER_BRIER),
                                   "n3": r5(3 * s3 * PTS_PER_BRIER) if s3
                                   else None},
    }

    # ------------------------------------------------ Q2: jackknife on sigma
    jk = []
    for i, s in enumerate(order):
        rest = np.delete(b, i)
        sj = float(np.std(rest, ddof=1))
        jk.append({
            "dropped_seed": s,
            "dropped_brier": float(b[i]),
            "z_of_dropped": float((b[i] - m9) / s9) if s9 > 0 else None,
            "sigma_without_it": sj,
            "sigma_without_it_score": r5(sj * PTS_PER_BRIER),
            "pct_change_vs_full": float(100.0 * (sj - s9) / s9)
            if s9 > 0 else None,
        })
    jk_sorted = sorted(jk, key=lambda d: d["sigma_without_it"])
    M["Q2_jackknife"] = {
        "note": "sigma recomputed on the other n-1 seeds. a large NEGATIVE "
                "pct_change means that one seed was carrying the spread",
        "per_seed": jk,
        "most_influential_seed": jk_sorted[0]["dropped_seed"],
        "sigma_range_over_jackknife": [jk_sorted[0]["sigma_without_it"],
                                       jk_sorted[-1]["sigma_without_it"]],
    }

    # ------------------------------------------------- Q3: chi-square CI on sigma
    lo9, hi9 = sigma_ci(s9, n, 0.95)
    ci = {"n9_95pct": {"df": n - 1, "lo": lo9, "hi": hi9,
                       "lo_score": r5(lo9 * PTS_PER_BRIER),
                       "hi_score": r5(hi9 * PTS_PER_BRIER),
                       "width_ratio_hi_over_lo": hi9 / lo9}}
    if s3:
        lo3, hi3 = sigma_ci(s3, b3.size, 0.95)
        ci["n3_95pct"] = {"df": b3.size - 1, "lo": lo3, "hi": hi3,
                          "lo_score": r5(lo3 * PTS_PER_BRIER),
                          "hi_score": r5(hi3 * PTS_PER_BRIER),
                          "width_ratio_hi_over_lo": hi3 / lo3}
        overlap = not (hi9 < lo3 or hi3 < lo9)
        ci["intervals_overlap"] = bool(overlap)
        ci["n3_point_inside_n9_ci"] = bool(lo9 <= s3 <= hi9)
        ci["n9_point_inside_n3_ci"] = bool(lo3 <= s9 <= hi3)
        ci["reading"] = (
            "the intervals overlap, so there is no ground to say the n=3 "
            "estimate was WRONG; what changed is how tightly sigma is pinned"
            if overlap else
            "the intervals are disjoint: the n=3 estimate is outside what "
            "n=9 supports")
    # the historical spread the experiment was called to explain
    hist = {"exp_030_control": 4.897825479068653e-06,
            "exp_036_argmin": 3.491428813285408e-05,
            "exp_037_R0": 3.881562551959927e-05}
    ci["historical_sigmas_vs_n9_ci"] = {
        k: {"sigma": v, "sigma_score": r5(v * PTS_PER_BRIER),
            "inside_n9_95ci": bool(lo9 <= v <= hi9)}
        for k, v in hist.items()}
    M["Q3_confidence_interval"] = ci

    # ------------------------------------------------------- Q4: shape, only
    sh = stats.shapiro(b) if n >= 3 else None
    M["Q4_distribution_shape_OBSERVATION_ONLY"] = {
        "warning": "recorded, not concluded. n=9 has almost no power to "
                   "reject normality, and this experiment was registered not "
                   "to draw a conclusion here. a skew finding opens a "
                   "SEPARATE question about whether an sd-based threshold is "
                   "the right instrument; it does not answer it",
        "sorted_brier": sorted(float(x) for x in b),
        "z_scores_sorted": sorted(float((x - m9) / s9) for x in b) if s9 > 0
        else None,
        "mean_minus_median": float(m9 - np.median(b)),
        "skew_g1": float(stats.skew(b, bias=False)),
        "excess_kurtosis_g2": float(stats.kurtosis(b, bias=False)),
        "shapiro_W": float(sh.statistic) if sh else None,
        "shapiro_p": float(sh.pvalue) if sh else None,
        "iqr_brier": float(np.subtract(*np.percentile(b, [75, 25]))),
        "mad_brier": float(np.median(np.abs(b - np.median(b)))),
    }

    # ----------------------------------------- counterfactual, NOT a verdict
    M["counterfactual_NOT_A_VERDICT"] = {
        "warning": "ILLUSTRATIVE ONLY. do not put this column next to a "
                   "verdict column, and do not re-judge exp_038 with it. "
                   "choosing a denominator after seeing a numerator is the "
                   "same error as the exp_033 / exp_036 ddof incident",
        "threshold_score_under_n3_sigma": r5(3 * s3 * PTS_PER_BRIER)
        if s3 else None,
        "threshold_score_under_n9_sigma": r5(3 * s9 * PTS_PER_BRIER),
        "threshold_moved_by_score": r5(3 * (s9 - s3) * PTS_PER_BRIER)
        if s3 else None,
    }

    mins = [v["minutes"] for v in seeds.values()
            if v["origin"].startswith("exp_039")]
    M["train_time_min_new_units_sum"] = round(float(sum(mins)), 2)
    M["new_unit_minutes"] = mins

    # the project's standard fold block, so the file is readable by the
    # existing tooling.  only fold_2024 exists in this experiment.
    M["fold_2024"] = {
        "brier": m9,
        "score": r5(score_of(m9)),
        "bss": float(1 - m9 / REF_BRIER),
        "auc": float(np.mean([v["auc"] for v in seeds.values()])),
        "n_seeds": n,
        "sigma_brier_ddof1": s9,
        "note": "mean over the seeds; this experiment measures the spread, "
                "not a new model level",
    }

    with open(os.path.join(root, "metrics.json"), "w") as f:
        json.dump(M, f, indent=1)
    print(json.dumps({"n": n, "sigma9": s9, "sigma3": s3,
                      "ratio": (s9 / s3) if s3 else None,
                      "missing": missing, "drift": len(drift)}), flush=True)


if __name__ == "__main__":
    main()
