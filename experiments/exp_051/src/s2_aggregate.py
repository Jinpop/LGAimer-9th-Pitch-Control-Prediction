"""exp_051 STAGE 2 — aggregate the three seeds, apply the PRE-REGISTERED rule.

Rule (registered in config before the run, clue S8):
    all 3 seeds same sign AND |mean delta| >= 25.65339 points
    (3 sigma, sigma = 8.55113), control = exp_048 A1 @1400
    (3-seed mean Brier read from exp_048's res files, not transcribed).
"""

import json
import os

REF_BRIER = 0.2498069
SIGMA = 8.55113
THRESHOLD = 25.65339
SEEDS = (42, 7, 2024)
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_051")
E48 = os.path.expanduser("~/SungjinPark/experiments/exp_048")
ROUNDS = ("655", "1000", "1400")
CELLS = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def main():
    r51, r48 = {}, {}
    for s in SEEDS:
        with open(os.path.join(OUT, "res",
                               f"res_exp051_u2024_seed{s}.json")) as f:
            r51[s] = json.load(f)
        with open(os.path.join(E48, "res",
                               f"res_exp048_s1_seed{s}.json")) as f:
            r48[s] = json.load(f)

    agg = {"exp_id": "exp_051", "stage": 2,
           "control": "exp_048 A1 (67-col stacker on 47-col cell OOF, @1400)",
           "treatment": "T1 (67-col stacker on 63-col cell OOF, @1400)"}

    # gates
    agg["gates"] = {str(s): {
        "A0_control_gate": r51[s]["A0_control_gate_vs_exp048_A1"]["PASS"],
        "A0_sha_match": r51[s]["A0_control_gate_vs_exp048_A1"]["sha_match"],
    } for s in SEEDS}
    assert all(v["A0_control_gate"] for v in agg["gates"].values())

    # per-round comparison
    by_round = {}
    for r in ROUNDS:
        per = {}
        for s in SEEDS:
            b1 = r51[s]["stacker_by_rounds"][
                "route_B_staged_from_cap_fit"][r]["valid_all_brier"]
            b0 = r48[s]["arms"]["A1"]["by_round"][r]["valid_all_brier"]
            per[str(s)] = {
                "brier_T1": b1, "brier_A0": b0,
                "delta_brier": b1 - b0,
                "delta_score": float(f"{-100000.0 * (b1 - b0) / REF_BRIER:.5f}"),
            }
        mean_b1 = sum(per[str(s)]["brier_T1"] for s in SEEDS) / 3
        mean_b0 = sum(per[str(s)]["brier_A0"] for s in SEEDS) / 3
        ds = [per[str(s)]["delta_score"] for s in SEEDS]
        mean_ds = sum(ds) / 3
        sd = (sum((d - mean_ds) ** 2 for d in ds) / 2) ** 0.5
        by_round[r] = {
            "per_seed": per,
            "mean_brier_T1": mean_b1, "mean_brier_A0": mean_b0,
            "mean_score_T1": float(f"{score_of(mean_b1):.5f}"),
            "mean_score_A0": float(f"{score_of(mean_b0):.5f}"),
            "mean_delta_score": float(f"{mean_ds:.5f}"),
            "sd_delta_score_ddof1": float(f"{sd:.5f}"),
            "all_same_sign": bool(all(d > 0 for d in ds)
                                  or all(d < 0 for d in ds)),
        }
    agg["by_round"] = by_round

    # the pre-registered decision, at 1400 only
    h = by_round["1400"]
    agg["pre_registered_decision"] = {
        "rule": ("all 3 seeds same sign AND |mean delta| >= 25.65339 points "
                 "(3 sigma, sigma = 8.55113).  Registered before the run; "
                 "not revised after seeing the numbers (clue S8)."),
        "sigma_score": SIGMA, "threshold_3sigma": THRESHOLD,
        "control": "exp_048 A1 @1400 (per-seed briers read from res files)",
        "per_seed_delta_score": {str(s): h["per_seed"][str(s)]["delta_score"]
                                 for s in SEEDS},
        "all_same_sign": h["all_same_sign"],
        "mean_delta_score": h["mean_delta_score"],
        "abs_mean_delta_score": abs(h["mean_delta_score"]),
        "threshold_met": bool(abs(h["mean_delta_score"]) >= THRESHOLD),
        "delta_in_sigma": float(f"{h['mean_delta_score'] / SIGMA:.5f}"),
    }
    d = agg["pre_registered_decision"]
    d["VERDICT"] = ("ADOPT" if (d["all_same_sign"] and d["threshold_met"]
                                and h["mean_delta_score"] > 0)
                    else "REJECT")

    # cells: 47col (AH) vs 63col self metrics + ins gain share
    cells = {}
    for c in CELLS:
        per = {}
        for s in SEEDS:
            cmpv = r51[s]["cells_vs_AH_47col"]["per_cell"][c]
            per[str(s)] = {
                "ah_47col": {k: cmpv["ah_47col_own"][k]
                             for k in ("brier", "auc", "logloss")},
                "exp051_63col": {k: cmpv["exp051_63col_own"][k]
                                 for k in ("brier", "auc", "logloss")},
                "argmin_iter_47col": cmpv["ah_argmin_iter"],
                "argmin_iter_63col": cmpv["exp051_argmin_iter"],
                "iter_used_pinned": cmpv["iter_used_pinned"],
                "ins_gain_share": r51[s]["cells"][c]["ins_gain_share"],
            }
        cells[c] = {
            "per_seed": per,
            "mean_ins_gain_share": float(f"{sum(per[str(s)]['ins_gain_share'] for s in SEEDS) / 3:.6g}"),
            "mean_auc_47": float(f"{sum(per[str(s)]['ah_47col']['auc'] for s in SEEDS) / 3:.6f}"),
            "mean_auc_63": float(f"{sum(per[str(s)]['exp051_63col']['auc'] for s in SEEDS) / 3:.6f}"),
            "mean_logloss_47": float(f"{sum(per[str(s)]['ah_47col']['logloss'] for s in SEEDS) / 3:.6f}"),
            "mean_logloss_63": float(f"{sum(per[str(s)]['exp051_63col']['logloss'] for s in SEEDS) / 3:.6f}"),
        }
    agg["cells_47_vs_63"] = cells

    # OOF movement
    oof = {}
    for c in CELLS:
        col = f"p_{c}"
        oof[col] = {seg: {
            "mean_abs_over_seeds": float(f"{sum(r51[s]['oof_delta_vs_AH']['per_column'][col][seg]['mean_abs'] for s in SEEDS) / 3:.6g}"),
            "p99_abs_over_seeds": float(f"{sum(r51[s]['oof_delta_vs_AH']['per_column'][col][seg]['p99_abs'] for s in SEEDS) / 3:.6g}"),
            "max_abs_over_seeds": max(r51[s]["oof_delta_vs_AH"]["per_column"][col][seg]["max_abs"] for s in SEEDS),
            "corr_new_old_mean": float(f"{sum(r51[s]['oof_delta_vs_AH']['per_column'][col][seg]['corr_new_old'] for s in SEEDS) / 3:.6g}"),
        } for seg in ("train_rows", "valid_rows")}
    agg["oof_movement_vs_AH"] = oof

    # runtime
    agg["runtime_min_per_seed"] = {str(s): r51[s]["minutes"] for s in SEEDS}
    agg["oof_min_per_seed"] = {str(s): r51[s]["oof_min"] for s in SEEDS}

    with open(os.path.join(OUT, "res", "summary.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("SUMMARY: " + json.dumps({
        "verdict": d["VERDICT"],
        "per_seed_delta": d["per_seed_delta_score"],
        "mean_delta": d["mean_delta_score"],
        "delta_in_sigma": d["delta_in_sigma"],
        "mean_brier_T1_1400": h["mean_brier_T1"]}))


if __name__ == "__main__":
    main()
