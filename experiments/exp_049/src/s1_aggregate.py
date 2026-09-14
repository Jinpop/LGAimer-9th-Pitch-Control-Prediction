"""exp_049 — aggregate the three seeds into metrics.json and apply the rule.

The decision rule was registered in config.yaml BEFORE the run:
    sigma_score = 8.55113, threshold = 3 * sigma = 25.65339,
    adopt iff all three seeds move the same way AND |mean delta| >= threshold.
Neither number is recomputed from this experiment's spread.  The observed
per-seed spread is reported beside it as a separate quantity.
"""
import json
import os
import statistics
import sys

REF_BRIER = 0.2498069
SIGMA_SCORE = 8.55113
THRESHOLD = 25.65339
SEEDS = [42, 7, 2024]
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_049")
E048 = os.path.expanduser("~/SungjinPark/experiments/exp_048")
ARMS = ["A0", "A1", "L_no_p6", "L_no_idea", "L_no_i3"]
CAP = "1400"


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def r5(x):
    return float(f"{x:.5f}")


def main():
    per = {}
    for s in SEEDS:
        with open(os.path.join(OUT, "res", f"res_exp049_s1_seed{s}.json")) as f:
            per[s] = json.load(f)

    m = {"exp_id": "exp_049", "date": "2026-08-15",
         "title": ("team-mate Q's 14 target-independent columns (p6_* 6, "
                   "idea_matchup_pitch 1, i3_anom_* 7) into the R-branch "
                   "stacker (67 -> 81), fold_2024, rounds fixed at 1400"),
         "scope": "stage 0 + stage 1 only.  No deployment package was built.",
         "score_conversion": {
             "formula": "100000 * (1 - Brier / 0.2498069)",
             "ref_brier": REF_BRIER, "decimals": 5,
             "sensitivity": ("Brier 1e-06 = 0.4003 points; a 6-decimal Brier "
                             "converts with +/-0.2002 of rounding uncertainty. "
                             "The Brier is the raw quantity.")},
         "unit": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
                  "n_train": per[42]["n_train_full"],
                  "n_valid": per[42]["n_valid"],
                  "n_valid_R": per[42]["n_valid_R"],
                  "n_valid_F": per[42]["n_valid_F"],
                  "arm_base": "C5_B3", "seeds": SEEDS,
                  "stacker_rounds": 1400}}

    with open(os.path.join(OUT, "stage0_gates.json")) as f:
        m["stage0"] = json.load(f)

    m["A0_reproduction_gate_vs_exp048"] = {
        str(s): {k: v for k, v in
                 per[s]["A0_reproduction_gate_vs_exp048"].items()
                 if k != "what"} for s in SEEDS}
    m["A0_reproduction_gate_vs_exp048"]["ALL_BIT_IDENTICAL"] = bool(all(
        per[s]["A0_reproduction_gate_vs_exp048"]["bit_identical"]
        for s in SEEDS))

    fold = {}
    for r in ("655", "1000", "1400"):
        block = {}
        for a in ARMS:
            b = {str(s): per[s]["arms"][a]["by_round"][r]["valid_all_brier"]
                 for s in SEEDS}
            mb = sum(b.values()) / len(b)
            block[a] = {
                "per_seed_brier": b, "mean_brier": mb,
                "mean_score": r5(score_of(mb)),
                "per_seed_score": {k: r5(score_of(v)) for k, v in b.items()},
                "n_features": per[42]["arms"][a]["n_features"]}
        # deltas vs A0
        d = {}
        for a in ARMS[1:]:
            ps = {str(s): r5(-100000.0 * (
                block[a]["per_seed_brier"][str(s)]
                - block["A0"]["per_seed_brier"][str(s)]) / REF_BRIER)
                for s in SEEDS}
            vals = list(ps.values())
            d[a] = {"per_seed_score": ps,
                    "mean_delta_score": r5(sum(vals) / len(vals)),
                    "mean_delta_brier": (block[a]["mean_brier"]
                                         - block["A0"]["mean_brier"]),
                    "sd_delta_score_ddof1": r5(statistics.stdev(vals)),
                    "all_same_sign": bool(all(v > 0 for v in vals)
                                          or all(v < 0 for v in vals))}
        # LOO: what each family is worth = A1 minus (A1 without it)
        loo = {}
        for a, fam in (("L_no_p6", "p6_* (6)"),
                       ("L_no_idea", "idea_matchup_pitch (1)"),
                       ("L_no_i3", "i3_anom_* (7)")):
            ps = {str(s): r5(-100000.0 * (
                block["A1"]["per_seed_brier"][str(s)]
                - block[a]["per_seed_brier"][str(s)]) / REF_BRIER)
                for s in SEEDS}
            vals = list(ps.values())
            loo[fam] = {
                "removed_arm": a, "n_features_without": block[a]["n_features"],
                "per_seed_worth_score": ps,
                "mean_worth_score": r5(sum(vals) / len(vals)),
                "sd_ddof1": r5(statistics.stdev(vals)),
                "all_same_sign": bool(all(v > 0 for v in vals)
                                      or all(v < 0 for v in vals))}
        block["delta_vs_A0"] = d
        block["LOO_family_worth"] = loo
        fold[r] = block
    m["fold_2024"] = fold

    cap = fold[CAP]
    dA1 = cap["delta_vs_A0"]["A1"]
    m["headline_fold_2024_at_1400"] = {
        "control_A0_67col_mean_brier": cap["A0"]["mean_brier"],
        "control_A0_67col_mean_score": cap["A0"]["mean_score"],
        "treatment_A1_81col_mean_brier": cap["A1"]["mean_brier"],
        "treatment_A1_81col_mean_score": cap["A1"]["mean_score"],
        "mean_delta_score": dA1["mean_delta_score"],
        "mean_delta_brier": dA1["mean_delta_brier"]}

    # the control must equal exp_048's published A1
    with open(os.path.join(E048, "..", "exp_048", "metrics.json")) as f:
        e048m = json.load(f)
    ref = e048m["fold_2024"]["1400"]["A1_67col"]
    m["control_matches_exp048_A1"] = {
        "exp048_mean_brier": ref["mean_brier"],
        "exp049_A0_mean_brier": cap["A0"]["mean_brier"],
        "abs_diff": abs(ref["mean_brier"] - cap["A0"]["mean_brier"]),
        "per_seed_identical": {
            str(s): bool(ref["per_seed_brier"][str(s)]
                         == cap["A0"]["per_seed_brier"][str(s)])
            for s in SEEDS},
        "exp048_mean_score": ref["mean_score"]}

    m["pre_registered_decision"] = {
        "registered_in": "experiments/exp_049/config.yaml (before the run)",
        "sigma_score": SIGMA_SCORE, "threshold_multiple": 3,
        "threshold_score": THRESHOLD,
        "rule": ("adopt iff all three seeds agree in sign AND "
                 "|mean delta| >= 25.65339 points"),
        "control": "exp_048 A1 (67 columns @1400)",
        "observed_mean_delta_score": dA1["mean_delta_score"],
        "observed_all_same_sign": dA1["all_same_sign"],
        "observed_sigma_multiple": r5(abs(dA1["mean_delta_score"])
                                      / SIGMA_SCORE),
        "ADOPTED": bool(dA1["all_same_sign"]
                        and abs(dA1["mean_delta_score"]) >= THRESHOLD),
        "note": ("the threshold multiple (3) is a RULE and the observed sigma "
                 "multiple is DATA; they are different quantities.  Neither "
                 "sigma nor the threshold was recomputed after seeing the "
                 "result.")}

    m["delta_by_segment_A1"] = {
        "valid_R_mean_delta_score": r5(sum(
            per[s]["deltas"][CAP]["delta_score_valid_R_only_A1"]
            for s in SEEDS) / 3),
        "valid_F_mean_delta_score": r5(sum(
            per[s]["deltas"][CAP]["delta_score_valid_F_only_A1"]
            for s in SEEDS) / 3),
        "note": "segments are auxiliary only -- clue S7."}

    # importance of the 14 inside A1, averaged over seeds
    names = per[42]["new_columns"]["names"]
    n_feat = per[42]["arms"]["A1"]["n_features"]
    m["importance_A1_81col"] = {
        "n_features": n_feat,
        "new14_mean_rank": {c: round(sum(
            per[s]["arms"]["A1"]["importance_rank"][c] for s in SEEDS) / 3, 2)
            for c in names},
        "new14_mean_gain": {c: float(f"{sum(per[s]['arms']['A1']['importance_PredictionValuesChange'][c] for s in SEEDS) / 3:.6g}")
                            for c in names},
        "top15_seed42": dict(list(
            per[42]["arms"]["A1"]["importance_PredictionValuesChange"].items()
        )[:15])}

    m["correlation"] = {
        "why": ("p6_* is a predicted pitch mix and we already carry an "
                "observed as-of pitch mix; exp_047 rejected nine axes partly "
                "on inter-axis correlation, so it is measured, not assumed."),
        "new14_vs_existing_seed42": per[42]["corr_new14_vs_existing"],
        "max_abs_vs_pitchmix_seed42": per[42]["corr_max_abs_vs_pitchmix"],
        "within_new14_seed42": per[42]["corr_within_new14"]}

    m["new_columns"] = per[42]["new_columns"]

    m["changed_vs_control"] = {
        "the_one": ("R-branch stacker input 67 -> 81 columns: p6_* (6), "
                    "idea_matchup_pitch (1), i3_anom_* (7)"),
        "control": "exp_048 A1, 67 columns @1400, same three seeds",
        "unchanged": ["cell models frozen (47 features, inherited OOF, sha "
                      "checked, not refitted)",
                      "stacker rounds fixed at 1400",
                      "masks, fold definition, seeds, CatBoost parameters, "
                      "libraries",
                      "the 16 ins_* columns and their anchor policy"],
        "dragged_along": {
            "stacker_input_width": "67 -> 81 (the one)",
            "fit_time_min_per_arm": {
                a: round(sum(per[s]["arms"][a]["fit_min"] for s in SEEDS) / 3, 3)
                for a in ARMS},
            "new_column_missingness": ("the 14 are NaN for every 2019 row "
                                       "(237,413 of 1,116,277 training rows = "
                                       "21.27%) because no antecedent season "
                                       "exists; 0% on the 2024 validation "
                                       "rows.  No sentinel fill."),
            "feature_provenance": ("these 14 are NOT byte-identical to Q's 14. "
                                   "Q's pitch model is fitted through 2024 and "
                                   "reused for every season; ours is refitted "
                                   "per season boundary on <= s-1 trackman, "
                                   "through a pitch-type dictionary and a "
                                   "trackman<->train pitcher map that are both "
                                   "exp_037's reconstruction.  '77 columns = "
                                   "identical to Q' does not hold.")}}

    m["runtime_min"] = {str(s): per[s]["minutes"] for s in SEEDS}
    m["train_time_min"] = round(max(per[s]["minutes"] for s in SEEDS), 2)
    m["inference_time_sec"] = None
    m["note_on_inference_time"] = ("not measured: no deployment package is in "
                                   "scope, and the packager was validating "
                                   "candidate AJ on the same box, so a wall "
                                   "clock reading would be contended.")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)

    print(json.dumps({
        "A0_gate_all_bit_identical":
            m["A0_reproduction_gate_vs_exp048"]["ALL_BIT_IDENTICAL"],
        "control_matches_exp048": m["control_matches_exp048_A1"]["abs_diff"],
        "A0_mean_brier": cap["A0"]["mean_brier"],
        "A1_mean_brier": cap["A1"]["mean_brier"],
        "mean_delta_score": dA1["mean_delta_score"],
        "per_seed": dA1["per_seed_score"],
        "same_sign": dA1["all_same_sign"],
        "sigma_multiple": m["pre_registered_decision"]["observed_sigma_multiple"],
        "ADOPTED": m["pre_registered_decision"]["ADOPTED"],
        "LOO": {k: v["mean_worth_score"]
                for k, v in cap["LOO_family_worth"].items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
