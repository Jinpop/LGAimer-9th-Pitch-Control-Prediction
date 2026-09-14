"""exp_048 — seed aggregation and the pre-registered decision."""

import json
import os

import numpy as np

EXP = os.path.expanduser("~/SungjinPark/experiments/exp_048")
RES = os.path.join(EXP, "res")
SEEDS = [42, 7, 2024]
ROUNDS = [655, 1000, 1400]
REF_BRIER = 0.2498069
SIGMA = 8.55113
THRESH = 25.65339


def s5(x):
    return float(f"{x:.5f}")


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def main():
    per = {s: json.load(open(os.path.join(RES, f"res_exp048_s1_seed{s}.json")))
           for s in SEEDS}
    gates = json.load(open(os.path.join(RES, "stage0_gates.json")))
    gate_e2 = json.load(open(os.path.join(RES, "stage0_gate_e2.json")))
    corr = json.load(open(os.path.join(RES, "stage1_corr_dform.json")))

    out = {
        "exp_id": "exp_048", "date": "2026-08-14",
        "title": ("teammate ins_* 16 columns into the R-branch stacker "
                  "(51 -> 67), fold_2024, stacker rounds fixed at 1400"),
        "scope": "stage 0 + stage 1 only.  No deployment package was built.",
        "score_conversion": {
            "formula": "100000 * (1 - Brier / 0.2498069)",
            "ref_brier": REF_BRIER, "decimals": 5,
            "sensitivity": ("Brier 1e-06 = 0.4003 points; a 6-decimal Brier "
                            "converts with +/-0.2002 of rounding uncertainty. "
                            "The Brier is the raw quantity."),
        },
        "unit": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
                 "n_train": per[42]["n_train_full"],
                 "n_valid": per[42]["n_valid"],
                 "n_valid_R": per[42]["n_valid_R"],
                 "n_valid_F": per[42]["n_valid_F"],
                 "arm_base": "C5_B3", "seeds": SEEDS,
                 "stacker_rounds": 1400},
    }

    # ---------------------------------------------------------------- gates
    out["stage0"] = {
        "pack_sha256_all_match": gates["pack_sha256"]["all_match"],
        "G0a_ast_row_independence": gates["G0a_ast_row_independence"]["PASS"],
        "G0a_apply_forbidden_calls":
            gates["G0a_ast_row_independence"]["per_function"]["apply"][
                "forbidden_hits"],
        "G0b_full_shuffle_identity": gates["G0b_full_shuffle_identity"],
        "G0c_isolated_rows": gates["G0c_isolated_rows"],
        "G0d_anchor_lt2024_equals_rebuild":
            gates["G0d_anchor_season_boundary"]["anchor_lt2024_vs_rebuild"][
                "all_bit_identical"],
        "G0d_deploy_anchor_equals_rebuild":
            gates["G0d_anchor_season_boundary"]["deploy_anchor_vs_rebuild"][
                "all_bit_identical"],
        "G0d_trap_demonstrated":
            gates["G0d_anchor_season_boundary"][
                "trap_deploy_anchor_on_2024_rows"],
        "G0e_columns_read_by_apply": gate_e2["columns_read_by_apply"],
        "G0e_target_read": gate_e2["target_read"],
        "G0e_PASS": gate_e2["PASS"],
        "G0e_superseded_criterion": gate_e2["superseded_criterion"],
        "G0e_first_attempt_numbers":
            gates["G0e_label_recovery_boundary"]["tuple_determinism"],
        "G0e_single_column_auc":
            {c: v["auc"] for c, v in
             gates["G0e_label_recovery_boundary"]["per_column"].items()},
        "G0f_missing_handling": gates["G0f_missing_handling"],
        "asof_origin_probe_2019": gates["asof_origin_probe_2019"],
        "unknown_pitcher_cross_check": gate_e2["unknown_pitcher_cross_check"],
    }
    out["stage0"]["ALL_PASS"] = bool(
        out["stage0"]["pack_sha256_all_match"]
        and out["stage0"]["G0a_ast_row_independence"]
        and gates["G0b_full_shuffle_identity"]["PASS"]
        and gates["G0c_isolated_rows"]["PASS"]
        and out["stage0"]["G0d_anchor_lt2024_equals_rebuild"]
        and out["stage0"]["G0d_deploy_anchor_equals_rebuild"]
        and out["stage0"]["G0e_PASS"])

    # -------------------------------------------------- A0 reproduction gate
    out["A0_reproduction_gate_vs_AH"] = {
        str(s): {k: per[s]["A0_reproduction_gate_vs_AH"][k]
                 for k in ("bit_identical", "abs_diff_brier",
                           "observed_pred_sha256", "ah_pred_sha256", "PASS")}
        for s in SEEDS}
    out["A0_reproduction_gate_vs_AH"]["ALL_BIT_IDENTICAL"] = all(
        per[s]["A0_reproduction_gate_vs_AH"]["bit_identical"] for s in SEEDS)

    # ------------------------------------------------------------- the arms
    fold = {}
    for r in ROUNDS:
        b0 = np.array([per[s]["arms"]["A0"]["by_round"][str(r)]
                       ["valid_all_brier"] for s in SEEDS])
        b1 = np.array([per[s]["arms"]["A1"]["by_round"][str(r)]
                       ["valid_all_brier"] for s in SEEDS])
        d = np.array([per[s]["delta_A1_minus_A0"][str(r)]
                      ["delta_score_converted"] for s in SEEDS])
        fold[str(r)] = {
            "A0_51col": {
                "per_seed_brier": {str(s): float(b0[i])
                                   for i, s in enumerate(SEEDS)},
                "mean_brier": float(b0.mean()),
                "mean_score": s5(score_of(b0.mean())),
                "per_seed_score": {str(s): s5(score_of(b0[i]))
                                   for i, s in enumerate(SEEDS)}},
            "A1_67col": {
                "per_seed_brier": {str(s): float(b1[i])
                                   for i, s in enumerate(SEEDS)},
                "mean_brier": float(b1.mean()),
                "mean_score": s5(score_of(b1.mean())),
                "per_seed_score": {str(s): s5(score_of(b1[i]))
                                   for i, s in enumerate(SEEDS)}},
            "delta_A1_minus_A0": {
                "per_seed_score": {str(s): s5(d[i])
                                   for i, s in enumerate(SEEDS)},
                "mean_delta_score": s5(float(d.mean())),
                "mean_delta_brier": float(b1.mean() - b0.mean()),
                "min_delta_score": s5(float(d.min())),
                "max_delta_score": s5(float(d.max())),
                "sd_delta_score_ddof1": s5(float(np.std(d, ddof=1))),
                "all_same_sign": bool(np.all(d > 0) or np.all(d < 0)),
            },
            "delta_by_segment": {
                "valid_R_mean_delta_score": s5(float(np.mean(
                    [per[s]["delta_A1_minus_A0"][str(r)]
                     ["delta_score_valid_R_only"] for s in SEEDS]))),
                "valid_F_mean_delta_score": s5(float(np.mean(
                    [per[s]["delta_A1_minus_A0"][str(r)]
                     ["delta_score_valid_F_only"] for s in SEEDS]))),
                "note": ("segment splits are auxiliary only -- clue S7: an F-side "
                         "change is not judged by valid_F."),
            },
        }
    out["fold_2024"] = fold

    # the headline the orchestrator's rule is applied to
    h = fold["1400"]
    out["headline_fold_2024_at_1400"] = {
        "brier_A0": h["A0_51col"]["mean_brier"],
        "score_A0": h["A0_51col"]["mean_score"],
        "brier_A1": h["A1_67col"]["mean_brier"],
        "score_A1": h["A1_67col"]["mean_score"],
        "delta_brier": h["delta_A1_minus_A0"]["mean_delta_brier"],
        "delta_score": h["delta_A1_minus_A0"]["mean_delta_score"],
    }
    out["control_matches_local_reference_AH"] = {
        "AH_recorded_mean_brier": 0.248132315233192,
        "AH_recorded_mean_score": 670.35169,
        "observed_mean_brier": h["A0_51col"]["mean_brier"],
        "observed_mean_score": h["A0_51col"]["mean_score"],
        "abs_diff_brier": abs(h["A0_51col"]["mean_brier"] - 0.248132315233192),
    }

    # ------------------------------------------------ pre-registered decision
    d1400 = np.array([per[s]["delta_A1_minus_A0"]["1400"]
                      ["delta_score_converted"] for s in SEEDS])
    out["pre_registered_decision"] = {
        "rule": ("all 3 seeds same sign AND |mean delta| >= 25.65339 points "
                 "(3 sigma, sigma = 8.55113).  Registered before the run; not "
                 "revised after seeing the numbers (clue S8)."),
        "sigma_score": SIGMA, "threshold_3sigma": THRESH,
        "control": "A0 (51 columns @1400)",
        "per_seed_delta_score": {str(s): s5(d1400[i])
                                 for i, s in enumerate(SEEDS)},
        "all_same_sign": bool(np.all(d1400 > 0) or np.all(d1400 < 0)),
        "mean_delta_score": s5(float(d1400.mean())),
        "abs_mean_delta_score": s5(float(abs(d1400.mean()))),
        "threshold_met": bool(abs(d1400.mean()) >= THRESH),
        "margin_over_threshold_in_sigma": s5(float(abs(d1400.mean()) / SIGMA)),
        "VERDICT": ("ADOPT" if (np.all(d1400 > 0) and abs(d1400.mean()) >= THRESH)
                    else ("REJECT (negative)" if np.all(d1400 < 0)
                          else "INCONCLUSIVE")),
    }

    # ----------------------------------------------------- gain of the 16
    imp = {}
    for s in SEEDS:
        imp[str(s)] = per[s]["arms"]["A1"]["importance_rank"]
    ranks = {}
    feats = list(per[42]["arms"]["A1"]["importance_rank"])
    for f in feats:
        rs = [imp[str(s)][f] for s in SEEDS]
        ranks[f] = {"rank_per_seed": rs, "mean_rank": s5(float(np.mean(rs))),
                    "gain_per_seed": [per[s]["arms"]["A1"][
                        "importance_PredictionValuesChange"][f] for s in SEEDS],
                    "mean_gain": s5(float(np.mean([
                        per[s]["arms"]["A1"][
                            "importance_PredictionValuesChange"][f]
                        for s in SEEDS])))}
    order = sorted(ranks, key=lambda f: ranks[f]["mean_rank"])
    ins_cols = [f for f in feats if f.startswith("ins_")]
    out["importance_A1_67col"] = {
        "type": "CatBoost PredictionValuesChange, averaged over 3 seeds",
        "top_20_overall": [{"feature": f, "mean_rank": ranks[f]["mean_rank"],
                            "mean_gain": ranks[f]["mean_gain"]}
                           for f in order[:20]],
        "the_16_ins_columns": [
            {"feature": f, "mean_rank": ranks[f]["mean_rank"],
             "mean_gain": ranks[f]["mean_gain"],
             "rank_per_seed": ranks[f]["rank_per_seed"]}
            for f in sorted(ins_cols, key=lambda f: ranks[f]["mean_rank"])],
        "best_ins_column": min(ins_cols, key=lambda f: ranks[f]["mean_rank"]),
        "best_ins_column_mean_rank": ranks[
            min(ins_cols, key=lambda f: ranks[f]["mean_rank"])]["mean_rank"],
        "teammate_claim_checked": ("the pack's README says ins_p_success was "
                                   "gain rank 1 on THEIR base; the rank it "
                                   "takes on OUR 67-column stacker is above."),
        "ins_p_success_mean_rank": ranks["ins_p_success"]["mean_rank"],
        "n_features": len(feats),
    }

    # ------------------------------------------------ overlap with exp_046
    out["overlap_with_exp046_d_form_eb"] = {
        "what": ("exp_046 built the same signal in our own code as `d_form_eb`. "
                 "Pearson r on the fold_2024 valid rows where both are defined."),
        "per_k": {k: {
            "pearson_r_vs_ins_p_success_dev":
                v["headline_vs_ins_p_success_dev"]["pearson_r"],
            "r_squared": v["headline_vs_ins_p_success_dev"]["r_squared"],
            "n_rows_both_defined":
                v["headline_vs_ins_p_success_dev"]["n_rows_both_defined"],
            "d_form_eb_defined_pct": v["d_form_eb_defined_pct"],
            "coverage": v["coverage"],
            "single_column_auc": v["single_column_auc_on_common_rows"],
            "corr_with_each_ins_col": v["corr_with_each_ins_col"],
        } for k, v in corr["per_k"].items()},
        "form_decomp_sha256": corr["sha256"]["form_decomp.py"],
    }

    # ----------------------------------------------------------- coverage
    out["coverage_valid_2024"] = per[42]["coverage_valid_2024"]
    out["ins_column_build"] = {
        "season_policy": per[42]["ins_columns"]["season_policy"],
        "frozen_anchor_equals_rebuild":
            per[42]["ins_columns"]["frozen_anchor_equals_rebuild"],
        "nan_pct_in_train_rows": per[42]["ins_columns"]["nan_pct_in_train_rows"],
        "nan_pct_in_valid_rows": per[42]["ins_columns"]["nan_pct_in_valid_rows"],
        "cols": per[42]["ins_columns"]["cols"],
    }

    # ------------------------------------------------------------ variables
    out["changed_vs_control"] = {
        "the_one": "stacker input columns 51 -> 67 (the 16 ins_* columns)",
        "carried_along_by_definition": [
            "stacker fit time per seed: A0 " + ", ".join(
                f"{per[s]['arms']['A0']['fit_min']:.2f}" for s in SEEDS)
            + " min -> A1 " + ", ".join(
                f"{per[s]['arms']['A1']['fit_min']:.2f}" for s in SEEDS)
            + " min",
            "stacker model size (more columns, more candidate splits)",
            ("season 2019 (211,627 rows, 18.96% of the fold's training set) "
             "carries NaN in all 16 columns -- it has no prior-season anchor.  "
             "This is a DESIGN CHOICE of this experiment, not part of the "
             "teammate pack; see ins_feats.py.  The 2024 validation rows carry "
             "no NaN."),
        ],
        "held_fixed": [
            "the four cell models and their OOF columns -- INHERITED byte-for-"
            "byte from candidate AH, not refitted (sha gate)",
            "stacker rounds = 1400, not re-chosen locally",
            "masks, fold definition, seeds, CatBoost params, library versions",
        ],
        "NOT_measured_here": [
            "the blend-level offset beta_f = 0.19 lives outside this branch; "
            "its interaction with the 16 columns is untouched and undecided",
            "deployment scale (this fold trains on 1,116,277 rows; a deployed "
            "package trains on 1,369,784) -- clue S1",
        ],
    }
    out["runtime_min"] = {str(s): per[s]["minutes"] for s in SEEDS}
    out["train_time_min"] = float(max(per[s]["minutes"] for s in SEEDS))
    out["inference_time_sec"] = None
    out["note_on_inference_time"] = (
        "not measured: no deployment package was built (out of scope).")

    p = os.path.join(EXP, "metrics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({
        "stage0_ALL_PASS": out["stage0"]["ALL_PASS"],
        "A0_gate_all_bit_identical":
            out["A0_reproduction_gate_vs_AH"]["ALL_BIT_IDENTICAL"],
        "headline": out["headline_fold_2024_at_1400"],
        "decision": out["pre_registered_decision"]["VERDICT"],
        "per_seed_delta":
            out["pre_registered_decision"]["per_seed_delta_score"],
        "best_ins_rank": out["importance_A1_67col"]["best_ins_column"],
    }, indent=1), flush=True)
    print("-> " + p, flush=True)


if __name__ == "__main__":
    main()
