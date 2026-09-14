"""exp_032 — assemble metrics.json, and put EVERY number the report quotes
into a `report_derived` block that can be regenerated and compared.

exp_030 shipped a BSS of 0.007937 in its report that existed nowhere in its
artefacts.  The rule since then is that a number may appear in prose only if it
appears in a file, so this script computes the report's derived quantities from
the artefacts, writes them down, and then re-derives them a second time by an
independent route wherever an independent route exists.  Disagreements are
written into `report_derived.self_check`, not silently reconciled.
"""

import hashlib
import json
import os
import sys

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from labels import FAIL_CELLS                                     # noqa: E402

OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_032")
EXP031 = os.path.expanduser("~/SungjinPark/experiments/exp_031")
EXP030 = os.path.expanduser("~/SungjinPark/experiments/exp_030")
REF = 0.2498069
FOLD_SEEDS = [42, 7, 2024]
BLOCK_SEEDS = [42, 7]


def sc(b):
    return round(100000.0 * (1.0 - b / REF), 5)


def sd(db):
    return round(-100000.0 * float(db) / REF, 5)


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            x = f.read(block)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def main():
    G = json.load(open(os.path.join(OUT_DIR, "metrics_grid.json")))
    F = json.load(open(os.path.join(OUT_DIR, "curves", "iter_fit.json")))
    M31 = json.load(open(os.path.join(EXP031, "metrics.json")))
    assert G["exp_id"] == "exp_032" and F["exp_id"] == "exp_032"

    res = {}
    for stem in ([f"C5B3_u2021_seed{s}_st3" for s in BLOCK_SEEDS]
                 + ["C5B3_u2023_seed42_st3"]
                 + [f"C5B3_u2024_seed{s}_st4" for s in FOLD_SEEDS]):
        res[stem] = json.load(open(os.path.join(OUT_DIR, "res",
                                                f"res_{stem}.json")))

    # ---- 1. block 2021, the third observation ------------------------------- #
    blk21 = {str(s): {
        "brier": res[f"C5B3_u2021_seed{s}_st3"]["brier_full"],
        "score": sc(res[f"C5B3_u2021_seed{s}_st3"]["brier_full"]),
        "cell_argmin": res[f"C5B3_u2021_seed{s}_st3"]["cell_argmin_iter"],
        "stack_argmin": res[f"C5B3_u2021_seed{s}_st3"]["stack_argmin_iter"],
        "any_argmin_at_cap": res[f"C5B3_u2021_seed{s}_st3"]["any_argmin_at_cap"],
        "cap": res[f"C5B3_u2021_seed{s}_st3"]["cap"],
        "minutes": res[f"C5B3_u2021_seed{s}_st3"]["minutes"],
        "n_train_full": res[f"C5B3_u2021_seed{s}_st3"]["n_train_full"],
        "n_train_mode": res[f"C5B3_u2021_seed{s}_st3"]["n_train_mode"],
        "nan_stack_pct": res[f"C5B3_u2021_seed{s}_st3"][
            "nan_stack_rows_in_train_pct"]} for s in BLOCK_SEEDS}

    # ---- 2. the argmin table across all four sizes -------------------------- #
    table = {"stacker": {}, **{c: {} for c in FAIL_CELLS}}
    for tgt in table:
        for o in F["targets"][tgt]["observations"]:
            table[tgt][f"{o['n']}_seed{o['seed']}"] = o["argmin"]
        table[tgt]["size_means"] = {k: round(v, 1) for k, v in
                                    F["targets"][tgt]["size_means_argmin"].items()}
        table[tgt]["exp030_row_ratio_policy_would_say"] = (
            F["policy"]["exp030_policy_for_contrast"]["stack_iter"]
            if tgt == "stacker" else
            F["policy"]["exp030_policy_for_contrast"]["cell_iter"][tgt])
        table[tgt]["exp032_fitted_law_says"] = (
            F["targets"][tgt]["fold_2024_iter_USED"])
        table[tgt]["selected_form"] = F["targets"][tgt]["SELECTED_FORM"]
        table[tgt]["loso_total_abs_err_log"] = {
            m: round(F["targets"][tgt]["fits"][m][
                "LOSO_TOTAL_ABS_ERR_LOG_SELECTION_CRITERION"], 5)
            for m in F["targets"][tgt]["fits"]}
        table[tgt]["loglog_slope"] = F["targets"][tgt]["fits"]["loglog"][
            "params"]["b_slope"]
        table[tgt]["loglog_max_residual_ratio"] = F["targets"][tgt]["fits"][
            "loglog"]["max_residual_ratio"]
        table[tgt]["loglog_r2_in_log_space"] = F["targets"][tgt]["fits"][
            "loglog"]["r2_in_log_space"]
        st = F["targets"][tgt]["straightness_of_the_three_size_means"]
        table[tgt]["curvature_log"] = st["curvature_log"]
        table[tgt]["curvature_as_ratio"] = st["curvature_as_ratio"]
        table[tgt]["max_seed_spread_log"] = st["max_seed_spread_log"]
        table[tgt]["curvature_over_seed_noise"] = st[
            "curvature_over_max_seed_spread"]
        table[tgt]["straightness_verdict"] = st["verdict"]
        # how wrong was the extrapolation, judged AFTER the fact by the new
        # fold_2024 unit's own curve?  Diagnostic only; it selected nothing.
        obs24 = [res[f"C5B3_u2024_seed{s}_st4"]["stack_argmin_iter"]
                 if tgt == "stacker" else
                 res[f"C5B3_u2024_seed{s}_st4"]["cell_argmin_iter"][tgt]
                 for s in FOLD_SEEDS]
        used = table[tgt]["exp032_fitted_law_says"]
        old = table[tgt]["exp030_row_ratio_policy_would_say"]
        table[tgt]["fold2024_own_argmin_AFTER_THE_FACT"] = dict(
            zip([str(s) for s in FOLD_SEEDS], obs24))
        table[tgt]["fold2024_own_argmin_mean"] = float(np.mean(obs24))
        table[tgt]["new_policy_over_own_argmin_mean"] = used / float(np.mean(obs24))
        table[tgt]["old_policy_over_own_argmin_mean"] = old / float(np.mean(obs24))
        table[tgt]["new_policy_is_closer_than_old"] = bool(
            abs(np.log(used / np.mean(obs24))) < abs(np.log(old / np.mean(obs24))))

    # ---- 3. base, seed std, offset, adoption -------------------------------- #
    B = G["base_no_offset_fold_2024"]
    H = G["SEED_STD_MEASUREMENT_HEADLINE"]
    A = G["adoption"]
    BC = G["base_change_adoption"]
    derived = {
        "block_2021_third_point": blk21,
        "argmin_vs_rows_table": table,
        "iteration_fit": {
            "selection_rule": F["selection_rule"],
            "per_target_form": {t: F["targets"][t]["SELECTED_FORM"]
                                for t in F["targets"]},
            "fold_2024_iter_used": F["policy"]["for_stage_C_fold_2024"],
            "deploy_iter_recorded": F["policy"]["for_stage_E_deploy_RECORD_ONLY"],
            "extrapolation_leverage": {
                t: F["targets"][t]["extrapolation_leverage_in_log_x_units"]
                for t in F["targets"]},
            "guard_fired_any": bool(any(F["targets"][t]["guard_fired"]
                                        for t in F["targets"])),
            "deploy_guard_fired_any": bool(any(F["targets"][t]["deploy_guard_fired"]
                                               for t in F["targets"])),
        },
        "new_base_fold_2024": {
            "brier_by_seed": B["by_seed"], "score_by_seed": B["by_seed_score"],
            "mean_brier": B["mean_brier"], "mean_score": B["mean_score"],
            "seed_std": B["seed_std"],
            "seed_std_score_points": B["seed_std_score_points"],
            "old_base_mean_score_exp031": sc(
                M31["base_no_offset_fold_2024"]["mean_brier"]),
            "old_base_seed_std_exp031": M31["base_no_offset_fold_2024"]["seed_std"],
            "score_gain_of_new_base_vs_old": BC["mean_delta_score_points"],
        },
        "seed_std_headline": H,
        "base_change_adoption": BC,
        "offset_adoption_on_new_base": A,
        "threshold_vs_std_naming": {
            "warning": "two different multiples get quoted in this project and "
                       "they differ by exactly 3x.  They are NOT comparable "
                       "across experiments unless both are the same one.",
            "offset_MULTIPLE_OF_THE_THRESHOLD": A["MULTIPLE_OF_THE_THRESHOLD"],
            "offset_MULTIPLE_OF_THE_CONTROL_STD": A["MULTIPLE_OF_THE_CONTROL_STD"],
            "base_change_MULTIPLE_OF_THE_THRESHOLD": BC["MULTIPLE_OF_THE_THRESHOLD"],
            "base_change_MULTIPLE_OF_THE_CONTROL_STD": BC[
                "MULTIPLE_OF_THE_CONTROL_STD"],
            "exp031_offset_MULTIPLE_OF_THE_THRESHOLD": M31["adoption"][
                "MULTIPLE_OF_THE_THRESHOLD"],
        },
        "runtime_minutes": {
            **{f"block2021_seed{s}": res[f"C5B3_u2021_seed{s}_st3"]["minutes"]
               for s in BLOCK_SEEDS},
            "gate_block2023_seed42": res["C5B3_u2023_seed42_st3"]["minutes"],
            **{f"fold2024_seed{s}": res[f"C5B3_u2024_seed{s}_st4"]["minutes"]
               for s in FOLD_SEEDS},
            "grid_compute_min": G["compute_time_min"],
        },
    }

    # ---- self-check: re-derive the headline numbers a second way ------------ #
    chk = {}
    bb = [B["by_seed"][str(s)] for s in FOLD_SEEDS]
    chk["mean_brier_recomputed"] = float(np.mean(bb))
    chk["mean_brier_matches"] = bool(abs(np.mean(bb) - B["mean_brier"]) < 1e-15)
    chk["seed_std_recomputed"] = float(np.std(bb, ddof=1))
    chk["seed_std_matches"] = bool(abs(np.std(bb, ddof=1) - B["seed_std"]) < 1e-15)
    chk["mean_score_recomputed"] = sc(float(np.mean(bb)))
    chk["mean_score_matches"] = bool(sc(float(np.mean(bb))) == B["mean_score"])
    chk["std_ratio_recomputed"] = float(B["seed_std"]
                                        / M31["base_no_offset_fold_2024"]["seed_std"])
    chk["std_ratio_matches"] = bool(abs(chk["std_ratio_recomputed"]
                                        - H["ratio_new_over_old"]) < 1e-12)
    # every prediction vector on disk still hashes to what its record says
    chk["pred_sha_reverified"] = {}
    for stem, r in res.items():
        p = os.path.join(OUT_DIR, r["pred_file"])
        chk["pred_sha_reverified"][stem] = bool(sha256_file(p) == r["pred_sha256"])
    chk["all_pred_sha_ok"] = all(chk["pred_sha_reverified"].values())
    chk["all_checks_pass"] = all(v for k, v in chk.items()
                                 if k.endswith("_matches") or k == "all_pred_sha_ok")
    derived["self_check"] = chk

    r24 = {s: res[f"C5B3_u2024_seed{s}_st4"] for s in FOLD_SEEDS}
    metrics = {
        "exp_id": "exp_032",
        "title": G["title"],
        "primary_metric": "fold_2024 valid_all (R+F) Brier",
        "conversion": {"formula": "score = 100000 * (1 - brier / 0.2498069)",
                       "sensitivity_1e-6_brier": 0.4003,
                       "caveat": "every Brier here is full precision; the "
                                 "+-0.20 rounding caveat does NOT apply"},
        "fold_2022": {"note": "not run in exp_032; block 2022 is REUSED from "
                              "exp_031 as beta supply, byte-identical",
                      "brier_by_seed": {str(s): M31["block_provenance"]
                                        [f"block_2022_seed{s}"]["brier"]
                                        for s in BLOCK_SEEDS}},
        "fold_2023": {"note": "block 2023 seed 42 was RE-RUN here as the "
                              "bit-identity gate; seed 7 is reused from exp_031",
                      "brier_by_seed": {
                          "42": res["C5B3_u2023_seed42_st3"]["brier_full"],
                          "7": M31["block_provenance"]["block_2023_seed7"]["brier"]},
                      "gate_bit_identical": G["gate1_block_bit_identity"][
                          "bit_identical_to_exp031"]},
        "fold_2021_new_block": {
            "note": "the third observation point created by this experiment",
            "brier_by_seed": {s: blk21[s]["brier"] for s in blk21},
            "score_by_seed": {s: blk21[s]["score"] for s in blk21}},
        "fold_2024": {
            "brier": B["mean_brier"], "score": B["mean_score"],
            "bss": r24[42]["valid_all"]["bss"],
            "auc": r24[42]["valid_all"]["auc"],
            "bss_note": "bss and auc are seed 42's, from its own res file's "
                        "valid_all block; brier and score are the 3-seed mean",
            "bss_by_seed": {str(s): r24[s]["valid_all"]["bss"] for s in FOLD_SEEDS},
            "auc_by_seed": {str(s): r24[s]["valid_all"]["auc"] for s in FOLD_SEEDS},
            "brier_by_seed": B["by_seed"], "score_by_seed": B["by_seed_score"],
            "seed_std": B["seed_std"],
            "valid_R": {str(s): r24[s]["valid_R"] for s in FOLD_SEEDS},
            "valid_F": {str(s): r24[s]["valid_F"] for s in FOLD_SEEDS},
            "iter_used": {"cell": r24[42]["cell_iter_used"],
                          "stack": r24[42]["stacker"]["iter_used"]},
        },
        "changed_variables_vs_exp031": [
            "1. fold_2024 base iteration counts: exp_030's row-ratio policy "
            "(cells 577/1346/449/897, stacker 481) -> exp_032's per-target "
            "fitted law.  This is the intended change.",
            "2. a THIRD block unit (2021, 432,501 full rows) now exists and is "
            "an input to the fit.  It did not exist in exp_031.",
            "3. block 2023 seed 42 was RE-RUN under exp_032's code (bit-identity "
            "gate).  Its vector is identical, so it changes no number.",
            "4. UNCHANGED and reused byte-for-byte: the beta supply (blocks 2022 "
            "and 2023 at their own argmin), the C5 spec, the 47 features, the "
            "k grid, the d form, labels.py and every inherited module.",
            "5. CONSEQUENTLY CHANGED without being chosen: the fold_2024 base's "
            "seed std, and therefore the adoption threshold every offset cell "
            "is measured against.",
            "6. wall-clock contention differed (3-wide here vs 4-wide in "
            "exp_031's stage A); CatBoost's thread_count stayed 6, and the "
            "bit-identity gate is what shows contention did not move results.",
        ],
        "report_derived": derived,
        "grid_metrics_file": "metrics_grid.json",
        "iter_fit_file": "curves/iter_fit.json",
        "source_sha256": G["source_sha256"],
        "train_csv_sha256": G["train_csv_sha256"],
        "versions": G["versions"],
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({"self_check": chk["all_checks_pass"],
                      "seed_std_new": B["seed_std"],
                      "seed_std_old": M31["base_no_offset_fold_2024"]["seed_std"],
                      "ratio": chk["std_ratio_recomputed"],
                      "base_score_new": B["mean_score"],
                      "base_score_old": sc(M31["base_no_offset_fold_2024"]
                                           ["mean_brier"]),
                      "offset_adopted": A["ADOPTED"],
                      "base_change_adopted": BC["ADOPTED"]}, indent=1))


if __name__ == "__main__":
    main()
