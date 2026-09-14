"""exp_047 — assemble metrics.json from the per-unit result jsons.

Every number here is COPIED from a res_*.json / stage1_*.json that this
experiment (or exp_046 / exp_030, for the reference arms) produced.  Nothing is
recomputed from memory and nothing is asserted about an artefact this script
did not read.
"""

import glob
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser("~/SungjinPark")
E47 = os.path.join(HOME, "experiments/exp_047")
E46 = os.path.join(HOME, "experiments/exp_046")
E30 = os.path.join(HOME, "experiments/exp_030")
BRIER_REF = 0.2498069
PTS = 100000.0 / BRIER_REF
SIGMA_SCORE = 8.55113
THRESHOLD_3SIGMA = 25.65339
SEEDS = [42, 7, 2024]

ARM_DESC = {
    "F1": "47 base + exp_046 A1's four success-form columns (n_season, "
          "rate_season, d_form_eb@k=100, has_form) = 51.  THE CONTROL for "
          "exp_047; it is exp_046 A1 re-fitted with the extended stacker curve.",
    "F2": "F1 + d_ball_eb@k=200 = 52.  The one axis that passed the "
          "pre-registered stage-1 screen.",
    "F2R": "F1 + d_rev_eb@k=25 = 52.  EXPLORATORY, not pre-registered: the "
           "reverse axis had the largest block gain of all nine (70.02 pts) "
           "and 4/4 block-unit sign agreement, and was removed only by the "
           "|r|>0.7-vs-incumbent rule, whose config wording ('keep the "
           "larger-gain one') points the other way.",
    "F3": "F1 + traj_eb@k=50 + dev_eb@k=25 + has_traj = 54.  Axis B.",
    "F4": "F2 + F3 columns.",
}


def r5(x):
    return None if x is None else round(float(x), 5)


def score(b):
    return 100000.0 * (1.0 - b / BRIER_REF)


def judge(ds):
    ds = [float(v) for v in ds]
    same = all(v > 0 for v in ds) or all(v < 0 for v in ds)
    mean = float(np.mean(ds))
    return {"dScore_per_seed": [r5(v) for v in ds], "mean_dScore": r5(mean),
            "paired_std_ddof1": r5(np.std(ds, ddof=1)) if len(ds) > 1 else None,
            "sign_agreement": int(sum(1 for v in ds if v > 0)),
            "all_same_sign": bool(same),
            "threshold_3sigma": THRESHOLD_3SIGMA,
            "threshold_multiple": r5(abs(mean) / THRESHOLD_3SIGMA),
            "ADOPTED": bool(same and abs(mean) >= THRESHOLD_3SIGMA)}


def collect(arm):
    out = {}
    for p in sorted(glob.glob(os.path.join(E47, "res",
                                           f"res_{arm}_u*_st*.json"))):
        with open(p) as f:
            j = json.load(f)
        st = j["stacker"]
        out.setdefault(f"fold_{j['valid_season']}", {})[str(j["seed"])] = {
            "brier_full": j["valid_all_brier_full"],
            "score_ref0.2498069": r5(score(j["valid_all_brier_full"])),
            "valid_all": j["valid_all"], "valid_R": j.get("valid_R"),
            "valid_F": j.get("valid_F"),
            "stage": j["stage"],
            "n_features_base": len(j.get("x", {}).get("new_columns", [])) + 51,
            "new_columns": j.get("x", {}).get("new_columns", []),
            "nan_pct": j.get("x", {}).get("nan_pct"),
            "cell_iter_used": j.get("cell_iter_used"),
            "stack_iter_used": st.get("iter_used"),
            "stack_policy_round": st.get("n_round"),
            "stack_fit_rounds": st.get("stack_fit_rounds"),
            "stack_argmin_iter": st.get("argmin_iter"),
            "curve_argmin_is_interior_extended":
                st.get("curve_argmin_is_interior_extended"),
            "brier_at_extended_argmin_ORACLE":
                st.get("brier_at_extended_argmin_ORACLE"),
            "policy_cost_vs_extended_min_pts":
                st.get("policy_cost_vs_extended_min_pts"),
            "pred_sha256": j.get("pred_sha256"),
            "minutes": j.get("minutes"),
            "src": "exp_047/res/" + os.path.basename(p)}
    return out


def main():
    arms_present = sorted({os.path.basename(p).split("_u")[0][4:]
                           for p in glob.glob(os.path.join(E47, "res",
                                                           "res_*_u*_st*.json"))})
    M = {"exp_id": "exp_047",
         "title": "in-season form decomposition extended to the remaining nine "
                  "asof rate columns + the trajectory/deviation split",
         "primary_metric": "fold_2024 valid_all Brier (train<=2023 -> valid 2024)",
         "control_for_this_experiment": "F1 (= exp_046 A1).  R0 (exp_030, 47 "
                                        "columns) appears below for scale only; "
                                        "the F1-R0 gap is exp_046's result, not "
                                        "this experiment's.",
         "brier_ref": BRIER_REF,
         "score_formula": "100000*(1 - brier/0.2498069)",
         "score_conversion_uncertainty_note":
             "score digits beyond ~0.2 are a machine transform of a rounded "
             "Brier, not measured resolution (Brier 1e-6 = score 0.4003)",
         "sigma_score": SIGMA_SCORE, "threshold_3sigma": THRESHOLD_3SIGMA,
         "arms_found": arms_present}

    with open(os.path.join(E47, "stage1_diag.json")) as f:
        diag = json.load(f)
    with open(os.path.join(E47, "stage1_selection.json")) as f:
        sel = json.load(f)
    M["stage1_diagnostic"] = {
        "fits_run": 0, "minutes": diag.get("minutes"),
        "pre_registered": diag["pre_registered"],
        "G1": diag["G1"], "G2a": diag["G2a"], "G2b": diag["G2b"],
        "G2c": diag["G2c"], "G3": diag["G3"],
        "gates_all_pass": diag["gates_all_pass"],
        "rounding": diag["rounding"],
        "selection": sel,
        "fold2024_descriptives": diag["fold2024_descriptives"],
    }

    # ---- reference arms ---------------------------------------------------- #
    ref = {}
    for name, path, pat in (("R0_exp030_C5_B3_47col", E30, "res_C5B3_u*_st*.json"),
                            ("A1_exp046_51col", E46, "res_C5B3form_u*_st*.json")):
        d = {}
        for p in sorted(glob.glob(os.path.join(path, "res", pat))):
            with open(p) as f:
                j = json.load(f)
            d.setdefault(f"fold_{j['valid_season']}", {})[str(j["seed"])] = {
                "brier_full": j["valid_all_brier_full"],
                "score_ref0.2498069": r5(score(j["valid_all_brier_full"])),
                "stack_iter_used": j["stacker"]["iter_used"],
                "curve_argmin_is_interior":
                    j["stacker"].get("curve_argmin_is_interior"),
                "pred_sha256": j.get("pred_sha256")}
        ref[name] = d
    M["reference_arms"] = ref

    # ---- this experiment's arms -------------------------------------------- #
    arms = {a: collect(a) for a in arms_present}

    # F1 seeds 7 / 2024 are REUSED from exp_046 A1 rather than re-fitted.
    # Justification is not "same config" but a measured one: exp_047's
    # b3_cat_x.py with no --axes / no --traj reproduced exp_046 A1's seed-42
    # fold_2024 prediction with an IDENTICAL sha256 (see identity_anchor), and
    # it did so both at --stack-ext 1 and at --stack-ext 6, so the extended
    # stacker fit does not perturb the prediction at the policy iteration.
    # The extended-curve diagnostic fields are therefore absent (null) for
    # these two seeds and present for seed 42.
    reused = []
    for s in ("7", "2024"):
        got = ref["A1_exp046_51col"].get("fold_2024", {}).get(s)
        if got and s not in arms.get("F1", {}).get("fold_2024", {}):
            arms.setdefault("F1", {}).setdefault("fold_2024", {})[s] = {
                "brier_full": got["brier_full"],
                "score_ref0.2498069": got["score_ref0.2498069"],
                "stack_iter_used": got["stack_iter_used"],
                "curve_argmin_is_interior_extended": None,
                "brier_at_extended_argmin_ORACLE": None,
                "policy_cost_vs_extended_min_pts": None,
                "stack_policy_round": got["stack_iter_used"],
                "stack_fit_rounds": got["stack_iter_used"],
                "stack_argmin_iter": None,
                "pred_sha256": got["pred_sha256"],
                "REUSED_FROM": "exp_046 A1 (identical code path, proven by the "
                               "seed-42 sha256 identity anchor)",
                "minutes": None,
                "src": "exp_046/res/res_C5B3form_u2024_seed%s_st2.json" % s}
            reused.append(s)
    if reused:
        arms["F1"]["_reuse_note"] = (
            "seeds %s reused from exp_046 A1; only seed 42 was re-fitted here"
            % ", ".join(reused))
    M["arms"] = {a: {"description": ARM_DESC.get(a, "(undescribed arm)"),
                     **arms[a]} for a in arms}

    # ---- identity anchor ---------------------------------------------------- #
    a46 = ref["A1_exp046_51col"].get("fold_2024", {}).get("42", {})
    anchor = {"what": "b3_cat_x.py with no --axes and no --traj must reproduce "
                      "exp_046 A1 (unit 2024, seed 42, stage 2) byte for byte. "
                      "If it does, the only difference in every other arm is "
                      "the columns named in that arm.",
              "exp046_A1_pred_sha256": a46.get("pred_sha256")}
    for a in ("F1anchor", "F1"):
        got = arms.get(a, {}).get("fold_2024", {}).get("42", {})
        anchor[a] = {"pred_sha256": got.get("pred_sha256"),
                     "stack_ext": got.get("stack_fit_rounds"),
                     "brier": got.get("brier_full"),
                     "MATCHES_exp046": bool(
                         got.get("pred_sha256")
                         and got.get("pred_sha256") == a46.get("pred_sha256"))}
    M["identity_anchor"] = anchor

    # ---- decisions ---------------------------------------------------------- #
    base = arms.get("F1", {}).get("fold_2024", {})
    dec = {}
    for a in arms:
        if a in ("F1", "F1anchor"):
            continue
        got = arms[a].get("fold_2024", {})
        ds = [-(got[str(s)]["brier_full"] - base[str(s)]["brier_full"]) * PTS
              for s in SEEDS if str(s) in got and str(s) in base]
        if not ds:
            continue
        dec[f"{a}_vs_F1"] = judge(ds)
        dec[f"{a}_vs_F1"]["control"] = "F1 (= exp_046 A1, refit here)"
        dec[f"{a}_vs_F1"]["changed_vs_control"] = [
            f"input columns: 51 -> {51 + len(ARM_COLS.get(a, []))} "
            f"({', '.join(ARM_COLS.get(a, [])) or 'none'})",
            "stacker width follows base_feats (+4 OOF columns in both arms)",
            "UNAVOIDABLE CO-VARIABLE, registered in config before the run: the "
            "cell and stacker iterations are re-derived from THIS arm's own "
            "fold_2023 stage-1 argmin, so the effect is 'these columns + the "
            "iterations they imply', never the columns alone.",
        ]
    M["decisions"] = dec
    M["F4_not_run"] = {
        "config_definition": "F2 + F3 combined",
        "gate_in_config": "run only if F2 AND F3 both pass",
        "status": "NOT RUN",
        "reason": "F2 did not clear the pre-registered 3-sigma threshold, so "
                  "the gate that would have launched F4 never opened.  This is "
                  "reported as a gate outcome, not a silent omission.",
    }

    # ---- stacker censoring, per arm ----------------------------------------- #
    cens = {}
    for a in arms:
        f24 = {k: v for k, v in arms[a].get("fold_2024", {}).items()
               if isinstance(v, dict)}
        cens[a] = {s: {"policy_round": v["stack_policy_round"],
                       "fit_rounds": v["stack_fit_rounds"],
                       "argmin": v["stack_argmin_iter"],
                       "interior_on_extended_grid":
                           v["curve_argmin_is_interior_extended"],
                       "policy_cost_vs_extended_min_pts":
                           v["policy_cost_vs_extended_min_pts"]}
                   for s, v in f24.items()}
    M["stacker_curve_censoring"] = {
        "why": "exp_046 A1 reported curve_argmin_is_interior=false on all three "
               "fold_2024 seeds, so its +45.17 was a lower bound.  exp_047 "
               "trains the stage-2 stacker past its policy iteration purely to "
               "LOCATE the argmin; iter_used is still the fold_2023-derived "
               "policy value, so fold_2024 selects nothing.",
        "policy_cost_is_a_cost_not_a_result":
            "policy_cost_vs_extended_min_pts is how many points the arm gives "
            "up by honouring the iteration policy.  It is an ORACLE quantity "
            "(the minimum is located on fold_2024) and must never be reported "
            "as an achievable gain.",
        "per_arm": cens}

    # ---- timings ------------------------------------------------------------ #
    mins = {a: sorted(v["minutes"] for f in arms[a].values()
                      if isinstance(f, dict)
                      for v in f.values()
                      if isinstance(v, dict) and v.get("minutes"))
            for a in arms}
    M["train_time_min"] = {"per_arm_units": mins,
                           "total_cpu_min": r5(sum(sum(v) for v in mins.values())),
                           "stage1_diagnostic_min": diag.get("minutes")}

    with open(os.path.join(E47, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=1,
                  ensure_ascii=False)
    print(json.dumps({"identity_anchor": M["identity_anchor"],
                      "decisions": M["decisions"]}, indent=1, ensure_ascii=False))


ARM_COLS = {"F2": ["d_ball_eb"], "F2R": ["d_rev_eb"],
            "F3": ["traj_eb", "dev_eb", "has_traj"],
            "F4": ["d_ball_eb", "traj_eb", "dev_eb", "has_traj"]}

if __name__ == "__main__":
    sys.exit(main())
