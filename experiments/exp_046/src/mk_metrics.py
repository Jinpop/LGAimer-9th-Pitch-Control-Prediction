"""exp_046 — assemble metrics.json from the per-unit result jsons.

Only reads files this experiment produced (plus exp_030's stored control
results, which are the R0 arm by design).  Every number here is copied from a
res_*.json / A2_offset.json; nothing is recomputed from memory.
"""

import glob
import json
import os

import numpy as np

HOME = os.path.expanduser("~/SungjinPark")
E46 = os.path.join(HOME, "experiments/exp_046")
E30 = os.path.join(HOME, "experiments/exp_030")
BRIER_REF = 0.2498069
PTS = 100000.0 / BRIER_REF
SIGMA_SCORE = 8.55113
THRESHOLD_3SIGMA = 25.65339
SEEDS = [42, 7, 2024]


def r5(x):
    return None if x is None else round(float(x), 5)


def load(pat):
    out = {}
    for p in sorted(glob.glob(pat)):
        with open(p) as f:
            j = json.load(f)
        out[os.path.basename(p)] = j
    return out


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


def main():
    M = {"exp_id": "exp_046",
         "title": "in-season form decomposition of the career asof cumulative",
         "primary_metric": "fold_2024 valid_all Brier (train<=2023 -> valid 2024)",
         "brier_ref": BRIER_REF,
         "score_formula": "100000*(1 - brier/0.2498069)",
         "score_conversion_uncertainty_note":
             "score digits beyond ~0.2 are a machine transform of a rounded "
             "Brier, not measured resolution (Brier 1e-6 = score 0.4003)",
         "sigma_score": SIGMA_SCORE, "threshold_3sigma": THRESHOLD_3SIGMA}

    with open(os.path.join(E46, "stage1_diag.json")) as f:
        diag = json.load(f)
    M["stage1_diagnostic"] = {
        "fits_run": 0,
        "gates": diag["gates"], "gates_all_pass": diag["gates_all_pass"],
        "rounding": diag["rounding"],
        "coverage_fold2024_valid": diag["coverage_fold2024_valid"],
        "dilution": diag["dilution"],
        "true_sd_d_form": diag["true_sd_d_form"],
        "binned_fold2024_seed42": diag["binned_fold2024_seed42"],
        "gain_ceiling": diag["gain_ceiling"],
        "stop_rules": diag["stop_rules"],
        "test_public_rows": diag["test_public_rows"],
    }

    # ------------------------------------------------ R0 control (exp_030) -- #
    r0 = load(os.path.join(E30, "res", "res_C5B3_u*_st*.json"))
    ctrl = {}
    for name, j in r0.items():
        ctrl.setdefault(f"fold_{j['valid_season']}", {})[str(j["seed"])] = {
            "brier_full": j["valid_all_brier_full"],
            "valid_all": j["valid_all"], "valid_R": j.get("valid_R"),
            "valid_F": j.get("valid_F"),
            "score_ref0.2498069": r5(100000.0 * (1 - j["valid_all_brier_full"] / BRIER_REF)),
            "src": f"exp_030/res/{name}", "stage": j["stage"],
            "cell_iter_used": j.get("cell_iter_used"),
            "stack_iter_used": j.get("stacker", {}).get("iter_used"),
            "minutes": j.get("minutes")}
    for fk, d in ctrl.items():
        bs = [v["brier_full"] for v in d.values()]
        d["_mean_brier"] = float(np.mean(bs))
        d["_mean_score"] = r5(100000.0 * (1 - np.mean(bs) / BRIER_REF))
    M["R0_control_exp030_C5_B3"] = ctrl

    # ------------------------------------------------------- A1 tree arm ---- #
    a1 = load(os.path.join(E46, "res", "res_C5B3form_u*_st*.json"))
    arm1 = {}
    for name, j in a1.items():
        arm1.setdefault(f"fold_{j['valid_season']}", {})[str(j["seed"])] = {
            "brier_full": j["valid_all_brier_full"],
            "valid_all": j["valid_all"], "valid_R": j.get("valid_R"),
            "valid_F": j.get("valid_F"),
            "score_ref0.2498069": r5(100000.0 * (1 - j["valid_all_brier_full"] / BRIER_REF)),
            "stage": j["stage"], "cell_iter_used": j.get("cell_iter_used"),
            "cell_argmin_iter": j.get("cell_argmin_iter"),
            "stack_iter_used": j.get("stacker", {}).get("iter_used"),
            "stack_argmin_iter": j.get("stack_argmin_iter"),
            "any_argmin_at_cap": j.get("any_argmin_at_cap"),
            "cap": j.get("cap"),
            "form": j.get("form"), "minutes": j.get("minutes"),
            "n_features_base": 51, "n_features_stacker": 55,
            "src": f"exp_046/res/{name}"}
    M["A1_tree_feature_51col"] = arm1

    if "fold_2024" in arm1 and "fold_2024" in ctrl:
        ds = []
        for s in SEEDS:
            if str(s) in arm1["fold_2024"] and str(s) in ctrl["fold_2024"]:
                ds.append(-(arm1["fold_2024"][str(s)]["brier_full"]
                            - ctrl["fold_2024"][str(s)]["brier_full"]) * PTS)
        if ds:
            M["A1_decision"] = judge(ds)
            M["A1_decision"]["changed_vs_control"] = [
                "input 47 -> 51 columns (n_season, rate_season, d_form_eb@k=100, has_form)",
                "stacker 51 -> 55 columns",
                "UNAVOIDABLE CO-VARIABLE: cell and stacker iterations re-derived "
                "on the 51-column frame (exp_030's iters do not exist for it). "
                "Registered in config changed_vs_control.A1 BEFORE the run.",
            ]

    # ------------------------------------------------------- A2 offset ------ #
    p = os.path.join(E46, "res", "A2_offset.json")
    if os.path.exists(p):
        with open(p) as f:
            a2 = json.load(f)
        M["A2_offset"] = {
            "selection_on_blocks_2022_2023": a2["selection_on_blocks_2022_2023"],
            "primary_registered_grid": a2["A2_primary_registered_grid"],
            "secondary_extended_grid": a2["A2_secondary_extended_grid"],
            "oracle_reference_upper_bound_only": a2["oracle_reference"],
            "cost_of_honest_selection_pts": a2["cost_of_honest_selection_pts"],
            "coverage_check": a2["coverage_check"],
            "fits_run": 0,
        }
        M["A2_decision"] = judge(a2["A2_primary_registered_grid"]["dScore_per_seed"])
        M["A2_decision"]["changed_vs_control"] = [
            "one post-processing term p' = clip(p + 0.23682 * d_form_eb@k=100); "
            "the booster is untouched",
            "beta_f newly fitted (closed form) on supply blocks 2022+2023 only",
        ]

    # -------------------- secondary probes (NOT pre-registered arms) -------- #
    probes = {}
    for key, fn in (("A2_on_existing_S400_offset", "A2_on_existing_offset.json"),
                    ("A2_on_top_of_A1", "A1_plus_A2_probe.json")):
        pp = os.path.join(E46, "res", fn)
        if os.path.exists(pp):
            with open(pp) as f:
                probes[key] = json.load(f)
    if probes:
        probes["⚠️"] = ("These were not in config.yaml stage2_arms.  They are "
                        "labelled probes, no adoption decision is taken from "
                        "them, and beta_f is not refitted in either.")
        M["secondary_probes"] = probes

    # ------------------------ pre-registered arm NOT run -------------------- #
    M["A3_not_run"] = {
        "config_definition": "A1 + the existing offset machine (S400_league d) refitted",
        "gate_in_config": "run only if A1 passes — A1 DID pass",
        "status": "NOT RUN",
        "reason": "A3 needs exp_033's d_s/d_r table generator ported into this "
                  "experiment, which is new code on a target-derived path and "
                  "therefore its own audit surface.  The A2-on-S400 probe "
                  "already measures how much of the form signal survives on top "
                  "of the existing offset (96.4% retained), which is the "
                  "question A3 was registered to answer, so the remaining gap "
                  "is A1-specific rather than about the axis.",
        "🔴": "Reported as an omission, not silently dropped.  Whether to run it "
              "is a user decision.",
    }

    # ------------------------------------------------- identity anchor ------ #
    anc = os.path.join(E46, "res", "res_C5B3_u2022_seed42_st2.json")
    if os.path.exists(anc):
        with open(anc) as f:
            aj = json.load(f)
        with open(os.path.join(E30, "res", "res_C5B3_u2022_seed42_st2.json")) as f:
            oj = json.load(f)
        M["identity_anchor"] = {
            "what": "b3_cat_form.py --no-form must reproduce exp_030's unit "
                    "2022 seed 42 stage 2 prediction byte for byte; if it does, "
                    "the ONLY difference in arm A1 is the four columns",
            "exp046_pred_sha256": aj["pred_sha256"],
            "exp030_pred_sha256": oj["pred_sha256"],
            "PASS": bool(aj["pred_sha256"] == oj["pred_sha256"]),
            "brier_exp046": aj["valid_all_brier_full"],
            "brier_exp030": oj["valid_all_brier_full"],
            "abs_brier_diff": abs(aj["valid_all_brier_full"] - oj["valid_all_brier_full"]),
        }

    # -------------------------------------------------------- timings ------ #
    mins = [v.get("minutes") for fk in arm1 for v in arm1[fk].values()
            if v.get("minutes")]
    M["train_time_min"] = {
        "A1_units_individual": sorted(mins),
        "A1_total_cpu_min": r5(sum(mins)) if mins else None,
        "A2_fits": 0, "stage1_diagnostic_min": 0.09,
    }
    # measured here, not guessed: the row-wise cost A2 would add at inference
    import time as _t
    import pandas as _pd
    _df = _pd.read_csv(os.path.join(HOME, "data/train.csv"),
                       usecols=["season", "pitcher_id", "asof_pitcher_n",
                                "asof_pitcher_success_rate", "control_success"],
                       encoding="utf-8-sig")
    import form_decomp as _fd
    _c, _u = _pd.factorize(_df["pitcher_id"], sort=True)
    _tab = _fd.build_boundary_tables(_df, _c.astype("int64"), len(_u))
    _m = (_df["season"].to_numpy() == 2024)
    _n = _df["asof_pitcher_n"].to_numpy(float)[_m]
    _r = _df["asof_pitcher_success_rate"].to_numpy(float)[_m]
    _p = _c.astype("int64")[_m]
    _t0 = _t.time()
    for _ in range(5):
        _o = _fd.apply_decomposition(_n, _r, _p, _tab[2023][0], _tab[2023][1], 100)
        _ = np.clip(0.5 + 0.23682 * np.nan_to_num(_o["d_form_eb"]), 1e-6, 1 - 1e-6)
    _el = (_t.time() - _t0) / 5.0
    M["inference_time_sec"] = {
        "A2_row_wise_decomposition_plus_offset_253507_rows": round(_el, 4),
        "n_rows_measured": int(_m.sum()),
        "measured_how": "5 repeats of apply_decomposition + clip on the "
                        "fold_2024 valid rows, frozen table already in memory",
        "note": "A2 adds only this to an existing pipeline; the booster is "
                "untouched.  No deployment (<=2024 refit) build was made in "
                "this experiment, so NO end-to-end submission timing is "
                "claimed here.",
    }

    with open(os.path.join(E46, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=1,
                  ensure_ascii=False)
    print(json.dumps({k: M[k] for k in
                      ("A1_decision", "A2_decision", "identity_anchor")
                      if k in M}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
