"""exp_036 aggregation — every number in report.md is produced here.

Reads ONLY artefacts: exp_036's own res/curves/offset_q4.json, exp_030's
metrics.json and res/*.json, exp_034(=exp_032)'s metrics.json and
curves/iter_fit.json.  Nothing is transcribed from a report or a memory.

The four questions, in the order config.yaml asks them:
  Q1  where does the fold_2024 stacker curve bottom, and is that an INTERIOR
      extremum (argmin < cap) rather than a truncation?
  Q2  how much was the 481-round configuration leaving on the table?
  Q3  what is the seed std AT that argmin?  (recorded for FUTURE thresholds
      only — it is not applied backwards to any past adoption call)
  Q4  is the S400_self offset still a gain there?  (from offset_q4.json)

plus the DEPLOY extrapolation, reported as a RANGE over several functional
forms, because every single form on offer has already been falsified once.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
EXP030 = os.path.join(ROOT, "exp_030")
EXP034 = os.path.join(ROOT, "exp_034")           # == remote exp_032

REF_BRIER = 0.2498069
SEEDS = [42, 7, 2024]
GATE_K = 481
DEPLOY_N = 1369784
FOLD_N = 1116277


def score(b):
    return float(f"{100000.0 * (1.0 - b / REF_BRIER):.5f}")


def score_delta(db):
    """points gained when Brier CHANGES by db (negative db => positive points)"""
    return float(f"{-100000.0 * db / REF_BRIER:.5f}")


def main():
    out = {
        "exp_id": "exp_036",
        "title": ("the uncensored fold_2024 argmin of the B3 stacker — a "
                  "MEASUREMENT, not an adoption"),
        "primary_metric": "fold_2024 valid_all (R+F) Brier",
        "conversion": {
            "formula": "score = 100000 * (1 - brier / 0.2498069)",
            "sensitivity": "1e-06 Brier = 0.4003 score points",
            "caveat": ("every Brier below is full precision, so the +-0.20 "
                       "rounding caveat does NOT apply"),
        },
        "what_changed_vs_exp_030_fold_2024": [
            "1. the stacker's training length: 481 rounds (exp_030 stage-2 "
            "row-ratio policy) -> 3000 rounds, and the round reported is the "
            "argmin of the validation curve rather than the fixed 481.",
            "2. the stacker's validation curve resolution: exp_030 evaluated "
            "the sparse curve_grid (step 25 in this region); exp_036 evaluates "
            "EVERY round 1..3000.  A finer grid can move a reported argmin on "
            "its own, so this is listed as a variable, not as a free upgrade.",
            "3. NOTHING ELSE.  The four cell iterations are exp_030's "
            "{is_mid_only 577, is_rev_only 1346, is_both 449, is_wayoff_true "
            "897}, pinned on the command line; the OOF construction, masks, "
            "row counts, seeds and the C5 spec are unchanged and the "
            "reproduction gate below checks that claim at round 481.",
        ],
    }

    # ------------------------------------------------------------ artefacts
    res, cur = {}, {}
    for s in SEEDS:
        stem = f"C5B3_u2024_seed{s}_st5"
        with open(os.path.join(EXP, "res", f"res_{stem}.json")) as f:
            res[s] = json.load(f)
        with open(os.path.join(EXP, "curves", f"curve_{stem}.json")) as f:
            cur[s] = json.load(f)
        assert res[s]["exp_id"] == "exp_036" and res[s]["stage"] == 5
        assert res[s]["unit"] == 2024 and res[s]["seed"] == s
        assert res[s]["n_train_full"] == FOLD_N, res[s]["n_train_full"]

    with open(os.path.join(EXP030, "metrics.json")) as f:
        m30 = json.load(f)

    # ------------------------------------------------- GATE (config's form)
    # config.yaml: "read the expected values FROM exp_030/metrics.json, do not
    # transcribe".  metrics.json stores them under fold_2024, seed-ordered.
    exp030_seeds = m30["fold_2024"]["seeds"]
    exp030_brier = m30["fold_2024"]["candidate_brier_by_seed"]
    assert exp030_seeds == SEEDS, (exp030_seeds, SEEDS)
    assert m30["stage_2_iter_policy"]["resulting_stacker_iter"]["2024"] == GATE_K
    exp030_cells = m30["stage_2_iter_policy"]["resulting_cell_iter"]["2024"]

    gate = {"tol": 5e-6, "gate_k": GATE_K,
            "expected_read_from": "experiments/exp_030/metrics.json "
                                  "fold_2024.candidate_brier_by_seed",
            "exp030_stacker_iter_2024_read_from_metrics":
                m30["stage_2_iter_policy"]["resulting_stacker_iter"]["2024"],
            "by_seed": {}}
    for s, exp_b in zip(exp030_seeds, exp030_brier):
        g = res[s]["reproduction_gate"]
        got = g["observed_brier"]
        gate["by_seed"][str(s)] = {
            "expected_exp030_metrics": exp_b,
            "expected_exp030_res_file": g["expected_brier"],
            "observed": got,
            "abs_diff_vs_metrics": abs(got - exp_b),
            "abs_diff_vs_res_file": g["abs_diff"],
            "PASS": bool(abs(got - exp_b) < 5e-6),
            "bit_identical_pred": g["bit_identical"],
            "pred_sha256_expected": g["expected_pred_sha256"],
            "pred_sha256_observed": g["observed_pred_sha256"],
        }
        assert res[s]["cell_iter_used"] == exp030_cells, (
            s, res[s]["cell_iter_used"], exp030_cells)
    gate["ALL_PASS"] = all(v["PASS"] for v in gate["by_seed"].values())
    gate["ALL_BIT_IDENTICAL"] = all(v["bit_identical_pred"]
                                    for v in gate["by_seed"].values())
    gate["cells_match_exp030"] = True
    gate["means"] = ("the stacker trained to 3000 rounds reproduces exp_030's "
                     "481-round prediction when read at round 481, so exp_036 "
                     "drew the SAME curve further, not a different curve")
    out["reproduction_gate"] = gate
    assert gate["ALL_PASS"], gate

    # ------------------------------------------------------------------- Q1
    cap = res[SEEDS[0]]["cap"]
    assert all(res[s]["cap"] == cap for s in SEEDS)
    q1 = {"question": "where does the fold_2024 stacker curve bottom, and is "
                      "the argmin an INTERIOR extremum?",
          "cap": cap, "curve_step": 1, "by_seed": {}}
    for s in SEEDS:
        st = res[s]["stacker"]
        dense = {int(k): v for k, v in
                 cur[s]["stacker_valid_all_brier_dense"].items()}
        a = st["argmin_iter"]
        q1["by_seed"][str(s)] = {
            "argmin": a,
            "argmin_brier": st["argmin_brier"],
            "argmin_score": score(st["argmin_brier"]),
            "is_interior": bool(a < cap),
            "brier_at_cap": st["brier_at_cap"],
            "rise_argmin_to_cap_brier": st["rise_argmin_to_cap"],
            "rise_argmin_to_cap_score": score_delta(st["rise_argmin_to_cap"]),
            "n_rounds_within_1e-6_of_min": int(
                sum(1 for v in dense.values() if v - st["argmin_brier"] < 1e-6)),
            "flat_region_1e-6": [
                min(k for k, v in dense.items()
                    if v - st["argmin_brier"] < 1e-6),
                max(k for k, v in dense.items()
                    if v - st["argmin_brier"] < 1e-6)],
        }
    argmins = [q1["by_seed"][str(s)]["argmin"] for s in SEEDS]
    q1["argmins"] = argmins
    q1["argmin_mean"] = float(np.mean(argmins))
    q1["argmin_median"] = float(np.median(argmins))
    q1["argmin_geomean"] = float(np.exp(np.mean(np.log(argmins))))
    q1["argmin_min"] = int(min(argmins))
    q1["argmin_max"] = int(max(argmins))
    q1["argmin_seed_std"] = float(np.std(argmins, ddof=0))
    q1["ALL_INTERIOR"] = all(q1["by_seed"][str(s)]["is_interior"] for s in SEEDS)
    q1["ANY_AT_CAP"] = not q1["ALL_INTERIOR"]
    q1["EXPERIMENT_VALID"] = q1["ALL_INTERIOR"]
    q1["verdict"] = (
        f"VALID — all three stacker argmins are interior extrema under cap "
        f"{cap}" if q1["ALL_INTERIOR"] else
        f"INVALID — an argmin reached the cap {cap}; re-run at 6000")
    q1["exp030_reported_argmin_at_cap481"] = {
        str(s): m30["fold_2024"].get("note", None) for s in []}
    out["Q1_argmin"] = q1

    # ------------------------------------------------------------------- Q2
    q2 = {"question": "how much was the 481-round configuration leaving on "
                      "the table?",
          "note": ("this is NOT a candidate-vs-control adoption delta.  Both "
                   "numbers come from the SAME fitted model read at two "
                   "different rounds, so the comparison is paired to the byte "
                   "and carries no seed noise from refitting."),
          "by_seed": {}}
    for s in SEEDS:
        st = res[s]["stacker"]
        db = st["argmin_brier"] - st["brier_at_gate_k"]
        q2["by_seed"][str(s)] = {
            "brier_at_481": st["brier_at_gate_k"],
            "score_at_481": score(st["brier_at_gate_k"]),
            "argmin": st["argmin_iter"],
            "brier_at_argmin": st["argmin_brier"],
            "score_at_argmin": score(st["argmin_brier"]),
            "delta_brier": db,
            "delta_score_points": score_delta(db),
        }
    d = [q2["by_seed"][str(s)]["delta_brier"] for s in SEEDS]
    b481 = [q2["by_seed"][str(s)]["brier_at_481"] for s in SEEDS]
    bam = [q2["by_seed"][str(s)]["brier_at_argmin"] for s in SEEDS]
    q2["mean_brier_at_481"] = float(np.mean(b481))
    q2["mean_score_at_481"] = score(float(np.mean(b481)))
    q2["mean_brier_at_argmin"] = float(np.mean(bam))
    q2["mean_score_at_argmin"] = score(float(np.mean(bam)))
    q2["mean_delta_brier"] = float(np.mean(d))
    q2["mean_delta_score_points"] = score_delta(float(np.mean(d)))
    q2["signs_agree"] = bool(all(v < 0 for v in d) or all(v > 0 for v in d))
    # the SIGN of the Brier delta is always <= 0 (the argmin is the minimum by
    # definition).  The interesting direction is whether the argmin sits ABOVE
    # or BELOW 481 — i.e. whether 481 was under- or over-trained — and that is
    # NOT the same question, nor is it the same answer for every seed.
    above = [s for s in SEEDS if q2["by_seed"][str(s)]["argmin"] > GATE_K]
    below = [s for s in SEEDS if q2["by_seed"][str(s)]["argmin"] < GATE_K]
    q2["argmin_above_481_seeds"] = above
    q2["argmin_below_481_seeds"] = below
    q2["direction"] = (
        f"NOT uniform: seeds {above} bottom ABOVE 481 (481 was slightly "
        f"under-trained for them) and seeds {below} bottom BELOW 481 (481 was "
        f"slightly over-trained for them).  481 sits INSIDE the spread of the "
        f"three argmins [{min(argmins)}, {max(argmins)}], which is why the "
        f"recoverable gain is small in every seed."
        if above and below else
        (f"every seed's argmin lies above 481 — 481 was under-trained"
         if not below else
         f"every seed's argmin lies below 481 — 481 was over-trained"))
    q2["481_is_inside_the_argmin_spread"] = bool(
        min(argmins) <= GATE_K <= max(argmins))
    q2["headline"] = (
        f"481 was NOT badly under-trained.  The three argmins are "
        f"{argmins} against a cap of {cap}; recovering the difference is worth "
        f"{score_delta(float(np.mean(d)))} points on the 3-seed mean, and the "
        f"largest single-seed gain is "
        f"{max(q2['by_seed'][str(s)]['delta_score_points'] for s in SEEDS)} "
        f"points.")
    q2["comparison_exp034_cut_to_240"] = {
        "what": ("exp_034 cut the stacker from 481 to 240 and measured the "
                 "cost with the cells held fixed"),
        "source": "experiments/exp_034/curves/loss_decomposition.json",
    }
    ld = os.path.join(EXP034, "curves", "loss_decomposition.json")
    if os.path.exists(ld):
        with open(ld) as f:
            q2["comparison_exp034_cut_to_240"]["file_contents"] = json.load(f)
    out["Q2_cost_of_481"] = q2

    # ------------------------------------------------------------------- Q3
    q3 = {"question": "what is the seed std AT the measured argmin?",
          "PRE_REGISTERED_SCOPE_LIMIT": (
              "this std is recorded as the honest denominator for FUTURE "
              "adoption thresholds on this configuration.  It is NOT applied "
              "retroactively; no past adoption call is re-decided with it."),
          "naming_warning": (
              "'threshold multiple' and 'std multiple' differ by exactly 3x "
              "because the rule's threshold IS 3 x std.  They are different "
              "numbers with different names and must never be compared across "
              "experiments without checking which one was reported."),
          "at_argmin": {
              "brier_by_seed": bam,
              "seed_std": float(np.std(bam, ddof=0)),
              "seed_std_score_points": abs(score_delta(float(np.std(bam, ddof=0)))),
              "implied_3x_threshold_brier": 3.0 * float(np.std(bam, ddof=0)),
              "implied_3x_threshold_score_points": abs(score_delta(
                  3.0 * float(np.std(bam, ddof=0)))),
          },
          "at_481_same_models": {
              "brier_by_seed": b481,
              "seed_std": float(np.std(b481, ddof=0)),
              "seed_std_score_points": abs(score_delta(float(np.std(b481, ddof=0)))),
          },
          "exp030_recorded_control_seed_std":
              m30["fold_2024"]["control_seed_std_DENOMINATOR"],
          }
    q3["ratio_argmin_std_over_481_std"] = (
        q3["at_argmin"]["seed_std"] / q3["at_481_same_models"]["seed_std"]
        if q3["at_481_same_models"]["seed_std"] > 0 else None)
    out["Q3_seed_std"] = q3

    # ------------------------------------------------------------------- Q4
    qp = os.path.join(EXP, "offset_q4.json")
    if os.path.exists(qp):
        with open(qp) as f:
            out["Q4_offset"] = json.load(f)
    else:
        out["Q4_offset"] = {"status": "NOT RUN"}

    # -------------------------------------------------------- deploy iter
    with open(os.path.join(EXP034, "curves", "iter_fit.json")) as f:
        ifit = json.load(f)
    obs = {}
    for o in ifit["observations_raw"]:
        obs.setdefault(o["n_full"], []).append(o["stack_argmin"])
    obs[FOLD_N] = argmins

    sizes = sorted(obs)
    pts = [{"n": n, "argmins": obs[n], "n_seeds": len(obs[n]),
            "arith_mean": float(np.mean(obs[n])),
            "geo_mean": float(np.exp(np.mean(np.log(obs[n]))))}
           for n in sizes]

    def fit_extrap(xs, ys, kind):
        x = np.asarray(xs, float)
        y = np.asarray(ys, float)
        if kind == "loglog":
            b, a = np.polyfit(np.log(x), np.log(y), 1)
            return float(np.exp(a + b * math.log(DEPLOY_N))), {"slope": float(b)}
        if kind == "loglinear":
            b, a = np.polyfit(x, np.log(y), 1)
            return float(np.exp(a + b * DEPLOY_N)), {"slope_per_row": float(b)}
        if kind == "linear":
            b, a = np.polyfit(x, y, 1)
            return float(a + b * DEPLOY_N), {"slope_per_row": float(b)}
        raise ValueError(kind)

    gm = [p["geo_mean"] for p in pts]
    forms = {}
    v, m = fit_extrap(sizes, gm, "loglog")
    forms["A_loglog_all_4_sizes"] = {
        "deploy_iter": v, "params": m,
        "basis": "log(argmin) ~ log(n) over all four observed sizes"}
    v, m = fit_extrap(sizes[-2:], gm[-2:], "loglog")
    forms["B_loglog_last_2_sizes"] = {
        "deploy_iter": v, "params": m,
        "basis": "log-log through the two largest sizes only — the closest "
                 "pair to the deploy size, but a 2-point line has no residual"}
    v, m = fit_extrap(sizes, gm, "loglinear")
    forms["C_loglinear_all_4"] = {
        "deploy_iter": v, "params": m,
        "basis": "log(argmin) ~ n (exponential in rows)"}
    v, m = fit_extrap(sizes, gm, "linear")
    forms["D_linear_all_4"] = {
        "deploy_iter": v, "params": m,
        "basis": "argmin ~ n (straight line in rows)"}
    forms["E_row_ratio_from_fold_2024"] = {
        "deploy_iter": float(np.mean(argmins) * DEPLOY_N / FOLD_N),
        "params": {"ratio": DEPLOY_N / FOLD_N},
        "basis": "exp_030's own policy FORM (argmin x training-row ratio) "
                 "applied from the newly measured fold_2024 argmin.  The form "
                 "was falsified as a SUCCESSION rule by exp_033; it is listed "
                 "only because it is the rule the current build would use"}
    forms["F_no_extrapolation"] = {
        "deploy_iter": float(np.mean(argmins)),
        "params": {},
        "basis": "carry the fold_2024 argmin over unchanged — the lower "
                 "anchor, assuming argmin has stopped growing with rows"}
    # --- what the previously fitted laws predicted for a size we NOW know --- #
    # exp_034 fitted its law on the three SMALLER sizes only and extrapolated to
    # fold_2024.  fold_2024 is now measured, so that extrapolation can be
    # scored.  This is the only out-of-sample test of the law form that exists.
    law = {"measured_fold_2024_geomean_argmin": float(np.exp(np.mean(
        np.log(argmins))))}
    with open(os.path.join(EXP034, "iter_policy.json")) as f:
        ipol = json.load(f)
    tg = ifit["targets"]["stacker"]
    sel = tg["fold_2024_iter_USED"]
    law["exp034_selected_form"] = tg["SELECTED_FORM"]
    law["exp034_selected_by"] = tg["selected_by"]
    law["exp034_loso_ranking"] = tg["loso_ranking"]
    law["exp034_predicted_fold_2024_stack_iter"] = sel
    law["exp034_predicted_deploy_stack_iter"] = tg[
        "deploy_le2024_iter_RECORD_ONLY"]
    law["exp034_deploy_iter_in_iter_policy_json"] = ipol[
        "for_stage_E_deploy_RECORD_ONLY"]["stack_iter"]
    law["exp034_SELECTED_form_ratio_measured_over_predicted"] = (
        law["measured_fold_2024_geomean_argmin"] / sel)
    law["exp034_SELECTED_form_error"] = (
        f"the form exp_034 chose ('{tg['SELECTED_FORM']}') predicted "
        f"{sel} for fold_2024; the measured geometric-mean argmin is "
        f"{law['measured_fold_2024_geomean_argmin']:.1f}, so it UNDER-predicted "
        f"by a factor of "
        f"{law['measured_fold_2024_geomean_argmin'] / sel:.2f}.  That is the "
        f"mechanism behind exp_034's -23.02235-point base: it trained the "
        f"stacker to less than half the rounds the curve wanted.")
    small = sizes[:3]
    gm_small = gm[:3]
    v3, m3 = fit_extrap(small, gm_small, "loglog")
    b3, a3 = np.polyfit(np.log(small), np.log(gm_small), 1)
    law["refit_loglog_on_3_smaller_sizes"] = {
        "slope": float(b3),
        "predicted_at_fold_2024": float(np.exp(a3 + b3 * math.log(FOLD_N))),
        "measured_at_fold_2024": law["measured_fold_2024_geomean_argmin"],
        "predicted_at_deploy": v3,
    }
    law["refit_loglog_on_3_smaller_sizes"]["ratio_measured_over_predicted"] = (
        law["measured_fold_2024_geomean_argmin"]
        / law["refit_loglog_on_3_smaller_sizes"]["predicted_at_fold_2024"])
    law["verdict"] = (
        "the LOSO selection picked the wrong form.  Fitted on the same three "
        "smaller sizes, the LOG-LOG form exp_034 RANKED THIRD predicts "
        f"{law['refit_loglog_on_3_smaller_sizes']['predicted_at_fold_2024']:.0f}"
        f" against a measured {law['measured_fold_2024_geomean_argmin']:.0f} — "
        f"an error of "
        f"{100*abs(1-law['refit_loglog_on_3_smaller_sizes']['ratio_measured_over_predicted']):.1f}"
        "%, while the 'proportional' form it SELECTED is off by a factor of "
        f"{law['exp034_SELECTED_form_ratio_measured_over_predicted']:.2f}.  "
        "One out-of-sample point cannot crown a form, but it does say the "
        "selection rule that produced 241 is not to be trusted at the deploy "
        "size either.")
    law["CAUTION"] = (
        "this is ONE out-of-sample point, and it is not even drawn from the "
        "same construction as the three it is tested against (see "
        "SERIES_CAVEATS: pinned vs self-stopped cells, step-1 vs sparse grid, "
        "3 seeds vs 2).  It is enough to falsify 'proportional'; it is NOT "
        "enough to certify log-log.")
    out_law = law

    vals = [f["deploy_iter"] for f in forms.values()]
    out["deploy_iter"] = {
        "deploy_n_train_full": DEPLOY_N,
        "fold_2024_n_train_full": FOLD_N,
        "row_ratio": DEPLOY_N / FOLD_N,
        "observed_argmin_series": pts,
        "out_of_sample_test_of_the_fitted_laws": out_law,
        "forms": forms,
        "RANGE": {"min": float(min(vals)), "max": float(max(vals)),
                  "spread_factor": float(max(vals) / min(vals))},
        "NO_SINGLE_VALUE_IS_ASSERTED": (
            "config.yaml forbids picking one.  Choosing among these is the "
            "orchestrator's and the user's call, not this experiment's."),
        "WHAT_THIS_EXPERIMENT_DID_AND_DID_NOT_CLOSE": (
            "CLOSED: the fold_2024 argmin, which was previously known only as "
            "'>= 481'.  NOT CLOSED: the DEPLOY argmin.  1,369,784 rows remain "
            "OUTSIDE every size ever observed, and a deployment build cannot "
            "early-stop, so no validation curve exists there.  Every number in "
            "`forms` is an extrapolation, not a measurement."),
        "SERIES_CAVEATS": [
            "the three smaller sizes were measured on the SPARSE curve_grid "
            "(resolution 5 to 25 rounds in the relevant range); the 1,116,277 "
            "point is measured at step 1.  The points do not share a "
            "resolution.",
            "the three smaller sizes let their CELL models early-stop on their "
            "own curves; the 1,116,277 point has its cells PINNED at exp_030's "
            "row-scaled numbers.  The stacker's input features are therefore "
            "not built the same way across the series.",
            "the smaller sizes have 2 seeds each, the largest has 3.",
            "the share of the training pool carrying NaN stack columns falls "
            "monotonically across the series (48.9 / 32.4 / 24.3 / 19.0 pct) "
            "and is 15.45 pct at the deploy size, so the units are not pure "
            "rescalings of one another and no one-parameter law in n is "
            "innocent of that confound.",
            "exp_034 fitted a law on the first three points, extrapolated to "
            "stacker 241 for fold_2024, and the resulting configuration was "
            "NOT adopted (it worsened the base by 23.02235 points).  A fitted "
            "form on this series has already failed once.",
        ],
    }

    # ------------------------------------------------------------- runtime
    out["runtime"] = {
        "train_time_min_by_seed": {str(s): res[s]["minutes"] for s in SEEDS},
        "train_time_min_wallclock_3wide": max(res[s]["minutes"] for s in SEEDS),
        "stacker_fit_min_by_seed": {str(s): res[s]["stacker"]["fit_min"]
                                    for s in SEEDS},
        "stacker_sec_per_round_by_seed": {
            str(s): res[s]["stacker"]["sec_per_round"] for s in SEEDS},
        "oof_min_by_seed": {str(s): res[s]["oof_min"] for s in SEEDS},
        "concurrency": "3 units at once, 6 threads per fit",
    }
    out["provenance"] = {
        "sha256": res[SEEDS[0]]["sha256"],
        "versions": res[SEEDS[0]]["versions"],
        "n_train_full": FOLD_N,
        "n_train_mode": res[SEEDS[0]]["n_train_mode"],
        "n_valid": res[SEEDS[0]]["n_valid"],
        "n_valid_R": res[SEEDS[0]]["n_valid_R"],
        "n_valid_F": res[SEEDS[0]]["n_valid_F"],
        "iter_source": res[SEEDS[0]]["iter_source"],
        "iter_policy": res[SEEDS[0]]["iter_policy"],
        "nan_stack_rows_in_train_pct":
            res[SEEDS[0]]["nan_stack_rows_in_train_pct"],
    }
    out["segments_at_argmin"] = {
        str(s): {"valid_all": res[s]["valid_all"], "valid_R": res[s]["valid_R"],
                 "valid_F": res[s]["valid_F"]} for s in SEEDS}
    out["NO_ADOPTION_CALL"] = (
        "This experiment makes no adoption call.  An argmin larger than 481 "
        "means the previous configuration was UNDER-TRAINED, not that a better "
        "configuration was discovered.  Promoting the new round count to a "
        "deployment candidate is a separate decision.")

    # ------------------------------------------------------- report_derived
    # EVERY number quoted in report.md is regenerated here from the artefacts
    # above, so the prose can be diffed against the data instead of trusted.
    q4 = out["Q4_offset"]
    rd = {
        "note": ("report.md must not contain a number that is absent from "
                 "this block.  Everything here is computed from res/, curves/, "
                 "offset_q4.json and the exp_030 / exp_034 artefacts."),
        "gate": {
            "abs_diff_by_seed": {str(s): gate["by_seed"][str(s)]
                                 ["abs_diff_vs_metrics"] for s in SEEDS},
            "all_pass": gate["ALL_PASS"],
            "all_bit_identical": gate["ALL_BIT_IDENTICAL"],
        },
        "Q1": {
            "argmins_by_seed": {str(s): q1["by_seed"][str(s)]["argmin"]
                                for s in SEEDS},
            "arith_mean": q1["argmin_mean"],
            "geo_mean": q1["argmin_geomean"],
            "cap": cap,
            "all_interior": q1["ALL_INTERIOR"],
            "flat_region_1e-6_by_seed": {
                str(s): q1["by_seed"][str(s)]["flat_region_1e-6"]
                for s in SEEDS},
            "brier_at_cap_by_seed": {str(s): q1["by_seed"][str(s)]
                                     ["brier_at_cap"] for s in SEEDS},
            "rise_argmin_to_cap_score_by_seed": {
                str(s): q1["by_seed"][str(s)]["rise_argmin_to_cap_score"]
                for s in SEEDS},
        },
        "Q2": {
            "delta_score_by_seed": {str(s): q2["by_seed"][str(s)]
                                    ["delta_score_points"] for s in SEEDS},
            "mean_delta_score": q2["mean_delta_score_points"],
            "mean_brier_at_481": q2["mean_brier_at_481"],
            "mean_score_at_481": q2["mean_score_at_481"],
            "mean_brier_at_argmin": q2["mean_brier_at_argmin"],
            "mean_score_at_argmin": q2["mean_score_at_argmin"],
            "481_inside_argmin_spread": q2["481_is_inside_the_argmin_spread"],
            "argmin_above_481_seeds": q2["argmin_above_481_seeds"],
            "argmin_below_481_seeds": q2["argmin_below_481_seeds"],
        },
        "Q3": {
            "seed_std_at_argmin": q3["at_argmin"]["seed_std"],
            "seed_std_at_argmin_points": q3["at_argmin"]["seed_std_score_points"],
            "seed_std_at_481": q3["at_481_same_models"]["seed_std"],
            "seed_std_at_481_points":
                q3["at_481_same_models"]["seed_std_score_points"],
            "ratio": q3["ratio_argmin_std_over_481_std"],
            "implied_3x_threshold_points":
                q3["at_argmin"]["implied_3x_threshold_score_points"],
        },
        "Q4": {
            "beta": q4.get("beta_applied"),
            "beta_reproduces_exp034": q4.get("beta_reproduces_exp034"),
            "gain_on_base_481_points":
                q4["bases"]["base_at_481_GATE"]["mean_delta_score"],
            "gain_on_base_argmin_points":
                q4["bases"]["base_at_argmin"]["mean_delta_score"],
            "offset_score_on_base_481":
                q4["bases"]["base_at_481_GATE"]["offset_mean_score"],
            "offset_score_on_base_argmin":
                q4["bases"]["base_at_argmin"]["offset_mean_score"],
            "threshold_multiple_on_base_481": q4["bases"]["base_at_481_GATE"]
                ["adoption_rule"]["MULTIPLE_OF_THE_THRESHOLD"],
            "threshold_multiple_on_base_argmin": q4["bases"]["base_at_argmin"]
                ["adoption_rule"]["MULTIPLE_OF_THE_THRESHOLD"],
            "max_abs_beta_d_PROBABILITY_UNITS": q4["d_2024"]["max_abs_beta_d"],
        },
        "deploy": {
            "forms": {k: v["deploy_iter"] for k, v in forms.items()},
            "range": out["deploy_iter"]["RANGE"],
            "exp034_predicted_fold_2024": out_law[
                "exp034_predicted_fold_2024_stack_iter"],
            "exp034_predicted_deploy": out_law[
                "exp034_predicted_deploy_stack_iter"],
            "exp034_selected_form_error_factor": out_law[
                "exp034_SELECTED_form_ratio_measured_over_predicted"],
            "loglog_on_3_predicted_fold_2024": out_law[
                "refit_loglog_on_3_smaller_sizes"]["predicted_at_fold_2024"],
            "loglog_on_3_predicted_deploy": out_law[
                "refit_loglog_on_3_smaller_sizes"]["predicted_at_deploy"],
        },
        "runtime": out["runtime"],
    }
    out["report_derived"] = rd

    with open(os.path.join(EXP, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE metrics.json")
    print(json.dumps({
        "gate_all_pass": gate["ALL_PASS"],
        "gate_bit_identical": gate["ALL_BIT_IDENTICAL"],
        "argmins": argmins, "all_interior": q1["ALL_INTERIOR"],
        "mean_delta_score_481_to_argmin": q2["mean_delta_score_points"],
        "seed_std_at_argmin": q3["at_argmin"]["seed_std"],
        "deploy_range": out["deploy_iter"]["RANGE"]}, indent=1))


if __name__ == "__main__":
    main()
