"""exp_037 — metrics.json and report_derived.json, from artefacts only.

Every number the report is allowed to contain is computed here from files on
disk.  Nothing is transcribed from a log line and nothing is remembered.

CONVERSION DISCIPLINE
  score = 100000 * (1 - brier / 0.2498069).  The reference is the 2025 test
  base rate, so:
    * fold_2024 Brier levels ARE converted (that is the fold the reference was
      calibrated for as a proxy),
    * fold_2023 Brier LEVELS ARE NOT converted — the field is left null and the
      reason is written into the artefact,
    * DELTAS are converted on every fold, because a Brier delta maps to a point
      delta at a fixed reference independently of the level.
  Converted values carry at most 5 decimals and an explicit note that a 6-digit
  Brier gives them +/- 0.2 of slack.

ADOPTION RULE (pre-registered 2026-08-08, applied forward only)
  three seeds agree in sign  AND  |mean delta| >= 3 * std(R0 seeds, ddof=1).
  Two different quantities are reported and are deliberately given different
  names so they are never compared to each other:
      adoption_threshold_sigma_multiplier   = 3      (the rule)
      observed_delta_in_R0_sigma_units      = |mean delta| / sigma  (the datum)
  Both are ddof=1.  Numbers from experiments that used ddof=0 (exp_036) are on
  a different scale and must not be placed beside these.
"""
import argparse
import glob
import json
import os

import numpy as np

from arms import ARMS, DUPLICATE_OF, EXPECTED_N

REF_BRIER = 0.2498069
LADDER = ["R0", "R1", "R2", "R3", "R4"]
LOO = ["L_no_p6", "L_no_sit", "L_no_idea", "L_no_i3", "L_no_p6ALL"]
MATE_LADDER_DELTA_SCORE = {"R1": 53.6, "R2": 17.7, "R3": 43.5, "R4": 22.2}


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def dscore(d):
    """Brier delta -> point delta.  Lower Brier = higher score."""
    return -100000.0 * d / REF_BRIER


def r5(x):
    return None if x is None else float(f"{x:.5f}")


def load(res_dir, arm, unit, seed, stage):
    p = os.path.join(res_dir, f"res_{arm}_u{unit}_seed{seed}_st{stage}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    feat = os.path.abspath(args.feat)
    res_dir = os.path.join(out, "res")

    M = {"exp_id": "exp_037",
         "title": "team-mate exp_032's 22 columns transplanted into our "
                  "C5_B3 base — ladder + LOO on fold_2023, promotion on "
                  "fold_2024",
         "conversion": {
             "formula": "100000 * (1 - brier / 0.2498069)",
             "reference_brier": REF_BRIER,
             "sensitivity": "Brier 1e-6 = 0.4003 points; a 6-digit Brier "
                            "therefore carries +/-0.2 of conversion slack, so "
                            "the last decimals of a converted value are a "
                            "machine transform of a rounded input, not a "
                            "measurement",
             "fold_2023_levels_not_converted": (
                 "the reference is the 2025 test base rate; converting a 2023 "
                 "validation level would produce a number with no referent. "
                 "Only DELTAS are converted on fold_2023.")},
         "leak_gates": {}, "stage_A": {}, "stage_B_ladder": {},
         "stage_C_loo": {}, "stage_D_promotion": {}, "failures": []}

    # ------------------------------------------------------------- stage A
    for name, path in [("verify_asof", os.path.join(feat, "verify_asof.json")),
                       ("verify_groupconst",
                        os.path.join(feat, "verify_groupconst.json")),
                       ("gate_2024", os.path.join(feat, "gate_2024.json"))]:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            M["leak_gates"][name] = {"PASS": d.get("PASS"),
                                     "file": os.path.relpath(path, out)}
            if name == "verify_asof":
                M["leak_gates"][name]["per_season"] = {
                    k: {kk: v[kk] for kk in v
                        if kk in ("PASS", "hist_season_max_RELOOKED_UP",
                                  "tm_season_max_RELOOKED_UP",
                                  "hist_violations_gt_s_minus_1",
                                  "tm_violations_gt_s_minus_1",
                                  "hist_set_equals_seasons_le_s_minus_1",
                                  "all_nan")}
                    for k, v in d.get("checks", {}).items()}
            if name == "gate_2024":
                M["leak_gates"][name]["max_abs_diff_over_22_cols"] = max(
                    [v["max_abs_diff"] for v in d.get("columns", {}).values()]
                    or [None])
                M["leak_gates"][name]["table_sha256_identical"] = \
                    d.get("table_sha256_identical")
        else:
            M["leak_gates"][name] = {"PASS": None, "file": None}

    for j in ("mapgen.json", "featgen_main.json"):
        p = os.path.join(feat, j)
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            if j == "mapgen.json":
                M["stage_A"]["pitcher_map_asof"] = {
                    c: {"n_pitchers_mapped": v["n_pitchers_mapped"],
                        "max_season_present": v["max_season_present"]}
                    for c, v in d["cutoffs"].items()}
                M["stage_A"]["pitch_label_recipe"] = d["lab_recipe"]
                M["stage_A"]["mapgen_minutes"] = d.get("minutes")
            else:
                M["stage_A"]["featgen_minutes"] = d.get("minutes")
                M["stage_A"]["feat_sha256"] = d.get("feat_sha256")
                M["stage_A"]["per_season"] = {
                    k: {"n_rows": v["n_rows"], "hist_n": v["hist_n"],
                        "tm_n": v.get("tm_n"),
                        "base_rate_used": v.get("base_rate_used"),
                        "mean_na": (round(float(np.mean(
                            list(v["na_rate"].values()))), 6)
                            if "na_rate" in v else 1.0)}
                    for k, v in d["per_season"].items()}

    # ------------------------------------------------------ stage B + C (2023)
    R = {}
    for a in ARMS:
        r = load(res_dir, a, 2023, 42, 1)
        if r is not None:
            R[a] = r
    for dup, src in DUPLICATE_OF.items():
        if dup not in R and src in R:
            R[dup] = dict(R[src], _alias_of=src)

    b0 = R["R0"]["brier_full"] if "R0" in R else None
    if b0 is not None:
        rows = []
        prev = None
        for a in LADDER:
            if a not in R:
                rows.append({"arm": a, "status": "MISSING"})
                prev = None
                continue
            b = R[a]["brier_full"]
            step = None if prev is None else b - prev
            rows.append({
                "arm": a, "adds": ARMS[a].get("adds", "— (control)"),
                "n_feats": R[a].get("n_feats_used", EXPECTED_N[a]),
                "brier_fold2023": b,
                "score_fold2023": None,
                "step_delta_brier": step,
                "step_delta_score": r5(dscore(step)) if step is not None else None,
                "cum_delta_brier": b - b0,
                "cum_delta_score": r5(dscore(b - b0)),
                "mate_step_delta_score_fold2024": MATE_LADDER_DELTA_SCORE.get(a),
                "stack_argmin": R[a]["stack_argmin_iter"],
                "cell_argmin": R[a]["cell_argmin_iter"],
                "argmin_at_cap": R[a].get("any_argmin_at_cap"),
                "cap": R[a].get("cap"),
                "minutes": R[a].get("minutes"),
            })
            prev = b
        M["stage_B_ladder"] = {
            "fold": "fold_2023 (train<=2022 -> valid 2023), seed 42, cells and "
                    "stacker self-stopping under a cap",
            "order_note": ("the rung order is the team-mate's, so that our step "
                           "deltas sit on the same ordinal axis as theirs. A "
                           "step delta is ORDER DEPENDENT: it is the value of "
                           "that block ON TOP OF the blocks before it, not its "
                           "standalone value."),
            "comparability_warning": ("the team-mate measured on fold_2024 "
                                      "inside a flat LGBM/CatBoost blend; we "
                                      "measure on fold_2023 inside C5_B3. Only "
                                      "deltas may be compared, never levels."),
            "rungs": rows}
        if "R0" in R and "exp030_gate" in R["R0"]:
            M["stage_B_ladder"]["exp030_reproduction_gate"] = R["R0"]["exp030_gate"]

        # ---- LOO, measured against R4
        b4 = R["R4"]["brier_full"] if "R4" in R else None
        loo = []
        for a in LOO:
            if a not in R:
                loo.append({"arm": a, "status": "MISSING"})
                continue
            b = R[a]["brier_full"]
            d = None if b4 is None else b - b4
            loo.append({
                "arm": a, "drops": ARMS[a].get("drops"),
                "alias_of": R[a].get("_alias_of"),
                "n_feats": R[a].get("n_feats_used", EXPECTED_N[a]),
                "brier_fold2023": b,
                "delta_vs_R4_brier": d,
                "delta_vs_R4_score": r5(dscore(d)) if d is not None else None,
                "value_of_removed_block_score":
                    r5(-dscore(d)) if d is not None else None,
                "reading": ("delta_vs_R4_score is THIS ARM's score minus R4's: "
                            "negative means removing the block cost that much. "
                            "value_of_removed_block_score is its negation, i.e. "
                            "what the block was worth on top of everything "
                            "else; negative there means the block was HARMFUL "
                            "and removing it helped."),
                "delta_vs_R0_brier": b - b0,
                "delta_vs_R0_score": r5(dscore(b - b0)),
                "stack_argmin": R[a]["stack_argmin_iter"],
                "minutes": R[a].get("minutes"),
            })
        M["stage_C_loo"] = {
            "fold": "fold_2023, seed 42",
            "reference": "R4 (all 69 columns)",
            "derivation_chain_note": (
                "idea_matchup_pitch and every i3_anom_* column are functions of "
                "the SAME P vector the p6_* columns are. L_no_p6 therefore does "
                "NOT remove the pitch-type signal — seven P-derived columns "
                "survive it — so it measures only what the raw probabilities "
                "add on top of their own derivatives. L_no_p6ALL is the arm "
                "that removes every P-derived column and is the one to read as "
                "'without the pitch model'."),
            "alias_note": ("L_no_i3 has the same column set as R3 by "
                           "construction (R4 minus i3 == R3); it is aliased, "
                           "not refitted, and its row here repeats R3's fit."),
            "arms": loo}

    # ------------------------------------------------------------- stage D
    sel = None
    p = os.path.join(out, "selection.json")
    if os.path.exists(p):
        with open(p) as f:
            sel = json.load(f)
    promote = sel["promote_arms"] if sel else ["R0"]
    seeds = [42, 7, 2024]
    D = {}
    for a in promote:
        per = {}
        for s in seeds:
            r = load(res_dir, a, 2024, s, 2)
            if r is None:
                continue
            per[s] = r
        if not per:
            continue
        D[a] = {
            "seeds": {str(s): {
                "brier_fold2024": per[s]["brier_full"],
                "score_fold2024": r5(score_of(per[s]["brier_full"])),
                "stack_round": per[s]["stack_round"],
                "cell_round": per[s]["cell_round"],
                "valid_R": per[s]["valid_R"], "valid_F": per[s]["valid_F"],
                "minutes": per[s].get("minutes"),
                "iter_source": per[s].get("iter_source"),
            } for s in per},
            "mean_brier": float(np.mean([per[s]["brier_full"] for s in per])),
            "std_brier_ddof1": (float(np.std([per[s]["brier_full"] for s in per],
                                             ddof=1)) if len(per) > 1 else None),
            "n_seeds": len(per),
            "n_feats": per[list(per)[0]].get("n_feats_used"),
        }
        D[a]["mean_score"] = r5(score_of(D[a]["mean_brier"]))
    M["stage_D_promotion"] = {"arms": D,
                              "selection": sel,
                              "iteration_policy":
                              "stage 2 — fold_2023 argmin x training-row ratio "
                              "(stacker x1.281957 on unit 2024). The fold_2024 "
                              "validation curve selected nothing."}

    cand = [a for a in D if a != "R0"]
    if "R0" in D and cand and D["R0"]["n_seeds"] == 3:
        a = cand[0]
        if D[a]["n_seeds"] == 3:
            db = [D[a]["seeds"][str(s)]["brier_fold2024"]
                  - D["R0"]["seeds"][str(s)]["brier_fold2024"] for s in seeds]
            sigma = float(np.std([D["R0"]["seeds"][str(s)]["brier_fold2024"]
                                  for s in seeds], ddof=1))
            mean_d = float(np.mean(db))
            signs = {int(np.sign(x)) for x in db}
            agree = len(signs) == 1 and 0 not in signs
            units = abs(mean_d) / sigma if sigma > 0 else float("inf")
            M["stage_D_promotion"]["adoption"] = {
                "arm": a,
                "per_seed_delta_brier": {str(s): d for s, d in zip(seeds, db)},
                "per_seed_delta_score": {str(s): r5(dscore(d))
                                         for s, d in zip(seeds, db)},
                "mean_delta_brier": mean_d,
                "mean_delta_score": r5(dscore(mean_d)),
                "R0_seed_sigma_brier_ddof1": sigma,
                # sigma is a MAGNITUDE, so it converts with the sign dropped;
                # running it through dscore() would print a spurious minus.
                "R0_seed_sigma_score_equiv": r5(100000.0 * sigma / REF_BRIER),
                "adoption_bar_score_equiv": r5(3 * 100000.0 * sigma
                                               / REF_BRIER),
                "ddof": 1,
                "ddof_note": ("pre-registered 2026-08-08, forward only. "
                              "exp_036 used ddof=0; its sigma-unit numbers are "
                              "on a different scale and must not be compared "
                              "with these."),
                "adoption_threshold_sigma_multiplier": 3,
                "observed_delta_in_R0_sigma_units": (
                    float(f"{units:.4f}") if np.isfinite(units) else None),
                "naming_warning": ("adoption_threshold_sigma_multiplier is the "
                                   "RULE (a constant, 3). "
                                   "observed_delta_in_R0_sigma_units is the "
                                   "DATUM. They are different quantities that "
                                   "happen to share a unit; do not read one as "
                                   "the other and do not compare either across "
                                   "experiments with a different ddof."),
                "signs_agree": bool(agree),
                "ADOPTED": bool(agree and abs(mean_d) >= 3 * sigma
                                and mean_d < 0),
                "verdict_note": ("ADOPTED requires sign agreement AND "
                                 "|mean delta| >= 3 sigma AND the mean delta "
                                 "pointing the improving way (Brier down)."),
            }
            # the price of choosing on one fold and reading on another
            if sel and sel.get("best_delta_vs_R0_score") is not None:
                s23 = float(sel["best_delta_vs_R0_score"])
                s24 = dscore(mean_d)
                M["stage_D_promotion"]["honest_selection_cost"] = {
                    "what": ("the arm was picked as the best of 9 on fold_2023 "
                             "and then read once on fold_2024. The gap between "
                             "the two is the price of that selection plus "
                             "whatever differs between the two seasons; this "
                             "experiment cannot separate the two causes."),
                    "fold2023_selected_delta_score": r5(s23),
                    "fold2024_mean_delta_score": r5(s24),
                    "transfer_ratio": float(f"{s24 / s23:.4f}") if s23 else None,
                    "shrinkage_score": r5(s23 - s24),
                }

    # ---------------------------------------------------------------- timing
    mins = []
    for f in glob.glob(os.path.join(res_dir, "res_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        mins.append(d.get("minutes") or 0.0)
    M["train_time_min_sum_of_units"] = round(float(np.sum(mins)), 2)
    M["n_units_completed"] = len(mins)

    prog = os.path.join(out, "PROGRESS.jsonl")
    if os.path.exists(prog):
        ev = [json.loads(x) for x in open(prog) if x.strip()]
        M["failures"] = [e for e in ev if e["event"] == "END" and e["rc"] != 0]
        M["n_progress_events"] = len(ev)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(M, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({"gates": {k: v["PASS"] for k, v in M["leak_gates"].items()},
                      "ladder": [r.get("cum_delta_score")
                                 for r in M["stage_B_ladder"].get("rungs", [])],
                      "adoption": M["stage_D_promotion"].get("adoption", {})
                      .get("ADOPTED")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
