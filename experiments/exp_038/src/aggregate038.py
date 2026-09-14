"""exp_038 — metrics.json, built from artefacts on disk and nothing else.

READING RULES ENCODED HERE
  * fold_2023 Brier LEVELS are never converted to points.  The conversion
    reference 0.2498069 is the 2025 test base rate; a 2023 level run through it
    produces a number with no referent.  fold_2023 DELTAS are converted, and
    fold_2024 levels and deltas both are.
  * the adoption denominator is the R0 seed std with ddof=1 and nothing else.
    `paired_delta_std` is computed because exp_033 showed candidate and control
    drift together, but it lives in `diagnostics`, never beside the verdict.
  * `adoption_threshold_sigma_multiplier` is the RULE (the constant 3);
    `observed_delta_in_R0_sigma_units` is the DATUM.  Same unit, different
    kind of thing.  They are given different names so no sentence can slide
    from one to the other.
  * the two iteration scales are written as two keys.  exp_037's aggregate
    hardcoded one rounded number for both and it was wrong for both; this file
    reads `scale_mode` and `scale_full` out of the stage-2 result jsons.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

from arms import ARMS, EXPECTED_N

REF_BRIER = 0.2498069
A_ARMS = ["A_idea", "A_cnt", "A_match"]
PROMOTED = "A_idea"          # fixed before stage A ran; see config B_promote
SEEDS = [42, 7, 2024]
PINNED_FEAT_SHA = (
    "312d0ccb0d293d0193256d2067404c4b11a93505a6d28d72d8a09b532475acd4")


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def dscore(d):
    """Brier delta -> point delta.  Lower Brier is a higher score."""
    return -100000.0 * d / REF_BRIER


def r5(x):
    return None if x is None else float(f"{x:.5f}")


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(res_dir, arm, unit, seed, stage):
    p = os.path.join(res_dir, f"res_{arm}_u{unit}_seed{seed}_st{stage}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-exp", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    src37 = os.path.abspath(os.path.expanduser(args.src_exp))
    res_dir = os.path.join(out, "res")

    with open(os.path.join(src37, "metrics.json")) as f:
        M37 = json.load(f)
    with open(os.path.join(out, "inherit_gate.json")) as f:
        GATE = json.load(f)

    M = {
        "exp_id": "exp_038",
        "title": ("`idea`'s two columns put on R0 alone, and split — a "
                  "MEASUREMENT, not a verdict"),
        "what_this_experiment_is": (
            "exp_037 measured the `idea` block twice, both times with other "
            "blocks in the matrix: +17.53141 as a ladder rung on top of "
            "p6+sit, +12.28771 as a leave-one-out from all 22 columns. "
            "Neither is its value on our 47 columns alone, and neither says "
            "which of the two columns does the work. The success criterion "
            "here is whether that separation was achieved, NOT whether the "
            "adoption bar was cleared. The config's honest_prior predicted "
            "roughly a third of the bar before anything ran."),
        "conversion": {
            "formula": "100000 * (1 - brier / 0.2498069)",
            "reference_brier": REF_BRIER,
            "sensitivity": ("Brier 1e-6 = 0.4003 points, so a 6-digit Brier "
                            "carries +/-0.2 of slack and the trailing digits "
                            "of a converted value are a machine transform of "
                            "a rounded input, not a measurement"),
            "fold_2023_levels_not_converted": (
                "the reference is the 2025 test base rate; a 2023 validation "
                "LEVEL run through it has no referent. On fold_2023 only "
                "DELTAS are converted."),
        },
        "changed_vs_exp_037": [
            "1. arm column sets: 22-column combinations -> the `idea` block "
            "and its two halves alone (49 / 48 / 48 columns)",
            "2. the R0 control is INHERITED by sha, not refitted "
            "(fold_2023 seed 42 and fold_2024 seeds 42/7/2024)",
            "3. featgen.py:209 nan_to_num(P, nan=-1) -> assert not "
            "np.isnan(P).any() (exp_037 audit finding N4)",
            "4. the two iteration scales are READ from the result jsons "
            "instead of a hardcoded rounded constant (audit finding N1)",
            "5. the promoted arm was fixed BEFORE stage A ran, so there is no "
            "selection over arms and therefore no selection cost",
            "carried-along, not intended: the feature parquet is not "
            "regenerated, so featgen.py never executes in this experiment and "
            "change 3 has no numerical effect here; b3_feat.py is inherited "
            "byte-identical and still carries EXP_ID='exp_037' inside the "
            "result jsons it writes",
        ],
        "inheritance": {
            "PASS": GATE.get("PASS"),
            "feat_parquet_sha256": GATE["checks"]["feat_parquet"].get(
                "observed_sha256"),
            "feat_parquet_pinned_matches": GATE["checks"]["feat_parquet"].get(
                "PASS"),
            "exp037_leak_gates": GATE["checks"].get("exp037_leak_gates"),
            "R0_units": {k: {"PASS": v.get("PASS"),
                             "sha_match": v.get("sha_match"),
                             "brier_recomputed_from_npy":
                                 v.get("brier_recomputed_from_npy"),
                             "abs_diff_vs_exp037":
                                 v.get("abs_diff_recomputed_vs_expected")}
                         for k, v in GATE["checks"]["R0_inherited_units"].items()},
            "why_no_new_audit": (
                "no new feature source, join, target encoding or time "
                "aggregation; the parquet is byte-identical to the one "
                "exp_037's audit cleared. A submission built on this would "
                "still require the pre-submission audit, without exception."),
        },
        "src_sha256": {os.path.basename(p): sha256_file(p) for p in
                       sorted(glob.glob(os.path.join(out, "src", "*.py")))
                       + sorted(glob.glob(os.path.join(out, "src", "*.sh")))},
        "arms": {a: {"n_feats": EXPECTED_N[a], "extra_cols": ARMS[a]["cols"]}
                 for a in A_ARMS},
    }

    # -------------------------------------------------- inherited R0 numbers
    b23_R0 = None
    for r in M37["stage_B_ladder"]["rungs"]:
        if r["arm"] == "R0":
            b23_R0 = r["brier_fold2023"]
    R0_24 = M37["stage_D_promotion"]["arms"]["R0"]
    b24_R0 = {s: R0_24["seeds"][str(s)]["brier_fold2024"] for s in SEEDS}

    # ============================================== stage A: the decomposition
    stageA, feat_sha_seen, caps = {}, set(), {}
    for a in A_ARMS:
        r = load(res_dir, a, 2023, 42, 1)
        if r is None:
            stageA[a] = {"MISSING": True}
            continue
        b = r["brier_full"]
        d = b - b23_R0
        feat_sha_seen.add(r["sha256"].get("feat_parquet"))
        caps[a] = r.get("cap")
        stageA[a] = {
            "n_feats": r["n_feats_used"], "extra_cols": r["extra_cols"],
            "brier_fold2023": b,
            "score_fold2023": None,
            "score_fold2023_note": "levels are not converted on fold_2023",
            "delta_vs_R0_brier": d,
            "delta_vs_R0_score": r5(dscore(d)),
            "stack_argmin": r["stack_argmin_iter"],
            "cell_argmin": r["cell_argmin_iter"],
            "any_argmin_at_cap": r["any_argmin_at_cap"],
            "cap": r.get("cap"),
            "minutes": r.get("minutes"),
            "extra_nan_rate_train": r.get("extra_nan_rate_train"),
            "extra_nan_rate_valid": r.get("extra_nan_rate_valid"),
            "valid_R_brier": r["valid_R"]["brier"],
            "valid_F_brier": r["valid_F"]["brier"],
        }
    M["stage_A_decompose"] = {
        "fold": "fold_2023 (train <= 2022 -> valid 2023), seed 42, "
                "self-stopping under a cap",
        "control": {"arm": "R0", "n_feats": 47,
                    "brier_fold2023": b23_R0,
                    "source": "INHERITED from exp_037 by sha; not refitted"},
        "cap": caps,
        "arms": stageA,
    }

    got = {a: stageA[a].get("delta_vs_R0_brier") for a in A_ARMS}
    if all(got[a] is not None for a in A_ARMS):
        s_singles = got["A_cnt"] + got["A_match"]
        inter = got["A_idea"] - s_singles
        M["stage_A_decompose"]["additivity"] = {
            "note": ("there is no reason for the two single-column deltas to "
                     "add up to the pair's delta, and the gap between them is "
                     "the interaction between the two columns inside the "
                     "stack. All three are reported; none is derived."),
            "delta_A_cnt_score": r5(dscore(got["A_cnt"])),
            "delta_A_match_score": r5(dscore(got["A_match"])),
            "sum_of_singles_score": r5(dscore(s_singles)),
            "delta_A_idea_score": r5(dscore(got["A_idea"])),
            "interaction_brier": inter,
            "interaction_score": r5(dscore(inter)),
            "interaction_reading": (
                "positive interaction_score means the pair is worth more than "
                "its two halves added; negative means the two columns overlap "
                "and the pair is worth less."),
        }
    M["stage_A_decompose"]["feat_parquet_sha_in_every_unit"] = sorted(
        x for x in feat_sha_seen if x)
    M["stage_A_decompose"]["feat_parquet_sha_matches_pin"] = bool(
        feat_sha_seen and feat_sha_seen == {PINNED_FEAT_SHA})

    # ============================================== stage B: the promoted arm
    per, scales = {}, {}
    for s in SEEDS:
        r = load(res_dir, PROMOTED, 2024, s, 2)
        if r is None:
            continue
        per[s] = r
        scales[s] = (r.get("scale_mode"), r.get("scale_full"))
    B = {
        "fold": "fold_2024 (train <= 2023 -> valid 2024), seeds 42/7/2024",
        "arm": PROMOTED,
        "arm_was_fixed_before_stage_A": True,
        "selection_cost": 0.0,
        "selection_cost_note": (
            "one candidate means no choice, and no choice means no winner's "
            "curse to pay. exp_037 chose the best of nine on fold_2023 and "
            "gave back 4.56965 points on fold_2024; that price is not paid "
            "here. What remains between the folds is season difference, not "
            "selection."),
        "control": {"arm": "R0", "source": "INHERITED from exp_037 by sha",
                    "seeds": {str(s): b24_R0[s] for s in SEEDS},
                    "mean_brier": R0_24["mean_brier"],
                    "mean_score": R0_24["mean_score"],
                    "std_brier_ddof1": R0_24["std_brier_ddof1"]},
    }
    if per:
        B["seeds"] = {str(s): {
            "brier_fold2024": per[s]["brier_full"],
            "score_fold2024": r5(score_of(per[s]["brier_full"])),
            "stack_round": per[s]["stack_round"],
            "cell_round": per[s]["cell_round"],
            "valid_R": per[s]["valid_R"], "valid_F": per[s]["valid_F"],
            "minutes": per[s].get("minutes"),
        } for s in per}
        bs = [per[s]["brier_full"] for s in SEEDS if s in per]
        B["mean_brier"] = float(np.mean(bs))
        B["mean_score"] = r5(score_of(B["mean_brier"]))
        B["std_brier_ddof1"] = (float(np.std(bs, ddof=1))
                                if len(bs) > 1 else None)
        B["n_seeds"] = len(bs)
        B["n_feats"] = per[list(per)[0]]["n_feats_used"]

        sm = {v[0] for v in scales.values()}
        sf = {v[1] for v in scales.values()}
        B["iteration_policy"] = {
            "rule": ("stage-2 rounds = the arm's OWN fold_2023 argmin x the "
                     "training-row ratio. The fold_2024 validation curve "
                     "selected nothing."),
            "scale_mode": (sm.pop() if len(sm) == 1 else sorted(map(str, sm))),
            "scale_full": (sf.pop() if len(sf) == 1 else sorted(map(str, sf))),
            "scales_are_two_different_numbers": True,
            "n1_note": ("exp_037's aggregate wrote 'stacker x1.281957' for "
                        "both; the cell scale and the stacker scale are not "
                        "the same number and neither equals that constant. "
                        "These two values are read from the result jsons."),
            "base_stack_argmin": per[list(per)[0]].get("base_stack_argmin"),
            "base_cell_argmin": per[list(per)[0]].get("base_cell_argmin"),
        }

    # ------------------------------------------------------------- adoption
    if per and len(per) == 3:
        db = [per[s]["brier_full"] - b24_R0[s] for s in SEEDS]
        sigma = float(np.std([b24_R0[s] for s in SEEDS], ddof=1))
        mean_d = float(np.mean(db))
        signs = {int(np.sign(x)) for x in db}
        agree = len(signs) == 1 and 0 not in signs
        units = abs(mean_d) / sigma if sigma > 0 else float("inf")
        B["adoption"] = {
            "rule": ("three seeds agree in sign AND |mean delta| >= 3 x "
                     "std(R0 seeds, ddof=1) AND the mean delta lowers Brier. "
                     "Pre-registered; not relaxed."),
            "per_seed_delta_brier": {str(s): d for s, d in zip(SEEDS, db)},
            "per_seed_delta_score": {str(s): r5(dscore(d))
                                     for s, d in zip(SEEDS, db)},
            "mean_delta_brier": mean_d,
            "mean_delta_score": r5(dscore(mean_d)),
            "R0_seed_sigma_brier_ddof1": sigma,
            "R0_seed_sigma_score_equiv": r5(100000.0 * sigma / REF_BRIER),
            "adoption_bar_score_equiv": r5(3 * 100000.0 * sigma / REF_BRIER),
            "ddof": 1,
            "adoption_threshold_sigma_multiplier": 3,
            "observed_delta_in_R0_sigma_units": (
                float(f"{units:.4f}") if np.isfinite(units) else None),
            "naming_warning": (
                "adoption_threshold_sigma_multiplier is the RULE, a constant "
                "3. observed_delta_in_R0_sigma_units is the DATUM. They share "
                "a unit and are not the same kind of quantity; neither may be "
                "compared across experiments that used a different ddof."),
            "signs_agree": bool(agree),
            "ADOPTED": bool(agree and abs(mean_d) >= 3 * sigma and mean_d < 0),
        }
        # fold_2023 -> fold_2024 transfer, with no selection in between
        d23 = stageA.get(PROMOTED, {}).get("delta_vs_R0_brier")
        if d23 is not None:
            s23, s24 = dscore(d23), dscore(mean_d)
            B["transfer"] = {
                "fold2023_delta_score": r5(s23),
                "fold2024_mean_delta_score": r5(s24),
                "transfer_ratio": (float(f"{s24 / s23:.4f}") if s23 else None),
                "note": ("with the arm fixed in advance this ratio contains "
                         "no winner's curse; what it contains is the "
                         "difference between the two seasons plus seed noise."),
            }
        # ----------------------------------------------------- segments
        segs = {}
        for name in ("valid_R", "valid_F"):
            cb = [R0_24["seeds"][str(s)][name]["brier"] for s in SEEDS]
            ab = [per[s][name]["brier"] for s in SEEDS]
            dd = float(np.mean(ab) - np.mean(cb))
            segs[name] = {
                "n": per[SEEDS[0]][name]["n"],
                "R0_mean_brier": float(np.mean(cb)),
                f"{PROMOTED}_mean_brier": float(np.mean(ab)),
                "mean_delta_brier": dd,
                "mean_delta_score_equiv": r5(dscore(dd)),
                "per_seed_delta_score_equiv": {
                    str(s): r5(dscore(per[s][name]["brier"]
                                      - R0_24["seeds"][str(s)][name]["brier"]))
                    for s in SEEDS},
                "R0_mean_auc": float(np.mean(
                    [R0_24["seeds"][str(s)][name]["auc"] for s in SEEDS])),
                f"{PROMOTED}_mean_auc": float(np.mean(
                    [per[s][name]["auc"] for s in SEEDS])),
            }
            segs[name]["conversion_caveat"] = (
                "a segment delta converted at the full-test reference answers "
                "'what would this be worth if the whole test looked like this "
                "segment'. The real test is roughly 89% R and 11% F, so these "
                "two numbers do not add up to the all-rows number.")
        B["segments"] = segs
    M["stage_B_promotion"] = B

    # ============================================================ diagnostics
    D = {}
    if per and len(per) == 3:
        db = [per[s]["brier_full"] - b24_R0[s] for s in SEEDS]
        pstd = float(np.std(db, ddof=1))
        D["paired_delta_std"] = {
            "brier": pstd,
            "score_equiv": r5(100000.0 * pstd / REF_BRIER),
            "what": ("the spread of the per-seed (candidate - control) "
                     "differences, i.e. how much the two arms move apart when "
                     "the seed changes"),
            "NOT_A_DENOMINATOR": (
                "the adoption test divides by the R0 seed std and by nothing "
                "else. This number is a diagnostic: exp_033 found it far "
                "smaller than the control std, meaning candidate and control "
                "swing together. It is deliberately kept out of the adoption "
                "block so no table can put the two side by side."),
        }
    gp = os.path.join(res_dir, f"gain_probe_{PROMOTED}.json")
    if os.path.exists(gp):
        with open(gp) as f:
            D["column_importance_probe"] = json.load(f)
    M["diagnostics"] = D

    # ------------------------------------------------ standard fold summary
    M["fold_2023"] = {
        "note": "levels are not point-converted on this fold; deltas are",
        "R0_brier_inherited": b23_R0,
        **{f"{a}_brier": stageA[a].get("brier_fold2023") for a in A_ARMS},
        **{f"{a}_delta_score": stageA[a].get("delta_vs_R0_score")
           for a in A_ARMS},
    }
    M["fold_2024"] = {
        "arm": PROMOTED,
        "brier": B.get("mean_brier"),
        "score": B.get("mean_score"),
        "bss": (float(np.mean([per[s]["valid_all"]["bss"] for s in per]))
                if per else None),
        "auc": (float(np.mean([per[s]["valid_all"]["auc"] for s in per]))
                if per else None),
        "R0_brier_inherited": R0_24["mean_brier"],
        "R0_score_inherited": R0_24["mean_score"],
        "mean_delta_score": B.get("adoption", {}).get("mean_delta_score"),
        "ADOPTED": B.get("adoption", {}).get("ADOPTED"),
    }

    # ---------------------------------------------------------------- timing
    mins = []
    for f in glob.glob(os.path.join(res_dir, "res_*.json")):
        with open(f) as fh:
            mins.append(json.load(fh).get("minutes") or 0.0)
    M["train_time_min_sum_of_units"] = round(float(np.sum(mins)), 2)
    M["n_units_completed"] = len(mins)
    M["units_reused_not_refitted"] = 4
    prog = os.path.join(out, "PROGRESS.jsonl")
    if os.path.exists(prog):
        ev = [json.loads(x) for x in open(prog) if x.strip()]
        M["failures"] = [e for e in ev if e["event"] == "END" and e["rc"] != 0]
        M["n_progress_events"] = len(ev)

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(M, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({
        "inherit_PASS": M["inheritance"]["PASS"],
        "fold2023_delta_score": {a: stageA[a].get("delta_vs_R0_score")
                                 for a in A_ARMS},
        "fold2024_mean_delta_score": B.get("adoption", {}).get(
            "mean_delta_score"),
        "bar": B.get("adoption", {}).get("adoption_bar_score_equiv"),
        "ADOPTED": B.get("adoption", {}).get("ADOPTED")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
