"""exp_050 aggregation.

Builds `metrics.json` from
  * the exp_050 grid runs   res/res_exp050_st{S}_oof{O}.json
  * exp_049's diagonal      experiments/exp_049/metrics.json   (st==oof, seeds 42/7/2024)

The quantity of interest is `L_no_p6 - A1` in converted score points
(positive => the 75-column arm is better).  It is reported per round, per grid
cell, and decomposed into the stacker-seed axis and the cell-OOF axis.

Run:  python aggregate.py <res_dir> <exp049_metrics.json> <out metrics.json>
"""
import json
import os
import statistics as st
import sys

REF_BRIER = 0.2498069
ROUNDS = ["655", "1000", "1400"]
OOF_SEEDS = [42, 7, 2024]
NEW_STACKER_SEEDS = [13, 101, 777]


def score(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def r5(x):
    return float(f"{x:.5f}")


def sd(xs):
    return r5(st.stdev(xs)) if len(xs) > 1 else None


def main():
    res_dir, e049_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(e049_path) as f:
        e049 = json.load(f)

    # ---- cells: (stacker_seed, oof_seed) -> {round: {A1, L}} ----------------
    cells = {}
    provenance = {}
    anchor = None
    missing = []

    for S in [42] + NEW_STACKER_SEEDS:
        for O in OOF_SEEDS:
            p = os.path.join(res_dir, f"res_exp050_st{S}_oof{O}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                d = json.load(f)
            key = f"st{S}_oof{O}"
            cells[key] = {
                "stacker_seed": S, "oof_seed": O, "source": "exp_050",
                "by_round": {r: {
                    "brier_A1": d["arms"]["A1"]["by_round"][r]["valid_all_brier"],
                    "brier_L_no_p6":
                        d["arms"]["L_no_p6"]["by_round"][r]["valid_all_brier"],
                } for r in ROUNDS},
                "minutes": d["minutes"],
            }
            provenance[key] = {
                "oof_sha256": d["inherited"]["oof_sha256"],
                "s1_seedvar_sha256": d["sha256"]["s1_seedvar.py"],
                "train_csv_sha256": d["sha256"]["train.csv"],
                "feat_parquet_sha256": d["inherited"]["feat_parquet_sha256"],
                "catboost": d["versions"]["catboost"],
            }
            if d["identity_anchor_vs_exp049_seed42"].get("applicable"):
                anchor = d["identity_anchor_vs_exp049_seed42"]

    # ---- exp_049's diagonal (stacker seed == oof seed) ---------------------
    for S in OOF_SEEDS:
        key = f"st{S}_oof{S}"
        if key in cells:
            continue  # st42_oof42 comes from exp_050 and is sha-anchored to it
        cells[key] = {
            "stacker_seed": S, "oof_seed": S, "source": "exp_049 metrics.json",
            "by_round": {r: {
                "brier_A1":
                    e049["fold_2024"][r]["A1"]["per_seed_brier"][str(S)],
                "brier_L_no_p6":
                    e049["fold_2024"][r]["L_no_p6"]["per_seed_brier"][str(S)],
            } for r in ROUNDS},
            "minutes": None,
        }

    for S in NEW_STACKER_SEEDS:
        for O in OOF_SEEDS:
            if f"st{S}_oof{O}" not in cells:
                missing.append(f"st{S}_oof{O}")

    # ---- the contrast ------------------------------------------------------
    for key, c in cells.items():
        for r in ROUNDS:
            b1 = c["by_round"][r]["brier_A1"]
            bl = c["by_round"][r]["brier_L_no_p6"]
            c["by_round"][r]["delta_brier_L_minus_A1"] = bl - b1
            c["by_round"][r]["delta_score_L_minus_A1"] = r5(
                -100000.0 * (bl - b1) / REF_BRIER)
            c["by_round"][r]["score_A1"] = r5(score(b1))
            c["by_round"][r]["score_L_no_p6"] = r5(score(bl))
            c["by_round"][r]["L_no_p6_better"] = bool(bl < b1)

    # ---- sign census over every cell we have ------------------------------
    census = {}
    for r in ROUNDS:
        vals = {k: c["by_round"][r]["delta_score_L_minus_A1"]
                for k, c in cells.items()}
        pos = [k for k, v in vals.items() if v > 0]
        census[r] = {
            "n_cells": len(vals),
            "n_positive_L_no_p6_better": len(pos),
            "n_negative_A1_better": len(vals) - len(pos),
            "positive_cells": sorted(pos),
            "mean_delta_score": r5(sum(vals.values()) / len(vals)),
            "sd_delta_score_ddof1": sd(list(vals.values())),
            "min_delta_score": r5(min(vals.values())),
            "max_delta_score": r5(max(vals.values())),
            "all_same_sign": len(pos) in (0, len(vals)),
            "per_cell": vals,
        }

    # ---- variance decomposition -------------------------------------------
    # stacker axis: hold the OOF fixed, vary the stacker seed
    # cell/OOF axis: hold the stacker seed fixed, vary the OOF
    decomp = {}
    for r in ROUNDS:
        by_oof = {}
        for O in OOF_SEEDS:
            xs = {k: c["by_round"][r] for k, c in cells.items()
                  if c["oof_seed"] == O}
            if len(xs) < 2:
                continue
            by_oof[str(O)] = {
                "n": len(xs),
                "stacker_seeds": sorted(c["stacker_seed"] for k, c in
                                        cells.items() if c["oof_seed"] == O),
                "A1_score": {k: v["score_A1"] for k, v in xs.items()},
                "L_no_p6_score": {k: v["score_L_no_p6"] for k, v in xs.items()},
                "sd_A1_score": sd([v["score_A1"] for v in xs.values()]),
                "sd_L_no_p6_score": sd([v["score_L_no_p6"]
                                        for v in xs.values()]),
                "delta_score": {k: v["delta_score_L_minus_A1"]
                                for k, v in xs.items()},
                "mean_delta_score": r5(
                    sum(v["delta_score_L_minus_A1"] for v in xs.values())
                    / len(xs)),
                "sd_delta_score": sd([v["delta_score_L_minus_A1"]
                                      for v in xs.values()]),
            }
        by_stacker = {}
        for S in NEW_STACKER_SEEDS:
            xs = {k: c["by_round"][r] for k, c in cells.items()
                  if c["stacker_seed"] == S}
            if len(xs) < 2:
                continue
            by_stacker[str(S)] = {
                "n": len(xs),
                "oof_seeds": sorted(c["oof_seed"] for k, c in cells.items()
                                    if c["stacker_seed"] == S),
                "sd_A1_score": sd([v["score_A1"] for v in xs.values()]),
                "sd_L_no_p6_score": sd([v["score_L_no_p6"]
                                        for v in xs.values()]),
                "delta_score": {k: v["delta_score_L_minus_A1"]
                                for k, v in xs.items()},
                "sd_delta_score": sd([v["delta_score_L_minus_A1"]
                                      for v in xs.values()]),
            }
        decomp[r] = {
            "fix_OOF_vary_stacker_seed": by_oof,
            "fix_stacker_seed_vary_OOF": by_stacker,
            "note": ("`fix_OOF_vary_stacker_seed` isolates the stacker's own "
                     "randomness.  `fix_stacker_seed_vary_OOF` isolates the "
                     "cell-model seed carried in the inherited OOF.  exp_049's "
                     "'seed' moved BOTH at once, so its sd is the sum of these "
                     "two sources and is not comparable to either alone."),
        }

    # ---- exp_049's own numbers, recomputed from its metrics.json ----------
    e049_recheck = {}
    for r in ROUNDS:
        f = e049["fold_2024"][r]
        vals = {}
        for S in OOF_SEEDS:
            b1 = f["A1"]["per_seed_brier"][str(S)]
            bl = f["L_no_p6"]["per_seed_brier"][str(S)]
            vals[str(S)] = r5(-100000.0 * (bl - b1) / REF_BRIER)
        e049_recheck[r] = {
            "delta_score_L_minus_A1_per_seed": vals,
            "mean": r5(sum(vals.values()) / 3),
            "sd_ddof1": sd(list(vals.values())),
            "all_positive": all(v > 0 for v in vals.values()),
            "metrics_json_all_same_sign_field":
                f["LOO_family_worth"]["p6_* (6)"]["all_same_sign"],
            "A1_delta_vs_A0_mean_score":
                f["delta_vs_A0"]["A1"]["mean_delta_score"],
            "A1_delta_vs_A0_sd": f["delta_vs_A0"]["A1"]["sd_delta_score_ddof1"],
            "L_no_p6_delta_vs_A0_mean_score":
                f["delta_vs_A0"]["L_no_p6"]["mean_delta_score"],
            "L_no_p6_delta_vs_A0_sd":
                f["delta_vs_A0"]["L_no_p6"]["sd_delta_score_ddof1"],
        }

    # ---- headline -----------------------------------------------------------
    headline = {}
    for r in ROUNDS:
        c = census[r]
        n = c["n_cells"]
        se = c["sd_delta_score_ddof1"] / (n ** 0.5)
        headline[r] = {
            "n_cells": n,
            "n_favouring_L_no_p6": c["n_positive_L_no_p6_better"],
            "mean_delta_score_L_minus_A1": c["mean_delta_score"],
            "sd_ddof1": c["sd_delta_score_ddof1"],
            "naive_standard_error": r5(se),
            "naive_sigma_multiple_from_zero": r5(
                abs(c["mean_delta_score"]) / se) if se else None,
            "exp049_preregistered_threshold_score": 25.65339,
            "🔴_SE_caveat": (
                "the 12 cells are NOT independent: three are exp_049's "
                "diagonal, and every cell sharing an OOF seed shares the same "
                "four frozen cell models.  This SE is therefore optimistic — "
                "the true spread is wider, not narrower."),
        }

    out = {
        "exp_id": "exp_050",
        "headline": headline,
        "date": "2026-08-15",
        "title": ("stacker-seed x cell-OOF grid for the exp_049 A1(81) vs "
                  "L_no_p6(75) ordering, fold_2024"),
        "🔴_deviation_from_config": {
            "what_config_asked": ("re-run the two arms with seeds 13 / 101 / "
                                  "777, each with its own fold_2024 OOF"),
            "why_not_done": (
                "the fold_2024 OOF exists only for seeds 42 / 7 / 2024 "
                "(build_AH/localref/oof/, and local_reference_AH.json has "
                "per_seed keys {42,7,2024}).  Regenerating one costs 17-21 "
                "min/seed measured from build_AH/localref/logs (19.67 / 22.37 "
                "/ 23.38 min per seed total, of which the stacker fit is "
                "2.10-2.45 min), and it would break config.yaml's own "
                "invariant list ('cells frozen, OOF inherited')."),
            "what_was_done_instead": (
                "a 4 x 3 grid: stacker seeds {42,13,101,777} x inherited OOF "
                "seeds {42,7,2024}.  The three exp_049 diagonal cells are "
                "carried in from exp_049/metrics.json.  This holds every "
                "config invariant and additionally SEPARATES the two noise "
                "sources exp_049 confounded."),
            "🔴_config_premise_was_wrong": (
                "config.yaml states 'exp_049 was 3/3'.  That is true at 655 "
                "and 1000 rounds only.  At 1400 -- the round every number "
                "config.yaml quotes comes from -- exp_049's own metrics.json "
                "already recorded all_same_sign = false (seed 42 goes the "
                "other way, -11.86175).  See exp049_recheck below."),
        },
        "score_conversion": {
            "formula": "100000 * (1 - Brier / 0.2498069)",
            "ref_brier": REF_BRIER, "decimals": 5,
            "sensitivity": ("Brier 1e-06 = 0.4003 points; a 6-decimal Brier "
                            "converts with +/-0.2002 of rounding uncertainty. "
                            "The Brier is the raw quantity."),
        },
        "unit": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
                 "n_train": 1116277, "n_valid": 253507, "n_valid_R": 223497,
                 "n_valid_F": 30010, "arm_base": "C5_B3",
                 "stacker_rounds": 1400, "cells_refitted": False},
        "arms": {"A1": "81 columns (67 base + p6_* 6 + idea 1 + i3_anom_* 7)",
                 "L_no_p6": "75 columns (A1 minus the six p6_* columns)"},
        "identity_anchor_vs_exp049_seed42": anchor,
        "cells_missing": missing,
        "exp049_recheck": e049_recheck,
        "sign_census_L_no_p6_minus_A1": census,
        "variance_decomposition": decomp,
        "grid_cells": cells,
        "provenance": provenance,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    for r in ROUNDS:
        c = census[r]
        print(f"@{r}: {c['n_positive_L_no_p6_better']}/{c['n_cells']} cells "
              f"favour L_no_p6, mean {c['mean_delta_score']:+.5f}, "
              f"sd {c['sd_delta_score_ddof1']}, "
              f"range [{c['min_delta_score']:+.5f}, {c['max_delta_score']:+.5f}]")
    print("missing cells:", missing or "none")


if __name__ == "__main__":
    main()
