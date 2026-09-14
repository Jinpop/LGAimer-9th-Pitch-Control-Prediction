"""exp_052 stage 0 — assemble metrics.json from the G0a and G0b artefacts.

CONVERSION RULE (CLAUDE.md): score = 100000 * (1 - brier / 0.2498069).
🔴 fold_2023 ABSOLUTE Briers are NOT converted -- the reference 0.2498069 is a
2025 base-rate quantity and has no referent on the 2023 plane.  Only the
DIFFERENCES between arms measured on the same plane are converted, at most 5
decimals.  Conversion sensitivity: 1e-6 Brier = 0.4003 score points.

PRE-REGISTERED GATE: 상한_81 = brier(C) - brier(D) >= 40 points.
"""
import json
import os

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
REF_BRIER = 0.2498069
GATE_POINTS = 40.0
TM_PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]
TM_LAB = ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]
F14_P6 = ["p6_curve", "p6_cutter", "p6_fastball", "p6_offspeed", "p6_sinker",
          "p6_slider"]


def pts(brier_reduction):
    """Convert a Brier REDUCTION (positive = better) on a fixed plane into
    leaderboard score points (positive = higher score)."""
    return round(brier_reduction / REF_BRIER * 100000.0, 5)


def main():
    res = {a: json.load(open(os.path.join(BASE, "res", f"g0b_arm{a}.json")))
           for a in "ABCD"}
    g0a = json.load(open(os.path.join(BASE, "res", "g0a_match.json")))
    prep = json.load(open(os.path.join(BASE, "res", "g0b_prep.json")))

    b = {a: res[a]["brier_at_argmin"] for a in "ABCD"}
    up67 = b["A"] - b["B"]
    up81 = b["C"] - b["D"]
    c_minus_a = b["A"] - b["C"]

    out = {
        "exp_id": "exp_052", "stage": 0,
        "note": ("stage 1 NOT run. fold_2024 NOT opened. fold_2023 absolute "
                 "Brier is NOT score-converted (ref is a 2025 quantity); only "
                 "same-plane differences are."),
        "G0a": {
            "unique_pct": g0a["gate"]["unique_pct"],
            "multi_ambiguous_pct": g0a["gate"]["multi_ambiguous_pct"],
            "unmatched_pct": g0a["gate"]["unmatched_pct"],
            "threshold_pct": g0a["gate"]["threshold_pct"],
            "pass": g0a["gate"]["pass"],
            "coverage_by_season": g0a["coverage_by_season"],
            "coverage_by_game_type": g0a["coverage_by_game_type"],
            "selection_bias": g0a["selection_bias"],
            "pitcher_map_vs_exp002": g0a["S4_vs_exp002"],
            "hand_agreement": g0a["S2_hand_agreement_on_matched"],
        },
        "G0b": {
            "fold": res["A"]["fold"], "seed": 42, "cap": res["A"]["cap"],
            "arms": {a: {"n_cols": res[a]["n_cols"],
                         "argmin_iter": res[a]["argmin_iter"],
                         "argmin_at_cap_INVALID": res[a]["argmin_at_cap_INVALID"],
                         "brier_at_argmin": res[a]["brier_at_argmin"],
                         "brier_at_1400": res[a]["brier_at_1400"]}
                     for a in "ABCD"},
            "상한_81_C_minus_D": {"brier": up81, "score_points": pts(up81),
                                  "gate_points": GATE_POINTS,
                                  "pass": bool(pts(up81) >= GATE_POINTS)},
            "상한_67_A_minus_B": {"brier": up67, "score_points": pts(up67)},
            "p6_carry_A_minus_C": {"brier": c_minus_a,
                                   "score_points": pts(c_minus_a)},
            "any_argmin_at_cap": any(res[a]["argmin_at_cap_INVALID"]
                                     for a in "ABCD"),
        },
        "gain_shares": {},
        "prep": {k: prep[k] for k in
                 ("n_train", "n_valid", "tm_has_tm_pct_train",
                  "tm_has_tm_pct_valid", "col_blocks", "arm_sizes",
                  "ins_nan_pct_by_season", "oof_nan_pct_by_season",
                  "tm_label_levels")},
        "train_time_min": {a: res[a]["total_min"] for a in "ABCD"},
    }

    # gain: which teacher-only block makes the bound, physics or the labels?
    for a in ("B", "D"):
        g = res[a]["gain"]
        tot = sum(g.values())
        phys = sum(g[c] for c in TM_PHYS)
        lab = sum(g[c] for c in TM_LAB)
        htm = g.get("has_tm", 0.0)
        rank = sorted(g, key=g.get, reverse=True)
        out["gain_shares"][a] = {
            "total": round(tot, 6),
            "tm_physics_8_pct": round(100 * phys / tot, 4),
            "tm_pitchtype_labels_3_pct": round(100 * lab / tot, 4),
            "has_tm_pct": round(100 * htm / tot, 4),
            "physics_over_labels_ratio": round(phys / lab, 4) if lab else None,
            "top15": [(c, round(g[c], 4), rank.index(c) + 1) for c in rank[:15]],
            "tm_col_ranks": {c: rank.index(c) + 1
                             for c in TM_PHYS + TM_LAB + ["has_tm"]},
        }
    for a in ("C", "D"):
        g = res[a]["gain"]
        tot = sum(g.values())
        out["gain_shares"].setdefault(a, {})["p6_6_pct"] = round(
            100 * sum(g[c] for c in F14_P6) / tot, 4)

    with open(os.path.join(BASE, "res", "stage0_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out["G0b"], indent=1, ensure_ascii=False))
    print(json.dumps(out["gain_shares"], indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
