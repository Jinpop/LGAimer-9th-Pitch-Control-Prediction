"""exp_037 — the arm registry.  One place where "which columns" is defined.

GROUPS are the team-mate's four blocks, in the team-mate's own order, so that
our ladder deltas sit on the same ordinal axis as theirs (+53.6 / +17.7 /
+43.5 / +22.2 points, measured by them on fold_2024 with a flat LGBM+CatBoost
blend; ours are measured on fold_2023 inside the C5_B3 stack, so ONLY the
deltas are comparable and never the levels).

THE DERIVATION CHAIN, WHICH THE LOO ARMS HAVE TO RESPECT
--------------------------------------------------------
`p6_*` are the six class probabilities P of the pitch-type model.  Three other
blocks are functions of that same P:

    idea_matchup_pitch  = COMBO[(pitcher_hand, batter_hand, argmax P)]
    i3_anom_{type}      = P / clip(count-wise mean P, 1e-3)
    i3_anom_l1          = sum |P - count-wise mean P|

and two are not:

    idea_cnt_te         = target encoding of (balls, strikes)
    i3_te48             = target encoding of (p_hand, b_hand, balls, strikes)

So deleting the six `p6_*` columns does NOT delete the pitch-type signal from
the design matrix — seven P-derived columns survive.  `L_no_p6` therefore
measures the marginal value of the RAW probabilities on top of their own
derivatives, which is a real quantity but not "what happens without p6".
`L_no_p6ALL` is the arm that answers that second question: it removes every
column computed from P and keeps only the two target encodings and `sit_*`.
Both are run and both are reported, because reporting only the first would
invite exactly the misreading this comment exists to prevent.
"""
TYPES = ["curve", "cutter", "fastball", "offspeed", "sinker", "slider"]

GROUPS = {
    "p6": [f"p6_{t}" for t in TYPES],
    "sit": ["sit_cnt", "sit_cnt_dev", "sit_run", "sit_run_dev",
            "sit_sc", "sit_sc_dev"],
    "idea": ["idea_matchup_pitch", "idea_cnt_te"],
    "i3": [f"i3_anom_{t}" for t in TYPES] + ["i3_anom_l1", "i3_te48"],
}
ALL_COLS = GROUPS["p6"] + GROUPS["sit"] + GROUPS["idea"] + GROUPS["i3"]
P_DERIVED = (GROUPS["p6"] + ["idea_matchup_pitch"]
             + [f"i3_anom_{t}" for t in TYPES] + ["i3_anom_l1"])

_LADDER = ["p6", "sit", "idea", "i3"]


def _g(names):
    out = []
    for n in names:
        out += GROUPS[n]
    return out


ARMS = {
    # ---- B, the ladder (team-mate's stacking order) ----------------------- #
    "R0": {"cols": [], "stage": "B",
           "role": "CONTROL — our 47 provided columns, C5_B3 unchanged"},
    "R1": {"cols": _g(["p6"]), "stage": "B", "adds": "p6"},
    "R2": {"cols": _g(["p6", "sit"]), "stage": "B", "adds": "sit"},
    "R3": {"cols": _g(["p6", "sit", "idea"]), "stage": "B", "adds": "idea"},
    "R4": {"cols": _g(_LADDER), "stage": "B", "adds": "i3"},
    # ---- C, leave-one-out from R4 ----------------------------------------- #
    "L_no_p6": {"cols": _g(["sit", "idea", "i3"]), "stage": "C",
                "drops": "p6 (raw probabilities only; 7 P-derived cols remain)"},
    "L_no_sit": {"cols": _g(["p6", "idea", "i3"]), "stage": "C",
                 "drops": "sit"},
    "L_no_idea": {"cols": _g(["p6", "sit", "i3"]), "stage": "C",
                  "drops": "idea"},
    "L_no_i3": {"cols": _g(["p6", "sit", "idea"]), "stage": "C",
                "drops": "i3"},
    "L_no_p6ALL": {"cols": [c for c in ALL_COLS if c not in P_DERIVED],
                   "stage": "C",
                   "drops": "every column computed from P (p6_*, "
                            "idea_matchup_pitch, i3_anom_*) — the honest "
                            "'without the pitch model' arm"},
}

# R3 and L_no_i3 are the SAME column set by construction (R4 minus i3 == R3).
# It is kept as a separate name so the ladder and the LOO each read cleanly;
# the driver skips the duplicate fit and the report says so.
DUPLICATE_OF = {"L_no_i3": "R3"}

N_BASE = 47
EXPECTED_N = {"R0": 47, "R1": 53, "R2": 59, "R3": 61, "R4": 69,
              "L_no_p6": 63, "L_no_sit": 63, "L_no_idea": 67,
              "L_no_i3": 61, "L_no_p6ALL": 55}

for _a, _s in ARMS.items():
    assert len(_s["cols"]) == len(set(_s["cols"])), _a
    assert set(_s["cols"]) <= set(ALL_COLS), _a
    assert N_BASE + len(_s["cols"]) == EXPECTED_N[_a], (_a, len(_s["cols"]))
assert ARMS["L_no_i3"]["cols"] == ARMS["R3"]["cols"]
