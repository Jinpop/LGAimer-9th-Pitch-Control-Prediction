#!/usr/bin/env python3
"""exp_087 — what the F-branch seed curve is worth AT THE FRAME, not at the branch.

The branch-level curve `seed_curve.json` answers "how much does averaging more F
seeds tighten the F BRANCH's own predictions".  That is not the number anyone
submits.  Two dampings sit between it and the frame:

  1. the branch mix — an F-branch move of delta changes the mixed stage-2
     probability by 0.5*delta on `game_type=='F'` rows and by 0 on R rows.  A
     VARIANCE therefore shrinks by 0.25, and only on 11.838% of rows.
  2. the outer blend — the partner arm carries w_v6 = 0.7, so our AA arm carries
     0.3, and the blend is in logit space.

Damping 1 is exact arithmetic on the frame this curve was measured on.
Damping 2 is NOT applied as a number here: the blend is in logit space and a
Brier delta does not carry through a logit-space weight by multiplication.  It
is named so that nobody reads the stage-2 figure as a leaderboard figure.

🔴 AND NONE OF IT IS A LEADERBOARD PREDICTION.  S1: fold_2024 (train<=2023) is
not the deployment model.  S2: the transfer ratio is axis-specific and has been
observed with both signs.  exp_077 measured a seed axis on another arm and found
only 0.2173 of the arm gain reached the blend, and warned in the ledger that
candidate Q's 35.99 does not move between arms.  This file produces an upper
bound on the stage-2 side and stops there.
"""
import json, sys

REF = 0.2498069
K = 100000.0 / REF
W_MIX_ON_F = 0.5
F_ROW_SHARE = 30010 / 253507
W_AA_OUTER = 0.3


def main(src, dst):
    c = json.load(open(src))
    F = c["segments"]["valid_F"]
    V_branch = F["V_across_seed_variance"]
    V_frame = W_MIX_ON_F ** 2 * V_branch * F_ROW_SHARE
    out = {
        "input": src,
        "branch_level_valid_F": {
            "V": V_branch, "b_points": K * V_branch,
            "score_3": F["score_curve"]["3"],
            "score_18": F["score_curve"]["18"],
            "gain_3_to_18_points": F["gain_3_to_18_points"],
            "gain_3_to_9_points": F["gain_3_to_9_points"],
            "gain_3_to_6_points": F["gain_3_to_6_points"],
        },
        "frame_level_after_the_0_5_branch_mix": {
            "damping": W_MIX_ON_F ** 2 * F_ROW_SHARE,
            "V_frame": V_frame,
            "b_points_frame": K * V_frame,
            "gain_3_to_18_points_frame": K * V_frame * (1.0 / 3 - 1.0 / 18),
            "gain_3_to_12_points_frame": K * V_frame * (1.0 / 3 - 1.0 / 12),
            "gain_3_to_9_points_frame": K * V_frame * (1.0 / 3 - 1.0 / 9),
            "gain_3_to_6_points_frame": K * V_frame * (1.0 / 3 - 1.0 / 6),
        },
        "still_upstream_of_the_leaderboard": {
            "outer_AA_share": W_AA_OUTER,
            "why_not_multiplied": "the outer blend is in logit space; a Brier delta does not pass through it by multiplication",
            "S1": "fold_2024 (<=2023) is not the deployed model",
            "S2": "transfer ratios are axis-specific and have come out with both signs",
            "exp_077": "a seed axis on the v6-CAT arm transferred 0.2173 of the arm gain to the blend, and the ledger warns Q's 35.99 does not move between arms",
        },
    }
    json.dump(out, open(dst, "w"), indent=1)
    b = out["branch_level_valid_F"]; f = out["frame_level_after_the_0_5_branch_mix"]
    print(f"branch valid_F: b = {b['b_points']:.4f} pts, 3->18 = +{b['gain_3_to_18_points']:.4f} pts")
    print(f"frame level   : b = {f['b_points_frame']:.4f} pts, 3->18 = +{f['gain_3_to_18_points_frame']:.4f} pts"
          f"  (3->9 +{f['gain_3_to_9_points_frame']:.4f}, 3->6 +{f['gain_3_to_6_points_frame']:.4f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
