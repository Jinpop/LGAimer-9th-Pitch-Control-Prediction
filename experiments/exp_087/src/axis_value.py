#!/usr/bin/env python3
"""exp_087 — what the F-branch seed axis is worth, ANCHORED ON A LEADERBOARD POINT.

The R-branch seed axis is the one axis in this project whose seed-averaging law
was pinned by an actual submission.  Candidate AV took the R branch from 9 to 18
seeds and the leaderboard moved +0.8716523199.  Under the averaging law the gain
from n to 2n is sigma^2/(2n), so

    sigma^2_R(frame, LB points) = 18 * 0.8716523199 = 15.68974

and the ledger closed the axis on it (18 -> inf is only another +0.87165).

This file ports that anchor to the F branch, and the port is a RATIO, not an
assumption.  Both branches have fold_2024 stage-2 prediction vectors at the same
three seeds on the same 253,507 rows:

    R branch : exp_048  pred_A1_u2024_seed{42,7,2024}_at1400.npy   (the DEPLOYED
               c67 arm: 47 base + 4 OOF + 16 `ins`, stacker at the shipped 1400)
               with exp_030's 51-column C5B3 vectors kept as a robustness check
    F branch : exp_037  pred_L_no_sit_u2024_seed{42,7,2024}_st2.npy (63 columns)

so the across-seed variance of each is measured the same way and the ratio
V_F / V_R carries no fold_2024-to-deployment scale factor with it — whatever
that factor is, it divides out.

What does NOT divide out is where each branch lands in the mix.  At inference

    R rows  (88.162%) : p = 1.0 * p_R
    F rows  (11.838%) : p = 0.5 * p_R + 0.5 * p_F

so a branch's variance reaches the frame multiplied by its squared weight,
averaged over rows:

    reach_R = 0.88162 * 1.0^2 + 0.11838 * 0.5^2
    reach_F = 0.88162 * 0.0^2 + 0.11838 * 0.5^2

🔴 WHAT THIS IS NOT.  It is not a leaderboard prediction and the downside is
symmetric (S5).  It assumes the averaging law, it assumes AV's anchor is still
the right scale for a package whose outer blend has moved since 2026-08-17, and
S2 says transfer ratios differ by axis and have come out with both signs.  What
it IS: the only estimate available that is anchored on a measured LB point
instead of on the honest surface, which S78 measured overshooting by 54x.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
REF = 0.2498069
K = 100000.0 / REF
AV_GAIN_9_TO_18 = 0.8716523199          # LB, candidate AV vs AR
SEEDS = [42, 7, 2024]
F_VEC = f"{H}/experiments/exp_037/pred/pred_L_no_sit_u2024_seed{{s}}_st2.npy"
R_VEC = f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed{{s}}_at1400.npy"
R_VEC_ALT = f"{H}/experiments/exp_030/pred/pred_C5B3_u2024_seed{{s}}_st2.npy"


def load(tpl):
    return np.stack([np.load(tpl.format(s=s)).astype(np.float64) for s in SEEDS])


def main(out_path):
    df = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig",
                     usecols=["season", "game_type", "control_success"])
    va = df["season"].to_numpy() == 2024
    y = df["control_success"].to_numpy(np.float64)[va]
    isF = df["game_type"].to_numpy()[va] == "F"
    del df
    n = int(va.sum()); nF = int(isF.sum())
    fF = nF / n

    P_F, P_R = load(F_VEC), load(R_VEC)
    P_R2 = load(R_VEC_ALT)
    assert P_F.shape == P_R.shape == P_R2.shape == (3, n), (P_F.shape, P_R.shape, n)

    # per-row across-seed variance (unbiased), averaged over rows
    V_F = float(np.mean(P_F.var(axis=0, ddof=1)))
    V_R = float(np.mean(P_R.var(axis=0, ddof=1)))
    V_F_onF = float(np.mean(P_F[:, isF].var(axis=0, ddof=1)))
    V_R_onF = float(np.mean(P_R[:, isF].var(axis=0, ddof=1)))
    V_R_onR = float(np.mean(P_R[:, ~isF].var(axis=0, ddof=1)))

    # the frame-level variance each branch actually contributes
    reach_R = (1 - fF) * V_R_onR * 1.0 ** 2 + fF * V_R_onF * 0.5 ** 2
    reach_F = fF * V_F_onF * 0.5 ** 2
    ratio = reach_F / reach_R

    # robustness: the same ratio computed against exp_030's 51-column R vectors
    V_R2_onF = float(np.mean(P_R2[:, isF].var(axis=0, ddof=1)))
    V_R2_onR = float(np.mean(P_R2[:, ~isF].var(axis=0, ddof=1)))
    reach_R2 = (1 - fF) * V_R2_onR + fF * V_R2_onF * 0.25
    ratio_alt = reach_F / reach_R2

    sigma2_R_LB = 18.0 * AV_GAIN_9_TO_18
    sigma2_F_LB = sigma2_R_LB * ratio
    sigma2_F_LB_alt = sigma2_R_LB * ratio_alt

    gains = {f"3_to_{m}": sigma2_F_LB * (1.0 / 3 - 1.0 / m)
             for m in (6, 9, 12, 15, 18)}
    gains["3_to_inf"] = sigma2_F_LB / 3.0

    # the same thing on the local fold_2024 surface, for contrast only
    local = {}
    for name, m, V in (("valid_all", np.ones(n, bool), None),
                       ("valid_F", isF, None)):
        Pm, ym = P_F[:, m], y[m]
        b3 = float(np.mean((Pm.mean(axis=0) - ym) ** 2))
        Vm = float(np.mean(Pm.var(axis=0, ddof=1)))
        local[name] = {"n": int(m.sum()), "brier_avg3": b3,
                       "V": Vm, "b_points": K * Vm,
                       "gain_3_to_18_points": K * Vm * (1.0 / 3 - 1.0 / 18)}

    out = {
      "anchor": {"candidate": "AV", "what": "R branch seeds 9 -> 18",
                 "LB_gain": AV_GAIN_9_TO_18,
                 "sigma2_R_frame_LB_points": sigma2_R_LB,
                 "ledger_note": "CV_LB.md closed this axis: 18 -> inf is only another +0.87165, under the 3-point threshold"},
      "fold_2024_variances": {
        "n_valid": n, "n_valid_F": nF, "F_row_share": fF,
        "V_F_branch_all_rows": V_F, "V_F_branch_on_F_rows": V_F_onF,
        "V_R_branch_all_rows": V_R, "V_R_branch_on_F_rows": V_R_onF,
        "V_R_branch_on_R_rows": V_R_onR,
        "V_F_over_V_R_on_F_rows": V_F_onF / V_R_onF},
      "frame_reach": {"reach_R": reach_R, "reach_F": reach_F,
                      "ratio_F_over_R": ratio,
                      "reach_R_alt_exp030_51col": reach_R2,
                      "ratio_F_over_R_alt": ratio_alt,
                      "sigma2_F_frame_LB_points_alt": sigma2_F_LB_alt,
                      "why": "branch variance times its squared mix weight, averaged over rows"},
      "PORTED_LB_ESTIMATE": {
        "sigma2_F_frame_LB_points": sigma2_F_LB,
        "gains_LB_points": gains,
        "🔴": "not a prediction; S5 the downside is symmetric, S2 transfer ratios differ by axis"},
      "local_fold_2024_surface_for_contrast_only": local,
    }
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(f"V_F/V_R on F rows = {V_F_onF / V_R_onF:.4f}")
    print(f"frame reach ratio F/R = {ratio:.6f}")
    print(f"ported sigma^2_F = {sigma2_F_LB:.5f} LB points "
          f"(robustness, exp_030 51-col R: {sigma2_F_LB_alt:.5f})")
    for k, v in gains.items():
        print(f"  F seeds {k}: {v:+.5f} LB points")
    for k, v in local.items():
        print(f"  local {k}: b = {v['b_points']:.4f} pts, 3->18 = {v['gain_3_to_18_points']:+.4f} pts")


if __name__ == "__main__":
    main(sys.argv[1])
