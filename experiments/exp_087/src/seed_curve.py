#!/usr/bin/env python3
"""exp_087 부가 측정 — the F branch's seed-count curve, derived not re-fitted.

WHY IT IS DERIVED
-----------------
The 15 fits this experiment runs are DEPLOYMENT fits (season <= 2024).  They
cannot be scored on 2024 — they trained on it.  A measured `valid_F` curve at
n = 3..18 would need FIFTEEN MORE fold_2024 fits of the 63-column arm, roughly
another six hours.  What exists is exp_037's three fold_2024 seed vectors for
this exact arm (`L_no_sit`, unit 2024, seeds 42 / 7 / 2024).

Three vectors are enough, because the law the question asks about is itself an
exchangeability law.  Write seed i's prediction on row r as

    p_ir = mu_r + eps_ir ,   E[eps] = 0 ,  Var(eps_ir) = v_r  across seeds.

The n-seed average has variance v_r/n, so

    E[Brier_n] = mean_r (mu_r - y_r)^2  +  V/n ,      V = mean_r v_r

i.e. exactly the `score(n) = a - b/n` shape, with b = K * V and
K = 100000 / 0.2498069.  V is estimated from the three vectors with the
unbiased across-seed sample variance (ddof=1), so `a` follows from the measured
3-seed Brier: a_brier = Brier_3 - V/3.

⚠️ WHAT THIS IS NOT.  It is not a measurement of eighteen seeds.  It assumes the
seeds are exchangeable and that v_r is estimated well by three draws (the
sampling error of a 2-degree-of-freedom variance is large: the 95% interval for
V is roughly [0.42, 5.4] times the point estimate).  The SHAPE is exact under
the model; the SCALE carries that interval.  It is reported as a derivation.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
E37 = f"{H}/experiments/exp_037"
ARM = "L_no_sit"
SEEDS = [42, 7, 2024]
REF_BRIER = 0.2498069                 # 대회정보.md 의 baseline
K = 100000.0 / REF_BRIER


def points(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def main():
    df = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig",
                     usecols=["season", "game_type", "control_success"])
    va = (df["season"].to_numpy() == 2024)
    y = df["control_success"].to_numpy(dtype=np.float64)[va]
    gt = df["game_type"].to_numpy()[va]
    del df
    n = int(va.sum())
    P = np.stack([np.load(f"{E37}/pred/pred_{ARM}_u2024_seed{s}_st2.npy")
                  for s in SEEDS]).astype(np.float64)
    assert P.shape == (3, n), (P.shape, n)
    assert np.isfinite(y).all()

    segs = {"valid_all": np.ones(n, dtype=bool),
            "valid_F": (gt == "F"), "valid_R": (gt == "R")}
    out = {"arm": ARM, "unit": 2024, "seeds_available": SEEDS,
           "n_valid": n, "ref_brier": REF_BRIER,
           "method": "exchangeable-seed decomposition; see module docstring",
           "segments": {}}
    for name, m in segs.items():
        Pm, ym = P[:, m], y[m]
        # per-seed Brier, the 3-seed average's Brier, and the across-seed variance
        b_single = [float(np.mean((Pm[i] - ym) ** 2)) for i in range(3)]
        pbar = Pm.mean(axis=0)
        b3 = float(np.mean((pbar - ym) ** 2))
        V = float(np.mean(Pm.var(axis=0, ddof=1)))       # unbiased, per row
        b_inf = b3 - V / 3.0
        curve = {str(k): b_inf + V / k for k in (1, 3, 6, 9, 12, 15, 18)}
        out["segments"][name] = {
            "n_rows": int(m.sum()),
            "brier_single_seed": b_single,
            "brier_single_seed_mean": float(np.mean(b_single)),
            "brier_avg3_measured": b3,
            "brier_avg3_from_law": b_inf + V / 3.0,
            "V_across_seed_variance": V,
            "b_points_per_1_over_n": K * V,
            "a_points_asymptote": points(b_inf),
            "brier_curve": curve,
            "score_curve": {k: points(v) for k, v in curve.items()},
            "gain_3_to_18_points": points(curve["18"]) - points(curve["3"]),
            "gain_3_to_9_points": points(curve["9"]) - points(curve["3"]),
            "gain_3_to_6_points": points(curve["6"]) - points(curve["3"]),
        }
        print(f"{name:<10} n={int(m.sum()):>7}  b3={b3:.8f}  V={V:.6e}  "
              f"b={K*V:.4f} pts  3->18 = +{out['segments'][name]['gain_3_to_18_points']:.4f} pts")
    # cross-check against exp_037's recorded seed-42 number
    r = json.load(open(f"{E37}/res/res_{ARM}_u2024_seed42_st2.json"))
    out["crosscheck_exp037_seed42_brier_full"] = r["brier_full"]
    out["crosscheck_recomputed_seed42_brier_all"] = out["segments"]["valid_all"]["brier_single_seed"][0]
    out["crosscheck_abs_diff"] = abs(r["brier_full"]
                                     - out["segments"]["valid_all"]["brier_single_seed"][0])
    print("crosscheck |Δ| vs exp_037 recorded seed-42 Brier =",
          out["crosscheck_abs_diff"])
    json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
