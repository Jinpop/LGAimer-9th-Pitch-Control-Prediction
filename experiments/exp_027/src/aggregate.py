"""exp_027 aggregation — Stage A ranking, STOP_GATE, Stage B adoption, blend.

Reporting rule (config stages.A_screen.reporting):
  fold_2023 ABSOLUTE Brier is NEVER converted to an LB score.  The conversion
  reference 0.2498069 is the 2025-test scale; pinning it to another fold's
  absolute level would be a false comparison (same class of error as the old
  valid-R yardstick).  Only DELTAS are converted:
      delta_score = 100000 * delta_brier / 0.2498069
  fold_2024 absolute Brier (Stage B) may carry a converted score, and does.
"""

import argparse
import glob
import json
import os

import numpy as np

REF = 0.2498069                       # 2025-test scale, deltas only on fold_2023
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_027")
DIAG_ARMS = {"X4H", "C5N"}                   # not promotable
PROMOTE_THRESHOLD = 1.0e-05           # config STOP_GATE.promote
MAX_PER_FAMILY, MAX_TOTAL = 1, 4


def d_score(d_brier):
    """Delta Brier -> delta LB points.  Sign convention: a NEGATIVE delta Brier
    (better) yields a POSITIVE point gain."""
    v = -100000.0 * d_brier / REF
    return float(f"{v:.5f}".rstrip("0").rstrip(".")) if v == v else None


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "res", pat))):
        r = json.load(open(p))
        out[os.path.basename(p)[4:-5]] = r
    return out


def stage_a(res):
    rows = []
    l1 = res.get("L1_seed42_fold_2023")
    assert l1 is not None, "L1 control missing"
    base = l1["argmin_brier"]
    for k, r in res.items():
        if r["fold"] != "fold_2023" or r["stage"] != "A":
            continue
        d = r["argmin_brier"] - base
        rows.append({
            "arm": r["arm"], "family": r["family"], "role": r["role"],
            "argmin_iter": r["argmin_iter"],
            "brier_fold2023_at_argmin": r["argmin_brier"],
            "delta_vs_L1_brier": float(f"{d:.6g}"),
            "delta_vs_L1_score_points": d_score(d),
            "beats_L1": bool(d < 0),
            "beats_L1_by_threshold": bool(d <= -PROMOTE_THRESHOLD),
            "argmin_at_cap": r["argmin_at_cap"], "cap": r["cap"],
            "fit_min": r["fit_min"], "diagnostic_only": r["arm"] in DIAG_ARMS,
            "valid_R_brier": r["valid_R"]["brier_full"],
            "valid_F_brier": r["valid_F"]["brier_full"],
            "auc": r["valid_all"].get("auc"),
            "stage_b_iter": int(round(r["argmin_iter"] * 1.281969)),
        })
    rows.sort(key=lambda x: x["brier_fold2023_at_argmin"])
    return rows, base


def stop_gate(rows):
    invalid = [r["arm"] for r in rows if r["argmin_at_cap"]]
    winners = [r for r in rows if r["beats_L1"] and not r["diagnostic_only"]]
    elig = [r for r in rows if r["beats_L1_by_threshold"]
            and not r["diagnostic_only"] and not r["argmin_at_cap"]]
    promoted, per_fam = [], {}
    for r in elig:                                    # already sorted best-first
        if len(promoted) >= MAX_TOTAL:
            break
        if per_fam.get(r["family"], 0) >= MAX_PER_FAMILY:
            continue
        promoted.append(r["arm"])
        per_fam[r["family"]] = per_fam.get(r["family"], 0) + 1
    return {
        "arms_invalid_argmin_at_cap": invalid,
        "n_arms_beating_L1_any_margin": len(winners),
        "arms_beating_L1_any_margin": [r["arm"] for r in winners],
        "n_arms_beating_L1_by_1e-05": len(elig),
        "arms_beating_L1_by_1e-05": [r["arm"] for r in elig],
        "promote_rule": f"<= -{PROMOTE_THRESHOLD} Brier, max "
                        f"{MAX_PER_FAMILY}/family, max {MAX_TOTAL} total",
        "promoted": promoted,
        "STOP": len(winners) == 0,
        "verdict": ("STOP — no arm beat the L1 control; the base is at a "
                    "reasonable level within the range we can search"
                    if len(winners) == 0 else
                    f"PROCEED to Stage B with {promoted} + L1"),
    }


def attribution(rows):
    b = {r["arm"]: r["brier_fold2023_at_argmin"] for r in rows}

    def pair(a, c, what):
        if a not in b or c not in b:
            return None
        d = b[a] - b[c]
        return {"contrast": f"{a} - {c}", "what": what,
                "delta_brier": float(f"{d:.6g}"),
                "delta_score_points": d_score(d),
                "reading": f"{a} is {'better' if d < 0 else 'worse'} than {c}"}

    out = {
        "has_time_share": pair("C1", "C4", "ordered target statistics (has_time), "
                                           "measured WITH native ID categoricals "
                                           "— i.e. where ordered TS actually acts"),
        "has_time_share_at_winner": pair(
            "C5", "C5N", "has_time at the WINNING configuration (IDs numeric). "
                         "Here every cat feature has 1, 2 or 8 levels, all at or "
                         "below one_hot_max_size=10, so CatBoost one-hot encodes "
                         "them and computes no target statistics to order"),
        "native_id_share": pair("C1", "C5", "pitcher_id/batter_id as native "
                                            "CatBoost categoricals"),
        "pure_family_as_specified": pair("X4", "L1",
                                         "XGBoost vs LightGBM at 15 leaves — "
                                         "CONFOUNDED, see caveats"),
        "pure_family_matched": pair("X4H", "L1",
                                    "XGBoost vs LightGBM with leaf-size and "
                                    "sampling put on the same scale"),
        "lgbm_regularisation_axis_closed": {
            "L2_minus_L1": float(f"{b['L2'] - b['L1']:.6g}") if "L2" in b else None,
            "L3_minus_L1": float(f"{b['L3'] - b['L1']:.6g}") if "L3" in b else None,
            "L2_score": d_score(b["L2"] - b["L1"]) if "L2" in b else None,
            "L3_score": d_score(b["L3"] - b["L1"]) if "L3" in b else None,
            "closed": bool("L2" in b and "L3" in b
                           and b["L2"] >= b["L1"] and b["L3"] >= b["L1"]),
            "meaning": "closed=True means pushing regularisation past exp_010's "
                       "endpoint does NOT help, i.e. exp_010's grid was long "
                       "enough after all",
        },
    }
    out["caveats"] = [
        "X4 - L1 conflates three things: (a) the family, (b) XGBoost is not "
        "told the three string columns are categorical while LightGBM is "
        "(exp_021 D vs D_NOCAT_DIAG sizes that at 1.1e-05 Brier on this exact "
        "fold/seed), (c) XGBoost min_child_weight is a hessian sum, so 5000 is "
        "a ~20,000-row constraint vs LightGBM's 5000 rows, and X4 also lacks "
        "L1's feature_fraction 0.7 / bagging 0.8.  X4H removes (c).",
        "C1 - C4 and C1 - C5 each change exactly one thing and are clean.",
    ]
    return out


def stage_b(res):
    per, arms = {}, set()
    for k, r in res.items():
        if r["stage"] != "B":
            continue
        arms.add(r["arm"])
        per.setdefault(r["arm"], {})[r["seed"]] = r
    if not per or "L1" not in per:
        return None
    seeds = sorted(set().union(*[set(v) for v in per.values()]))
    l1 = np.array([per["L1"][s]["valid_all"]["brier_full"] for s in seeds
                   if s in per["L1"]], dtype=float)
    l1_std = float(np.std(l1, ddof=1)) if len(l1) > 1 else None
    thr = 3.0 * l1_std if l1_std else None
    out = {"seeds": seeds, "L1_seed_briers": [float(f"{v:.10g}") for v in l1],
           "L1_seed_std": l1_std, "adoption_threshold_abs_mean_delta": thr,
           "rule": "3 seeds same sign AND |mean delta| >= 3 x L1 seed std",
           "arms": {}}
    for a in sorted(arms):
        b = np.array([per[a][s]["valid_all"]["brier_full"] for s in seeds
                      if s in per[a]], dtype=float)
        if len(b) != len(l1):
            out["arms"][a] = {"incomplete": True,
                              "seeds_present": sorted(per[a])}
            continue
        d = b - l1
        mean_d = float(d.mean())
        same = bool(np.all(d < 0) or np.all(d > 0))
        out["arms"][a] = {
            "seed_briers": [float(f"{v:.10g}") for v in b],
            "mean_brier": float(f"{b.mean():.10g}"),
            "mean_score": round(100000.0 * (1.0 - b.mean() / REF), 5),
            "seed_std": float(np.std(b, ddof=1)),
            "paired_delta_per_seed": [float(f"{v:.6g}") for v in d],
            "paired_std": float(np.std(d, ddof=1)),
            "mean_delta_vs_L1": float(f"{mean_d:.6g}"),
            "mean_delta_score_points": d_score(mean_d),
            "signs_agree": same,
            "passes_magnitude": bool(thr is not None and abs(mean_d) >= thr),
            "ADOPTED": bool(a != "L1" and same and mean_d < 0
                            and thr is not None and abs(mean_d) >= thr),
        }
    return out


def _logit(p, eps=1e-7):
    q = np.clip(p, eps, 1 - eps)
    return np.log(q / (1 - q))


def blend(res, promoted, stage_b_iters):
    """Logit-average blend of L1 + the promoted arms.

    Weights are fit on fold_2023 ONLY; fold_2024 is read, never fit.

    The weight search is an EXHAUSTIVE GRID on the simplex, not a gradient
    optimiser.  scipy SLSQP was tried first and returned its own starting point
    unchanged: over the whole weight range the objective moves by ~4e-05 Brier,
    which is below its default convergence tolerance, so "it did not move" was
    an optimiser artefact rather than an optimum.  A grid cannot fail that way.
    """
    names = ["L1"] + [a for a in promoted if a != "L1"]
    n = len(names)
    P23 = []
    for a in names:
        f = os.path.join(OUT_DIR, "pred", f"pred_{a}_seed42_fold_2023.npy")
        if not os.path.exists(f):
            return {"skipped": f"missing {f}"}
        P23.append(np.load(f))
    P23 = np.vstack(P23)
    y23 = np.load(os.path.join(OUT_DIR, "y_fold_2023.npy"))
    L23 = _logit(P23)

    def brier(w, Lm, y):
        return float(np.mean((1.0 / (1.0 + np.exp(-(w[:, None] * Lm).sum(0))) - y) ** 2))

    if n == 2:                       # 1-D simplex -> exact scan, then refine
        ws = np.arange(0.0, 1.0 + 1e-12, 0.001)
        vals = [brier(np.array([1 - t, t]), L23, y23) for t in ws]
        j = int(np.argmin(vals))
        w_best = np.array([1 - ws[j], ws[j]])
        best_val = vals[j]
    else:
        from itertools import product
        step = 0.02
        grid = [g for g in product(np.arange(0, 1 + 1e-12, step), repeat=n)
                if abs(sum(g) - 1.0) < 1e-9]
        w_best, best_val = None, np.inf
        for g in grid:
            v = brier(np.array(g), L23, y23)
            if v < best_val:
                w_best, best_val = np.array(g), v

    w_eq = np.full(n, 1.0 / n)
    singles23 = {a: float(np.mean((P23[i] - y23) ** 2)) for i, a in enumerate(names)}
    best_single_name = min(singles23, key=singles23.get)
    out = {
        "members": names,
        "weight_search": "exhaustive grid on the simplex (SLSQP returned its "
                         "starting point and was discarded — see docstring)",
        "weights_fit_on": "fold_2023 only; fold_2024 read-only",
        "fold_2023": {
            "singles": {a: float(f"{v:.10g}") for a, v in singles23.items()},
            "best_single": best_single_name,
            "best_single_brier": float(f"{singles23[best_single_name]:.10g}"),
            "equal_weight_brier": float(f"{brier(w_eq, L23, y23):.10g}"),
            "fitted_weights": [float(f"{v:.6g}") for v in w_best],
            "fitted_brier": float(f"{best_val:.10g}"),
            "fitted_gain_vs_best_single_brier": float(
                f"{best_val - singles23[best_single_name]:.6g}"),
            "fitted_gain_vs_best_single_points": d_score(
                best_val - singles23[best_single_name]),
            "beats_best_single": bool(best_val < singles23[best_single_name]),
        },
    }
    # ---- fold_2024, weights frozen from fold_2023 -------------------------- #
    y24 = np.load(os.path.join(OUT_DIR, "y_fold_2024.npy"))
    seeds, per_seed = [42, 7, 2024], {}
    for s in seeds:
        P, ok = [], True
        for a in names:
            k = stage_b_iters.get(a, {}).get("fold_2024_iter")
            f = os.path.join(OUT_DIR, "pred",
                             f"pred_{a}_seed{s}_fold_2024_it{k}.npy")
            if not os.path.exists(f):
                ok = False
                break
            P.append(np.load(f))
        if not ok:
            continue
        P = np.vstack(P)
        L24 = _logit(P)
        sing = {a: float(np.mean((P[i] - y24) ** 2)) for i, a in enumerate(names)}
        per_seed[s] = {
            "singles": {a: float(f"{v:.10g}") for a, v in sing.items()},
            "equal_weight_brier": float(f"{brier(w_eq, L24, y24):.10g}"),
            "fitted_weight_brier": float(f"{brier(w_best, L24, y24):.10g}"),
        }
    if per_seed:
        bs = float(np.mean([per_seed[s]["singles"][best_single_name]
                            for s in per_seed]))
        bl = float(np.mean([per_seed[s]["fitted_weight_brier"] for s in per_seed]))
        out["fold_2024"] = {
            "per_seed": per_seed,
            "mean_best_single_brier": float(f"{bs:.10g}"),
            "mean_best_single_score": round(100000.0 * (1.0 - bs / REF), 5),
            "mean_blend_brier": float(f"{bl:.10g}"),
            "mean_blend_score": round(100000.0 * (1.0 - bl / REF), 5),
            "blend_minus_best_single_brier": float(f"{bl - bs:.6g}"),
            "blend_minus_best_single_points": d_score(bl - bs),
            "beats_best_single": bool(bl < bs),
            "note": "weights came from fold_2023; this fold only reads them",
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", nargs="*", default=None)
    args = ap.parse_args()
    res = load("*.json")
    rows, base = stage_a(res)
    m = {
        "exp_id": "exp_027",
        "reporting_rule": {
            "fold_2023": "ABSOLUTE Brier only; deltas converted via "
                         f"100000 * dBrier / {REF}",
            "fold_2024": "absolute Brier + converted score both reported",
            "why": "the 0.2498069 reference is the 2025-test scale",
        },
        "gate": json.load(open(os.path.join(OUT_DIR, "gate.json")))
        if os.path.exists(os.path.join(OUT_DIR, "gate.json")) else None,
        "L1_control_fold2023_brier": base,
        "stage_A_ranking": rows,
        "STOP_GATE": stop_gate(rows),
        "attribution": attribution(rows),
    }
    sb = stage_b(res)
    if sb:
        m["stage_B"] = sb
    if args.blend:
        sbi = os.path.join(OUT_DIR, "stageB_iters.json")
        m["stage_C_blend"] = blend(res, args.blend,
                                   json.load(open(sbi)) if os.path.exists(sbi) else {})
    m["fit_time_by_family_min"] = {
        f: {"n": int(sum(1 for r in rows if r["family"] == f)),
            "mean_fit_min": round(float(np.mean(
                [r["fit_min"] for r in rows if r["family"] == f])), 3),
            "mean_sec_per_round": round(float(np.mean(
                [res[f"{r['arm']}_seed42_fold_2023"]["fit_sec_per_round"]
                 for r in rows if r["family"] == f])), 5)}
        for f in ("lightgbm", "xgboost", "catboost")
        if any(r["family"] == f for r in rows)}
    p = os.path.join(OUT_DIR, "metrics.json")
    json.dump(m, open(p, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in m.items() if k != "gate"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
