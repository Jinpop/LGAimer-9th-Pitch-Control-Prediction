"""exp_011 probe -- validate every assumption the shrinkage rests on, BEFORE training.

Checks
  P1  rate * n is an integer  -> the numerator really is recoverable (EDA sec.0)
      This is the load-bearing assumption: if a rate's denominator is NOT the
      `n` column we paired it with, the recovered numerator is garbage.
  P2  denominator distribution per family (how much shrinkage room exists)
  P3  two independent estimates of the league success prior agree
      (direct target mean vs the per-entity pooled recovery) -> validates the
      pooled estimator that the five NON-target priors depend on
  P4  prev{1,3,5}-game denominator identifiability (is it recoverable at all?)
  P5  prior table for all three modes, written to priors.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (  # noqa: E402
    PREV_COLS,
    PRIOR_MODES,
    RATE_SPECS,
    TARGET,
    _safe_ratio,
    _window_totals,
    build_prior_table,
    save_priors,
)

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_011")
SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    print(f"loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)
    rep = {}

    # ---- P1: numerator recovery -------------------------------------------
    p1 = {}
    for name, rate_col, denom_col, _kind, _fam in RATE_SPECS:
        m = df[denom_col] > 0
        prod = (df.loc[m, rate_col] * df.loc[m, denom_col]).to_numpy()
        err = np.abs(prod - np.round(prod))
        p1[name] = {
            "denominator": denom_col,
            "n_rows": int(m.sum()),
            "max_abs_int_error": float(np.nanmax(err)),
            "frac_within_1e-6": float(np.mean(err < 1e-6)),
            "frac_within_1e-3": float(np.mean(err < 1e-3)),
            "frac_within_1e-2": float(np.mean(err < 1e-2)),
        }
        print(
            f"P1 {name:14s} ({denom_col}): max|err|={p1[name]['max_abs_int_error']:.3e} "
            f"<1e-6={p1[name]['frac_within_1e-6']:.4f} <1e-2={p1[name]['frac_within_1e-2']:.4f}",
            flush=True,
        )
    rep["P1_numerator_recovery"] = p1

    # ---- P2: denominator distributions ------------------------------------
    p2 = {}
    for denom_col in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        v = df[denom_col].to_numpy(dtype=float)
        p2[denom_col] = {
            "n_missing": int(np.isnan(v).sum()),
            "min": float(np.nanmin(v)),
            "q01": float(np.nanpercentile(v, 1)),
            "q05": float(np.nanpercentile(v, 5)),
            "q25": float(np.nanpercentile(v, 25)),
            "median": float(np.nanmedian(v)),
            "q75": float(np.nanpercentile(v, 75)),
            "q95": float(np.nanpercentile(v, 95)),
            "max": float(np.nanmax(v)),
            "frac_eq_0": float(np.nanmean(v == 0)),
            "frac_lt_100": float(np.nanmean(v < 100)),
            "frac_lt_500": float(np.nanmean(v < 500)),
            "frac_lt_1000": float(np.nanmean(v < 1000)),
        }
        print(
            f"P2 {denom_col}: median={p2[denom_col]['median']:.0f} "
            f"q05={p2[denom_col]['q05']:.0f} <100={p2[denom_col]['frac_lt_100']:.3f} "
            f"missing={p2[denom_col]['n_missing']}",
            flush=True,
        )
    rep["P2_denominator_dist"] = p2

    # ---- P3: prior estimator cross-check ----------------------------------
    p3 = {}
    for y in [2021, 2022, 2023, 2024]:
        win = df.loc[df["season"] <= y]
        tot = _window_totals(win)
        direct = float(win[TARGET].mean())
        pooled_p = _safe_ratio(*tot["p_success"])
        pooled_b = _safe_ratio(*tot["b_success"])
        p3[str(y)] = {
            "direct_target_mean": round(direct, 8),
            "pooled_via_pitcher": round(pooled_p, 8),
            "pooled_via_batter": round(pooled_b, 8),
            "abs_diff_pitcher": round(abs(direct - pooled_p), 8),
            "abs_diff_batter": round(abs(direct - pooled_b), 8),
        }
        print(
            f"P3 <= {y}: direct={direct:.6f} pooled_p={pooled_p:.6f} pooled_b={pooled_b:.6f}",
            flush=True,
        )
        assert abs(direct - pooled_p) < 2e-3, p3[str(y)]
        assert abs(direct - pooled_b) < 2e-3, p3[str(y)]
    rep["P3_prior_estimator_crosscheck"] = p3

    # ---- P4: prev-game denominator identifiability -------------------------
    BMAX = 200
    p4 = {}
    bs = np.arange(1, BMAX + 1, dtype=float)
    for col in PREV_COLS[:3]:
        v = df[col].dropna().to_numpy(dtype=float)
        samp = rng.choice(v, size=min(20000, v.size), replace=False)
        prod = np.outer(samp, bs)
        ok = np.abs(prod - np.round(prod)) < 1e-6
        first = np.where(ok.any(axis=1), ok.argmax(axis=1) + 1, -1)
        found = first > 0
        n_cand = ok.sum(axis=1)
        p4[col] = {
            "n_sampled": int(samp.size),
            "frac_identifiable_b_le_200": float(np.mean(found)),
            "median_smallest_b": float(np.median(first[found])) if found.any() else None,
            "frac_smallest_b_le_10": float(np.mean(first[found] <= 10)) if found.any() else None,
            "median_n_candidate_b": float(np.median(n_cand)),
            "note": "smallest-b is biased low for reducible fractions; see features.py",
        }
        print(
            f"P4 {col}: identifiable={p4[col]['frac_identifiable_b_le_200']:.3f} "
            f"median_b={p4[col]['median_smallest_b']} "
            f"frac_b<=10={p4[col]['frac_smallest_b_le_10']}",
            flush=True,
        )
    rep["P4_prev_game_denominator"] = p4

    # ---- P5: prior table ---------------------------------------------------
    seasons_out = sorted(int(s) for s in df["season"].unique()) + [2025]
    priors = build_prior_table(df, seasons_out)
    priors["meta"] = {
        "exp_id": "exp_011",
        "built_from": "train.csv (full)",
        "rule": "prior[mode][f][s] uses ONLY rows with season < s; "
                "s = first season falls back to that season itself",
        "modes": list(PRIOR_MODES),
        "seasons": [str(s) for s in seasons_out],
    }
    save_priors(priors, os.path.join(OUT_DIR, "priors.json"))
    rep["P5_prior_table"] = priors["prior"]
    rep["P5_prior_windows"] = priors["windows"]
    rep["P5_per_season_rate"] = priors["per_season_rate"]
    for mode in PRIOR_MODES:
        print(f"P5 prior[{mode}]:", flush=True)
        print(
            json.dumps(
                {k: {s: round(x, 5) for s, x in v.items()} for k, v in priors["prior"][mode].items()},
                indent=1,
            ),
            flush=True,
        )

    rep["season_base_rate"] = {
        f"{int(s)}_{g}": round(float(x), 6)
        for (s, g), x in df.groupby(["season", "game_type"])[TARGET].mean().items()
    }

    with open(os.path.join(OUT_DIR, "probe.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"DONE probe in {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
