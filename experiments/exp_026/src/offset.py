"""exp_019 — conditional-deviation offsets (stage 2).

WHAT `d` IS
-----------
exp_018 gave the tree a shrunk `(pitcher, batter_hand)` rate.  Here we do not
feed the tree anything; we build the *deviation*

    d = shrunk(pitcher, hand) rate  -  that pitcher's UNCONDITIONAL rate

which is exactly the part of the conditional lookup that is not already in the
47 provided columns, and then add `beta * d` to the model's probability.

`d == 0` is not a sentinel fill.  It is the value the estimator itself produces
when there is nothing to say:

  * cell empty, pitcher seen  ->  (0 + k*overall)/(0 + k) = overall  ->  d = 0
  * pitcher never seen        ->  overall is NaN, no correction is defined,
                                  and we apply none, i.e. p' = p

Both mean "do not move this row".  A missingness flag would be a *feature*
decision; this is a *no-op* decision, and the two are different things.

AS-OF
-----
The tables are the ones `features_platoon.build_season_forward_chained` builds:
a season-s row only ever sees seasons <= s-1.  This module recomputes the
pitcher-level denominators (which `fp._apply` does not return) from `fp._agg`
with the identical mask, and asserts that the shrunk rate it reconstructs is
bit-comparable to the column `fp` wrote.  `features_platoon.py` is byte-
identical to exp_018 (sha 68d63063...).

SLOPE FITTING
-------------
`fit_slope` is a no-intercept least squares of the residual (y - p) on d.  No
intercept on purpose: an intercept would be a global probability shift, i.e.
the exp_005 season-drift correction, which is a different mechanism and was
measured to *hurt* on this base (+3.0e-5).  The Brier change from p -> p + beta*d
is  -2*beta*E[d r] + beta^2*E[d^2], minimised at beta = E[d r]/E[d^2], which is
that same no-intercept slope.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import features_platoon as fp

# deviation name -> the fp column it is the deviation of
DEV_SPEC = {"d_s": "pit_success_rate_vs_hand",
            "d_r": "pit_reverse_rate_vs_hand"}

EPS = 1e-6


def build_deviations(df, ok, succ_season, f_excl, verbose=True):
    """Return (dev dict of arrays, hit mask, src_max_season, diagnostics).

    `df` must already carry the columns written by
    `fp.build_season_forward_chained` — they are used as a cross-check only.
    """
    season = df["season"].to_numpy()
    pit = df[fp.PITCHER].to_numpy()
    hand = df[fp.HAND].to_numpy()
    nums = fp.numerators(df)
    base_ok = ok & ~f_excl
    seasons = sorted(set(season.tolist()))

    n = len(df)
    dev = {k: np.zeros(n, dtype=np.float64) for k in DEV_SPEC}
    hit = np.zeros(n, dtype=bool)
    src = np.full(n, np.nan)
    diag = {"per_season": {}, "cross_check_max_abs_dev_vs_features_platoon": 0.0}

    for s in seasons[1:]:
        max_src = s - 1
        m = base_ok & (season <= max_src) & (succ_season <= max_src)
        cell, pitc = fp._agg(pit, hand, nums, m)

        rows = np.flatnonzero(season == s)
        mi = pd.MultiIndex.from_arrays([pit[rows], hand[rows]])
        c = cell.reindex(mi)
        p = pitc.reindex(pd.Index(pit[rows]))

        n_cell = c["n"].to_numpy(dtype=np.float64)
        cell_hit = ~np.isnan(n_cell)
        n_cell0 = np.where(np.isnan(n_cell), 0.0, n_cell)
        n_pit = p["n"].to_numpy(dtype=np.float64)          # NaN <=> unseen

        for dname, fname in DEV_SPEC.items():
            key, k = fp.RATE_SPEC[fname]
            s_cell = c[key].to_numpy(dtype=np.float64)
            s_cell = np.where(np.isnan(s_cell), 0.0, s_cell)
            overall = p[key].to_numpy(dtype=np.float64) / n_pit
            shrunk = (s_cell + k * overall) / (n_cell0 + k)

            # cross-check against what features_platoon wrote for these rows
            ref = df[fname].to_numpy(dtype=np.float64)[rows]
            fin = np.isfinite(shrunk) & np.isfinite(ref)
            if fin.any():
                mad = float(np.max(np.abs(shrunk[fin] - ref[fin])))
                diag["cross_check_max_abs_dev_vs_features_platoon"] = max(
                    diag["cross_check_max_abs_dev_vs_features_platoon"], mad)
                assert mad < 1e-12, (fname, s, mad)
            assert np.isfinite(shrunk).sum() == np.isfinite(ref).sum()

            d = shrunk - overall
            dev[dname][rows] = np.where(np.isfinite(d), d, 0.0)

        hit[rows] = cell_hit
        src[rows] = max_src

        # a cell hit implies the pitcher was seen; d must be exactly 0 on a miss
        for dname in DEV_SPEC:
            miss = rows[~cell_hit]
            assert np.all(dev[dname][miss] == 0.0), (dname, s)

        diag["per_season"][str(int(s))] = {
            "table_max_season": int(max_src),
            "target_rows": int(len(rows)),
            "cell_hit_pct": round(100.0 * float(cell_hit.mean()), 3),
            "sd_d_s": round(float(np.std(dev["d_s"][rows])), 6),
            "sd_d_r": round(float(np.std(dev["d_r"][rows])), 6),
            "sd_d_s_hit_only": (round(float(np.std(dev["d_s"][rows][cell_hit])), 6)
                                if cell_hit.any() else None),
            "corr_d_s_d_r": (round(float(np.corrcoef(
                dev["d_s"][rows][cell_hit], dev["d_r"][rows][cell_hit])[0, 1]), 4)
                if cell_hit.sum() > 2 else None),
        }
        if verbose:
            ps = diag["per_season"][str(int(s))]
            print(f"  [dev] season {s}: table<= {max_src}, hit {ps['cell_hit_pct']}%, "
                  f"sd(d_s) {ps['sd_d_s']}, sd(d_r) {ps['sd_d_r']}, "
                  f"corr {ps['corr_d_s_d_r']}", flush=True)

    first = seasons[0]
    assert np.all(dev["d_s"][season == first] == 0.0)
    assert not hit[season == first].any()
    diag["season_%d_no_correction" % first] = True
    return dev, hit, src, diag


# --------------------------------------------------------------------------- #
#  stage 2 estimators — all closed form / 1-D, no refitting
# --------------------------------------------------------------------------- #
def fit_slope(resid, d):
    """beta = E[d r]/E[d^2] with an HC0 standard error.  No intercept."""
    sdd = float(np.dot(d, d))
    if sdd <= 0:
        return {"beta": 0.0, "se": None, "n": int(len(d)), "sum_d2": 0.0}
    beta = float(np.dot(d, resid) / sdd)
    e = resid - beta * d
    var = float(np.sum((d ** 2) * (e ** 2)) / (sdd ** 2))       # HC0
    return {"beta": round(beta, 6), "se": round(float(np.sqrt(var)), 6),
            "n": int(len(d)), "sum_d2": round(sdd, 6),
            "mean_d2": round(sdd / len(d), 10),
            "t": round(beta / np.sqrt(var), 3) if var > 0 else None,
            "expected_brier_gain": round(-(beta ** 2) * sdd / len(d), 9)}


def fit_gamma(p, y, d, bracket=(-40.0, 40.0)):
    """gamma minimising Brier of sigmoid(logit(p) + gamma*d).  1-D, bounded."""
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1.0 - np.clip(p, EPS, 1 - EPS)))

    def obj(g):
        return float(np.mean((1.0 / (1.0 + np.exp(-(z + g * d))) - y) ** 2))

    r = minimize_scalar(obj, bounds=bracket, method="bounded",
                        options={"xatol": 1e-8})
    return {"gamma": round(float(r.x), 6), "brier_at_gamma": round(float(r.fun), 9),
            "brier_at_0": round(obj(0.0), 9),
            "converged": bool(r.success), "n": int(len(d))}


def apply_linear(p, d, beta):
    return np.clip(p + beta * d, EPS, 1.0 - EPS)


def apply_logit(p, d, gamma):
    pc = np.clip(p, EPS, 1.0 - EPS)
    z = np.log(pc / (1.0 - pc)) + gamma * d
    return 1.0 / (1.0 + np.exp(-z))


def weighted_trend(seasons, betas, ses):
    """Weighted least squares of beta on season, plus the 2024 extrapolation."""
    x = np.asarray(seasons, dtype=float)
    b = np.asarray(betas, dtype=float)
    w = 1.0 / np.asarray(ses, dtype=float) ** 2
    xm = float(np.sum(w * x) / np.sum(w))
    bm = float(np.sum(w * b) / np.sum(w))
    sxx = float(np.sum(w * (x - xm) ** 2))
    if sxx <= 0:
        return None
    slope = float(np.sum(w * (x - xm) * (b - bm)) / sxx)
    resid = b - (bm + slope * (x - xm))
    dof = max(len(x) - 2, 1)
    s2 = float(np.sum(w * resid ** 2) / dof)
    se_slope = float(np.sqrt(s2 / sxx))
    return {"slope_per_season": round(slope, 6),
            "se_slope_per_season": round(se_slope, 6),
            "t": round(slope / se_slope, 3) if se_slope > 0 else None,
            "level_at_mean_season": round(bm, 6),
            "extrapolated_2024": round(bm + slope * (2024.0 - xm), 6),
            "pooled_minus_extrapolated_note":
                "positive => pooled OOF slope is LARGER than the trend would "
                "predict for 2024, i.e. the applied correction over-shoots"}
