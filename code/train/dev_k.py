"""exp_023 — the personal deviation `d_s` at an ARBITRARY shrink constant k.

WHY THIS FILE EXISTS AT ALL
---------------------------
`offset.build_deviations` (byte-preserved from exp_019/exp_022) hard-codes
k = 111 because it reads `fp.RATE_SPEC["pit_success_rate_vs_hand"][1]` and then
cross-checks its own output against the column `features_platoon` wrote — a
check that is only meaningful at that k.  exp_023 sweeps k, so the shrink math
has to be re-expressible.

THE FORM OF `d`, AND WHY IT IS NOT THE INHERITED ONE
-----------------------------------------------------
The inherited code writes the deviation as a literal subtraction

    d = (s_cell + k*overall) / (n_cell + k)  -  overall                     (A)

On a cell MISS (`s_cell = 0`, `n_cell = 0`) that is `(k*overall)/k - overall`,
which IEEE-754 does NOT guarantee to be 0: `k*x/k == x` fails for most k.  It
happened to hold at k = 111 for every row of this file; the first run of this
sweep died on `k = 200, season 2021`.  exp_022 hit the same class of bug in the
league term (`k*L/(n+k)` broke `BK2 ⊇ BK1` bitwise) and fixed it by regrouping
to `L*(k/(n+k))`; the lesson had not been carried over to the personal path.

The fix here is to expand the subtraction so `overall` cancels ANALYTICALLY:

    (s + k*o)/(n + k) - o = (s + k*o - o*n - o*k)/(n + k) = (s - n*o)/(n + k)  (B)

an identity.  On a miss the numerator is `0 - 0*o = 0` for every k, so (B)
returns exactly 0.0 by construction rather than by luck.  (B) is also better
conditioned on HIT rows: (A) subtracts two numbers near 0.49 to get one near
0.02, losing ~1.5 digits to cancellation, while (B) never forms them.

CONSEQUENCE, STATED PLAINLY: at k = 111, (B) is NOT bit-identical to (A) on hit
rows — the two differ at the ulp level (~1e-18, i.e. ~1e-16 relative).  So the
k = 111 arm reproduces exp_022 to floating-point noise, not byte for byte, and
`fit_grid.py` reports the measured max |(A) - (B)| instead of asserting
equality.  The reproduction gate that actually governs — fold_2024 Brier within
5e-6 of exp_022 BK1f's 0.248037745 — is unaffected by a 1e-18 perturbation and
is run BOTH ways (through this module and through the inherited (A) path) so
the exact exp_022 number stays on the record.

WHAT IS NOT RE-IMPLEMENTED
--------------------------
The season-forward-chained aggregation `fp._agg` and the league table
`backoff.build_league_dev` are called, not rewritten.  The league table is the
code the exp_022 leak audit independently recomputed; it is byte-preserved and
does not depend on k.

MISS ROWS AND k = 0
-------------------
`d` is taken from (B) on cell-HIT rows and is exactly 0.0 on cell-MISS rows.
That 0 is not a sentinel fill: with an empty cell the shrunk estimate IS the
pitcher's unconditional rate, so the deviation is analytically zero.  Branching
on `cell_hit` rather than dividing also keeps k = 0 well defined — at k = 0 a
miss row would be 0/0 — and keeps the NaN of a never-seen pitcher (`overall`
undefined) out of the arithmetic instead of relying on a finiteness sweep
afterwards.

That miss rows carry exactly 0.0 is load-bearing, not cosmetic: the hard-backoff
arm adds `L` only where `d == 0`, so its HIT rows must stay bit-identical to the
control's.  The exp_022 audit accepted the coverage attribution on that basis.
It is asserted per season, per k, and the assertion is not relaxed.
"""

import numpy as np
import pandas as pd

import features_platoon as fp

SUCC_KEY = "succ"
SUCC_COL = "pit_success_rate_vs_hand"


def build_multi_k(df, ok, succ_season, f_excl, ks, verbose=True):
    """Return (dict k -> d array, hit mask, n_cell array, src, diagnostics).

    `d[k][i]` is the season-as-of personal deviation for row i at shrink
    constant k; `n_cell[i]` is the (pitcher, batter_hand) cell count that fed
    it, 0 on a miss and NaN for the first season (no table exists there).
    """
    season = df["season"].to_numpy()
    pit = df[fp.PITCHER].to_numpy()
    hand = df[fp.HAND].to_numpy()
    nums = fp.numerators(df)
    base_ok = ok & ~f_excl
    seasons = sorted(set(season.tolist()))

    n = len(df)
    dev = {k: np.zeros(n, dtype=np.float64) for k in ks}
    hit = np.zeros(n, dtype=bool)
    n_cell_out = np.full(n, np.nan)
    src = np.full(n, np.nan)
    diag = {"ks": list(ks), "per_season": {}}

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

        s_cell = c[SUCC_KEY].to_numpy(dtype=np.float64)
        s_cell = np.where(np.isnan(s_cell), 0.0, s_cell)
        overall = p[SUCC_KEY].to_numpy(dtype=np.float64) / n_pit

        # numerator of form (B).  A cell hit implies the pitcher was seen, so
        # `overall` is finite wherever this value is used; on miss rows it may
        # be NaN and is discarded by the `cell_hit` branch below, never by a
        # post-hoc finiteness sweep.
        with np.errstate(invalid="ignore", divide="ignore"):
            num = s_cell - n_cell0 * overall
            for k in ks:
                denom = n_cell0 + k
                # every HIT row has n_cell > 0, hence denom > 0 even at k = 0
                safe = np.where(cell_hit, denom, 1.0)
                d = np.where(cell_hit, num / safe, 0.0)
                dev[k][rows] = d

        hit[rows] = cell_hit
        n_cell_out[rows] = n_cell0
        src[rows] = max_src

        for k in ks:
            assert np.all(dev[k][rows[~cell_hit]] == 0.0), ("d != 0 on miss", k, s)

        diag["per_season"][str(int(s))] = {
            "table_max_season": int(max_src),
            "target_rows": int(len(rows)),
            "cell_hit_pct": round(100.0 * float(cell_hit.mean()), 3),
            "sd_d_by_k": {str(k): round(float(np.std(dev[k][rows])), 6) for k in ks},
            "sd_d_hit_only_by_k": {
                str(k): round(float(np.std(dev[k][rows][cell_hit])), 6) for k in ks},
        }
        if verbose:
            print(f"  [dev_k] season {s}: table<= {max_src}, "
                  f"hit {diag['per_season'][str(int(s))]['cell_hit_pct']}%, "
                  f"sd(d) {diag['per_season'][str(int(s))]['sd_d_by_k']}", flush=True)

    first = seasons[0]
    for k in ks:
        assert np.all(dev[k][season == first] == 0.0)
    assert not hit[season == first].any()
    diag["season_%d_no_correction" % first] = True
    return dev, hit, n_cell_out, src, diag


def league_weight(n_cell0, k):
    """w = k / (n_cell + k), the weight the league term gets under `league`.

    * k > 0, cell MISS (n = 0):  w = k/k = 1.0 EXACTLY, so the league arm
      provably contains the hard backoff on miss rows rather than approximately
      containing it (the exp_022 grouping argument, preserved).
    * k = 0, cell HIT  (n > 0):  w = 0 — no shrinkage, no league term.
    * k = 0, cell MISS (n = 0):  0/0.  Taken as the limit from k -> 0+, which is
      1.0 for every k > 0.  Documented convention, not a fill: with no data in
      the cell the estimator has only the league level to offer, and at every
      other point of the grid it offers exactly that.
    """
    n_cell0 = np.asarray(n_cell0, dtype=np.float64)
    denom = n_cell0 + k
    safe = np.where(denom > 0, denom, 1.0)
    w = np.where(denom > 0, k / safe, 1.0)
    if k > 0:
        assert np.all(w[n_cell0 == 0] == 1.0)
    return w


def build_arm(d_k, hit, L_eff, n_cell0, k, target):
    """`d` for one grid arm.

    self   : d = d_s(k) + (cell miss) * L          -- hard backoff, exp_022 G1
    league : d = d_s(k) + L * k/(n_cell + k)       -- exp_022 G2 form
    """
    if target == "self":
        return d_k + np.where(hit, 0.0, L_eff)
    if target == "league":
        return d_k + L_eff * league_weight(n_cell0, k)
    raise ValueError(target)
