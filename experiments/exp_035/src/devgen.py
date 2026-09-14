"""exp_026 — the SAME deviation machinery, on an arbitrary numerator key.

WHY THIS FILE EXISTS
--------------------
Everything deployed so far corrects the model with `d_s`, the personal deviation
of the SUCCESS rate.  exp_026 puts `d_r` — the deviation of the MARGINAL REVERSE
rate — next to it.  The two are built by identical arithmetic on different
numerators, but the two audited modules that build `d_s` each hard-code the
`"succ"` numerator:

  * `dev_k.build_multi_k`  -> `SUCC_KEY = "succ"`   (personal, arbitrary k)
  * `backoff.build_league_dev` -> `SUCC_KEY = "succ"` (league (ph, bh) cells)

Both are BYTE-PRESERVED in this experiment (their sha256 are asserted by the
driver) because a leak audit has already read them.  So the reverse axis needs
its own code path.

WHAT IS DONE INSTEAD OF COPY-PASTE
-----------------------------------
This module is the two functions above with the numerator key lifted to an
argument and NOTHING else changed — same masks, same forward chaining, same
expanded float form, same runtime asserts.  The driver then does the only check
that makes that claim verifiable: it runs THIS code at `key="succ"` and asserts
the result is BIT-IDENTICAL to what the byte-preserved modules produce
(`np.array_equal` on the deviation arrays, the hit mask, the cell counts, the
league array and its coverage mask).  If the generalisation had drifted in any
way that matters to a float, that check fails and the run aborts.  The reverse
axis is therefore not "believed to be the same code" — it is the same code,
demonstrated on the axis whose output is already on the record.

THE FLOAT FORM (inherited reasoning, restated because it applies here too)
--------------------------------------------------------------------------
The personal deviation is the EXPANDED form

    d = (s_cell - n_cell * overall) / (n_cell + k)

not the literal subtraction `(s + k*o)/(n + k) - o`, which is not exactly 0 on a
cell MISS for a general k (`k*x/k != x` in IEEE-754).  The league term is
grouped as `L * (k/(n+k))`, not `k*L/(n+k)`, so that on a miss it is `L * 1.0`
exactly.  exp_023's first launch died on the first of these; exp_022 hit the
second.  Both apply to `d_r` unchanged.

LEAK BOUNDARY
-------------
`rev` is a RECOVERED label (`is_rev_only + is_both`, from `labels.recover`,
which differences the same pitcher's NEXT row).  It is legal here for exactly
the reason `succ` is: it is consumed only as the NUMERATOR OF A FROZEN TABLE
built on seasons <= s-1, never as a row-level input.  A season-s row reads a
table that cannot contain season s, so it cannot contain its own successor.
The driver re-verifies this against an independent second read of train.csv and
additionally asserts that every deviation is CONSTANT within
(season, pitcher, batter_hand) — a row-level label reaching a row's own `d`
would break that constancy.
"""

import numpy as np
import pandas as pd

import features_platoon as fp

PIT_HAND = "pitcher_hand"
BAT_HAND = "batter_hand"


# --------------------------------------------------------------------------- #
#  personal (pitcher, batter_hand) deviation at arbitrary k — generic numerator
#  == dev_k.build_multi_k with SUCC_KEY lifted to `key`
# --------------------------------------------------------------------------- #
def build_multi_k(df, ok, succ_season, f_excl, ks, key, verbose=True):
    """Return (dict k -> d array, hit mask, n_cell array, src, diagnostics)."""
    season = df["season"].to_numpy()
    pit = df[fp.PITCHER].to_numpy()
    hand = df[fp.HAND].to_numpy()
    nums = fp.numerators(df)
    assert key in nums, key
    base_ok = ok & ~f_excl
    seasons = sorted(set(season.tolist()))

    n = len(df)
    dev = {k: np.zeros(n, dtype=np.float64) for k in ks}
    hit = np.zeros(n, dtype=bool)
    n_cell_out = np.full(n, np.nan)
    src = np.full(n, np.nan)
    diag = {"ks": list(ks), "numerator_key": key, "per_season": {}}

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

        s_cell = c[key].to_numpy(dtype=np.float64)
        s_cell = np.where(np.isnan(s_cell), 0.0, s_cell)
        overall = p[key].to_numpy(dtype=np.float64) / n_pit

        with np.errstate(invalid="ignore", divide="ignore"):
            num = s_cell - n_cell0 * overall
            for k in ks:
                denom = n_cell0 + k
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
            print(f"  [devgen:{key}] season {s}: table<= {max_src}, "
                  f"hit {diag['per_season'][str(int(s))]['cell_hit_pct']}%, "
                  f"sd(d) {diag['per_season'][str(int(s))]['sd_d_by_k']}", flush=True)

    first = seasons[0]
    for k in ks:
        assert np.all(dev[k][season == first] == 0.0)
    assert not hit[season == first].any()
    diag["season_%d_no_correction" % first] = True
    return dev, hit, n_cell_out, src, diag


# --------------------------------------------------------------------------- #
#  league (pitcher_hand, batter_hand) deviation — generic numerator
#  == backoff.build_league_dev with SUCC_KEY lifted to `key`
# --------------------------------------------------------------------------- #
def build_league_dev(df, ok, succ_season, f_excl, key, verbose=True):
    """Return (L array, has_L mask, src_max_season array, diagnostics)."""
    season = df["season"].to_numpy()
    ph = df[PIT_HAND].to_numpy()
    bh = df[BAT_HAND].to_numpy()
    nums = fp.numerators(df)
    assert key in nums, key
    num = nums[key]
    hand_ok = ~pd.isna(df[PIT_HAND]).to_numpy() & ~pd.isna(df[BAT_HAND]).to_numpy()

    base_ok = ok & ~f_excl
    seasons = sorted(set(season.tolist()))

    n = len(df)
    L = np.zeros(n, dtype=np.float64)
    has = np.zeros(n, dtype=bool)
    src = np.full(n, np.nan)
    diag = {"key": f"({PIT_HAND}, {BAT_HAND})", "numerator_key": key,
            "seasons": [int(s) for s in seasons], "per_season": {},
            "hand_missing_rows": int((~hand_ok).sum())}

    for s in seasons[1:]:
        max_src = s - 1
        m = base_ok & (season <= max_src) & (succ_season <= max_src) & hand_ok

        # --- RUNTIME LEAK ASSERT: no row of season >= s may feed this table --- #
        used = season[m]
        assert used.size > 0, f"empty league table for season {s}"
        assert int(used.max()) <= max_src, (
            f"LEAGUE AS-OF VIOLATION season {s}: table saw season {int(used.max())}")
        assert int(used.max()) < s
        assert np.isfinite(num[m]).all(), ("recovered numerator is NaN inside the "
                                           "table mask", key, s)

        g = pd.DataFrame({"ph": ph[m], "bh": bh[m], "y": num[m]})
        cell = g.groupby(["ph", "bh"], sort=True).agg(n=("y", "size"), s=("y", "sum"))
        pgrp = g.groupby("ph", sort=True).agg(n=("y", "size"), s=("y", "sum"))
        rate_cell = cell["s"] / cell["n"]
        rate_ph = pgrp["s"] / pgrp["n"]
        dev_cell = rate_cell - rate_ph.reindex(
            cell.index.get_level_values("ph")).to_numpy()

        rows = np.flatnonzero((season == s) & hand_ok)
        mi = pd.MultiIndex.from_arrays([ph[rows], bh[rows]])
        v = dev_cell.reindex(mi).to_numpy(dtype=np.float64)
        found = np.isfinite(v)
        L[rows] = np.where(found, v, 0.0)
        has[rows] = found
        src[(season == s)] = max_src

        chk = float(np.max(np.abs(
            (dev_cell * cell["n"]).groupby(level="ph").sum()
            / cell["n"].groupby(level="ph").sum())))
        assert chk < 1e-12, ("league deviations are not zero-sum within ph", s, chk)

        diag["per_season"][str(int(s))] = {
            "table_max_season": int(max_src),
            "table_rows": int(m.sum()),
            "table_cells": int(len(cell)),
            "target_rows": int((season == s).sum()),
            "rows_with_L_pct": round(100.0 * float(has[season == s].mean()), 3),
            "L_by_cell": {f"{int(a)}x{int(b)}": round(float(val), 6)
                          for (a, b), val in dev_cell.items()},
            "n_by_cell": {f"{int(a)}x{int(b)}": int(val)
                          for (a, b), val in cell["n"].items()},
        }
        if verbose:
            print(f"  [devgen-league:{key}] season {s}: table<= {max_src} "
                  f"({int(m.sum()):,} rows, {len(cell)} cells) -> L "
                  f"{diag['per_season'][str(int(s))]['L_by_cell']}", flush=True)

    first = seasons[0]
    assert np.all(L[season == first] == 0.0)
    assert not has[season == first].any()
    diag["season_%d_no_correction" % first] = True
    return L, has, src, diag


# --------------------------------------------------------------------------- #
#  multi-term closed-form slope (no intercept)
# --------------------------------------------------------------------------- #
def fit_joint(resid, cols, round_to=6):
    """beta = (D'D)^-1 D'r for one or more deviation columns.  No intercept.

    An intercept is excluded for the same reason `offset.fit_slope` excludes it:
    it would be a global probability shift (the exp_005 season-drift correction),
    a different mechanism that was measured to hurt on this base.

    With a single column this is algebraically `offset.fit_slope`; the driver
    asserts the two agree to 1e-12 before relying on either.  The returned betas
    are ROUNDED to `round_to` decimals because that is what the deployed path
    does (`offset.fit_slope` rounds to 6), and the reproduction gate is defined
    on numbers produced that way.  The unrounded values are returned too.
    """
    D = np.column_stack([np.asarray(c, dtype=np.float64) for c in cols])
    r = np.asarray(resid, dtype=np.float64)
    A = D.T @ D
    b = D.T @ r
    beta = np.linalg.solve(A, b)
    e = r - D @ beta
    # HC0 sandwich
    Ainv = np.linalg.inv(A)
    meat = (D * (e ** 2)[:, None]).T @ D
    cov = Ainv @ meat @ Ainv
    se = np.sqrt(np.diag(cov))
    n = len(r)
    br = [round(float(x), round_to) for x in beta]
    return {
        "beta": br,
        "beta_unrounded": [float(x) for x in beta],
        "se": [round(float(x), 6) for x in se],
        "t": [round(float(bb / ss), 3) if ss > 0 else None
              for bb, ss in zip(beta, se)],
        "n": int(n),
        "n_terms": int(D.shape[1]),
        "gram": [[float(x) for x in row] for row in A],
        "cond_gram": float(np.linalg.cond(A)),
        "corr_terms": [[float(x) for x in row]
                       for row in np.corrcoef(D, rowvar=False)]
        if D.shape[1] > 1 else [[1.0]],
        "expected_brier_gain": float(-(beta @ A @ beta) / n),
    }


def apply_joint(p, cols, betas, eps=1e-6):
    out = np.asarray(p, dtype=np.float64).copy()
    for c, b in zip(cols, betas):
        out = out + b * np.asarray(c, dtype=np.float64)
    return np.clip(out, eps, 1.0 - eps)
