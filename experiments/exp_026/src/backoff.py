"""exp_022 — league-level (pitcher_hand x batter_hand) backoff for the offset.

THE PROBLEM
-----------
exp_019/exp_020's offset is `p' = clip(p + beta * d_s)` where

    d_s = EB-shrunk (pitcher_id, batter_hand) success rate
          - that pitcher's own unconditional success rate.

`d_s` is exactly 0 whenever the (pitcher, hand) cell is empty in the frozen
table — 21.5% of fold_2024 rows.  Those rows get no correction at all.

THE BACKOFF
-----------
Platoon advantage exists at the LEAGUE level before it exists per pitcher.  So
when the personal cell is empty, fall back to the league deviation

    L(ph, bh) = P(success | pitcher_hand=ph, batter_hand=bh)
              - P(success | pitcher_hand=ph)

which has the SAME unit and the SAME sign convention as `d_s` (both are
"conditional rate minus the corresponding unconditional rate"), and sums to zero
over `bh` within each `ph` group by construction.  Four cells: (1,1) (1,2)
(2,1) (2,2).

ARMS
----
  d_G0 = d_s                                     (control, exp_020 E1)
  d_G1 = d_s + (cell miss) * L                   HARD backoff.  `d_s` is exactly
                                                 0 on a miss (asserted by
                                                 offset.build_deviations), so on
                                                 HIT rows d_G1 IS d_s, bit for
                                                 bit.  That is the integrity
                                                 property the whole comparison
                                                 rests on and it is asserted.
  d_G2 = d_s + k * L / (n_cell + k)               SOFT shrink.  Same algebra as
                                                 the personal estimator with the
                                                 shrink target moved from the
                                                 pitcher's unconditional rate to
                                                 that rate PLUS L:
                                                   (s + k(overall+L))/(n+k) - overall
                                                   = d_s + kL/(n+k).
                                                 On a miss (n_cell = 0) this
                                                 equals L, i.e. G2 contains G1;
                                                 on a HIT row it moves the value
                                                 too, so G2 changes two things
                                                 at once and its delta cannot be
                                                 attributed to coverage.

AS-OF (this is the leak boundary, and it is asserted at runtime)
----------------------------------------------------------------
The league table for a season-s row is built from seasons <= s-1 ONLY, with the
identical row mask the personal table uses (`recov_ok`, successor inside the
window, exp_004 F-mask).  `build_league_dev` asserts, for every season it
builds, that `max(season of any row that fed the table) < s`.  Season 2019 has
no predecessor and gets L = 0 — a no-op, not a sentinel fill, exactly as the
personal table gets NaN there.  The test path would freeze the same table on
<= 2024, so training and inference compute the identical quantity.

MISSINGNESS
-----------
`pitcher_hand` or `batter_hand` missing -> L = 0 (the config's rule: fall back
to the current behaviour, do not invent a value).  A (ph, bh) pair that never
appeared in the table -> L = 0 for the same reason.  Neither is a fill: 0 is
"apply no correction", which is what we would do without this module at all.
"""

import numpy as np
import pandas as pd

import features_platoon as fp

PIT_HAND = "pitcher_hand"
BAT_HAND = "batter_hand"
SUCC_KEY = "succ"
SUCC_COL = "pit_success_rate_vs_hand"


def build_league_dev(df, ok, succ_season, f_excl, verbose=True):
    """Return (L array, has_L mask, src_max_season array, diagnostics).

    L[i] is the league (pitcher_hand, batter_hand) success deviation for row i,
    estimated on seasons <= season(i) - 1, or 0.0 where undefined.
    """
    season = df["season"].to_numpy()
    ph = df[PIT_HAND].to_numpy()
    bh = df[BAT_HAND].to_numpy()
    succ = fp.numerators(df)[SUCC_KEY]
    hand_ok = ~pd.isna(df[PIT_HAND]).to_numpy() & ~pd.isna(df[BAT_HAND]).to_numpy()

    base_ok = ok & ~f_excl
    seasons = sorted(set(season.tolist()))

    n = len(df)
    L = np.zeros(n, dtype=np.float64)
    has = np.zeros(n, dtype=bool)
    src = np.full(n, np.nan)
    diag = {"key": f"({PIT_HAND}, {BAT_HAND})", "seasons": [int(s) for s in seasons],
            "per_season": {}, "hand_missing_rows": int((~hand_ok).sum())}

    for s in seasons[1:]:
        max_src = s - 1
        m = base_ok & (season <= max_src) & (succ_season <= max_src) & hand_ok

        # --- RUNTIME LEAK ASSERT: no row of season >= s may feed this table --- #
        used = season[m]
        assert used.size > 0, f"empty league table for season {s}"
        assert int(used.max()) <= max_src, (
            f"LEAGUE AS-OF VIOLATION season {s}: table saw season {int(used.max())}")
        assert int(used.max()) < s

        g = pd.DataFrame({"ph": ph[m], "bh": bh[m], "y": succ[m]})
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

        # within a pitcher-hand group the deviations are a zero-sum split
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
            print(f"  [league] season {s}: table<= {max_src} "
                  f"({int(m.sum()):,} rows, {len(cell)} cells) -> L "
                  f"{diag['per_season'][str(int(s))]['L_by_cell']}", flush=True)

    first = seasons[0]
    assert np.all(L[season == first] == 0.0)
    assert not has[season == first].any()
    diag["season_%d_no_correction" % first] = True
    return L, has, src, diag


def build_arm_deviations(df, d_s, hit, L, has_L, eb_k):
    """d_G0 / d_G1 / d_G2 from the personal deviation and the league deviation.

    `df` must already carry `features_platoon`'s `pit_n_vs_hand` (the personal
    cell count) and `pit_success_rate_vs_hand` (NaN <=> pitcher unseen).
    """
    n_cell = df[fp.COUNT_COL].to_numpy(dtype=np.float64)
    n_cell0 = np.where(np.isfinite(n_cell), n_cell, 0.0)
    seen = np.isfinite(df[SUCC_COL].to_numpy(dtype=np.float64))
    Le = np.where(has_L, L, 0.0)

    # the estimator's own no-op on a miss — re-asserted here, not assumed
    assert np.all(d_s[~hit] == 0.0), "d_s is not exactly 0 on cell misses"
    assert np.all(n_cell0[hit] > 0), "hit rows with empty personal cell"
    assert np.all(n_cell0[~hit] == 0.0), "miss rows with a non-empty personal cell"

    d = {
        "G0": d_s.copy(),
        "G1": d_s + np.where(hit, 0.0, Le),
        # NOTE the grouping: `Le * (k/(n+k))`, not `k*Le/(n+k)`.  They are the
        # same number in exact arithmetic and NOT the same float: on a miss the
        # first form gives Le * 1.0 == Le exactly, so G2 provably contains G1
        # rather than approximately containing it.
        "G2": d_s + Le * (eb_k / (n_cell0 + eb_k)),
    }

    # ---- G1 INTEGRITY: the hard backoff may touch miss rows and nothing else -- #
    integrity = {
        "n_hit": int(hit.sum()), "n_miss": int((~hit).sum()),
        "hit_rows_bit_identical_to_G0": bool(
            np.array_equal(d["G1"][hit], d["G0"][hit])),
        "n_hit_rows_differing": int(np.sum(d["G1"][hit] != d["G0"][hit])),
        "max_abs_diff_on_hit_rows": float(
            np.max(np.abs(d["G1"][hit] - d["G0"][hit]))) if hit.any() else 0.0,
        "n_miss_rows_changed": int(np.sum(d["G1"][~hit] != d["G0"][~hit])),
        "miss_rows_G0_all_zero": bool(np.all(d["G0"][~hit] == 0.0)),
        "G2_equals_G1_on_miss_rows": bool(
            np.array_equal(d["G2"][~hit], d["G1"][~hit])),
        "n_hit_rows_G2_differs_from_G0": int(np.sum(d["G2"][hit] != d["G0"][hit])),
        "pitcher_seen_but_cell_empty": int(np.sum(seen & ~hit)),
    }
    assert integrity["hit_rows_bit_identical_to_G0"], (
        "G1 HARD-BACKOFF INTEGRITY FAILED — hit rows moved: " + str(integrity))
    assert integrity["G2_equals_G1_on_miss_rows"], integrity
    return d, integrity
