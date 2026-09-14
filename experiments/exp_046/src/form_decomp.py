"""exp_046 — in-season form decomposition of the career `asof` cumulative.

WHAT THIS MODULE DOES
---------------------
`asof_pitcher_n` is a CAREER-CUMULATIVE pitch counter (verified in exp_016
labels.py: +1 per row within a pitcher in file order, 100% of rows) and
`asof_pitcher_success_rate` is given to 6 decimals, so

    S_career[t] = round( asof_pitcher_n[t] * asof_pitcher_success_rate[t] )

recovers the cumulative number of successes BEFORE row t exactly (0 when n==0
-- that is arithmetic, not a sentinel fill).

Given a FROZEN per-pitcher table taken at a season boundary,

    n_end[p]  = asof_pitcher_n(last <=S-1 row of p) + 1
    S_end[p]  = S_career(last <=S-1 row of p) + control_success(last <=S-1 row)

every row of season S can split its own career cumulative into
"everything up to the end of last season" and "this season so far":

    n_season = asof_pitcher_n - n_end[p]
    S_season = S_career        - S_end[p]

======================================================================
!!  RULE 2-4 BOUNDARY  --  READ BEFORE TOUCHING ANYTHING HERE  !!
======================================================================
`apply_decomposition()` reads, for each row:
  * that row's OWN asof_pitcher_n / asof_pitcher_success_rate / pitcher_id
  * a table that was built ONLY from train.csv rows of earlier seasons

It contains NO groupby, NO sort, NO shift, NO cumsum.  It is a pure
row-wise lookup + arithmetic, so it behaves identically whether it is handed
1 row or 250,000 rows, and it never sees any other row of the frame it is
applied to.

The table builder `build_boundary_tables()` DOES use groupby -- it is only
ever called on train.csv.  Calling it on test/valid data would be a rule 2-4
violation (differencing the same columns across rows of the scoring frame
reconstructs the label; measured at 99.893% of rows / 100.0% accuracy on
train, so this is not hypothetical).  The two functions are kept separate for
exactly this reason.
======================================================================
"""

import numpy as np
import pandas as pd

PITCHER = "pitcher_id"
N_COL = "asof_pitcher_n"
RATE_COL = "asof_pitcher_success_rate"
TARGET = "control_success"

NEW_COLS = ["n_season", "rate_season", "d_form_eb", "has_form"]


def career_successes(n, rate):
    """S_career[t] = round(n*rate); exactly 0 when n == 0.

    Returns (S_career, max_abs_rounding_residual, min_distance_of_raw_to_a_half).
    The second and third are reported so the banker's-rounding hazard flagged
    in the exp_046 audit list is measured rather than assumed away.
    """
    n = np.asarray(n, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    raw = np.where(n == 0, 0.0, n * rate)
    s = np.round(raw)
    resid = np.abs(raw - s)
    frac = np.abs(raw - np.floor(raw))
    dist_half = np.abs(frac - 0.5)
    return s, float(np.nanmax(resid)), float(np.nanmin(dist_half))


def build_boundary_tables(df, pcode, n_pitchers):
    """TRAIN ONLY.  Per-season-boundary frozen tables.

    Returns dict {boundary_season: (n_end_array, S_end_array)} where the arrays
    are indexed by `pcode` and hold the pitcher's career state at the END of
    `boundary_season`, carried forward across seasons the pitcher missed.
    A pitcher never seen up to that boundary keeps (0, 0) -- a rookie, which is
    the correct arithmetic state, not an imputation.
    """
    season = df["season"].to_numpy()
    n = df[N_COL].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)
    s_car, _, _ = career_successes(n, df[RATE_COL].to_numpy())

    # last row of each (pitcher, season) in FILE ORDER == largest n, since n
    # increments by exactly +1 per row within a pitcher (exp_016 verified).
    last = (pd.DataFrame({"p": pcode, "s": season})
              .groupby(["p", "s"], sort=False).tail(1).index.to_numpy())

    cur_n = np.zeros(n_pitchers, dtype=np.float64)
    cur_s = np.zeros(n_pitchers, dtype=np.float64)
    tables = {}
    seasons = np.sort(np.unique(season))
    last_season = season[last]
    for s in seasons:
        idx = last[last_season == s]
        cur_n = cur_n.copy()
        cur_s = cur_s.copy()
        cur_n[pcode[idx]] = n[idx] + 1.0
        cur_s[pcode[idx]] = s_car[idx] + y[idx]
        tables[int(s)] = (cur_n, cur_s)
    return tables


def extend_table_for_unseen(n_end_tab, s_end_tab):
    """Append one slot for pitchers that never appear in train.csv at all.

    Their career state at the boundary is (0 pitches, 0 successes) -- that is
    ARITHMETIC, not a sentinel fill: a pitcher with no train history has thrown
    zero recorded pitches, exactly like a train rookie, for whom
    build_boundary_tables() already stores (0, 0).  The extra slot only extends
    the same rule to ids absent from the factorisation.

    Returns (n_end_ext, s_end_ext, unseen_index).
    """
    return (np.concatenate([n_end_tab, [0.0]]),
            np.concatenate([s_end_tab, [0.0]]),
            len(n_end_tab))


def apply_decomposition(n, rate, pcode, n_end_tab, s_end_tab, k):
    """ROW-WISE ONLY.  No groupby / sort / shift / cumsum.  See module header.

    `n`, `rate`, `pcode` are this row's own values; `n_end_tab`/`s_end_tab` are
    frozen train-derived arrays.  Returns a dict of the new columns plus the
    intermediates the diagnostics need.
    """
    s_car, _, _ = career_successes(n, rate)
    n_end = n_end_tab[pcode]
    s_end = s_end_tab[pcode]

    n_season = np.asarray(n, dtype=np.float64) - n_end
    s_season = s_car - s_end

    has_prior = n_end > 0
    has_ssn = n_season > 0
    # 🔴 no sentinel fills (CLAUDE.md).  Undefined ratios stay NaN and the
    # `has_form` flag column carries the "is it defined" information.
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_season = np.where(has_ssn, s_season / np.where(has_ssn, n_season, 1.0), np.nan)
        rate_prior = np.where(has_prior, s_end / np.where(has_prior, n_end, 1.0), np.nan)

    has_form = has_prior & has_ssn
    d_form = np.where(has_form, rate_season - rate_prior, np.nan)
    shrink = np.where(has_ssn, n_season / (n_season + float(k)), 0.0)
    d_form_eb = np.where(has_form, d_form * shrink, np.nan)

    return {
        "S_career": s_car, "n_end": n_end, "S_end": s_end,
        "n_season": n_season, "S_season": s_season,
        "rate_season": rate_season, "rate_prior": rate_prior,
        "d_form": d_form, "d_form_eb": d_form_eb,
        "has_form": has_form.astype(np.int8),
    }
