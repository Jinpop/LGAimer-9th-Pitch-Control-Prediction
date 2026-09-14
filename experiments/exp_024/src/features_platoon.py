"""exp_018 — (pitcher_id x batter_hand) CONDITIONAL pitcher-tendency lookups.

WHY
---
All 47 provided columns describe a pitcher UNCONDITIONALLY (`asof_pitcher_*` is
a career-to-date rate over every batter he faced).  A pitcher's command is not
symmetric across the batter's handedness: the catcher's target mirrors and
arm-side / glove-side command differ, and the size of that asymmetry differs
per pitcher.  This module builds the conditional counterpart:

    (pitcher, batter_hand) -> success / marginal-reverse / marginal-middle rate

shrunk empirical-Bayes towards **that pitcher's own unconditional rate** (NOT
the league mean — the unconditional level is already in the 47 columns, so what
we want to add is only the conditional DEVIATION).

AS-OF DISCIPLINE (this is the leak boundary)
--------------------------------------------
The numerators are labels.  A season-s row therefore may only be joined against
a table built from seasons <= s-1.  `build_season_forward_chained()` builds one
table per season and never lets season s see itself or later; season 2019 has no
predecessor and keeps NaN (a sentinel fill is banned by the project rules and
would also be a lie about what is known).  The same season-frozen structure is
what test 2025 would use (a table frozen on <= 2024), so training and inference
compute the identical quantity — no train/inference asymmetry, and no test row
ever consults another test row (rule 2-4).

Second half of the boundary, inherited from exp_016: `labels.recover()` derives
`is_X[t]` by differencing the SAME pitcher's row t+1.  A row whose successor
lives in a later season would need a later-season row to be recovered at all, so
those rows are excluded from every table (`succ_season <= max_season`).  This is
also what makes the table reproducible at deployment time, where the successor
of a pitcher's last 2024 row simply does not exist yet.
"""

import numpy as np
import pandas as pd

PITCHER = "pitcher_id"
HAND = "batter_hand"

# feature name -> (numerator key, EB k from config.yaml prior_evidence)
RATE_SPEC = {
    "pit_success_rate_vs_hand": ("succ", 111),
    "pit_reverse_rate_vs_hand": ("rev", 65),
    "pit_middle_rate_vs_hand": ("mid", 583),
}
COUNT_COL = "pit_n_vs_hand"
ALL_COLS = list(RATE_SPEC) + [COUNT_COL]

# arm -> the platoon columns it adds on top of the 47 provided ones
ARM_COLS = {
    "A0": [],
    "A1": ["pit_reverse_rate_vs_hand"],
    "A2": ["pit_reverse_rate_vs_hand", "pit_success_rate_vs_hand"],
    "A3": ["pit_reverse_rate_vs_hand", "pit_success_rate_vs_hand",
           "pit_middle_rate_vs_hand", COUNT_COL],
}


def numerators(df):
    """The three event indicators the tables count.

    `succ` is the provided label.  `rev` / `mid` are the MARGINAL modes, i.e.
    exactly the quantity the organisers' `asof_pitcher_{reverse,middle}_rate`
    measures (config prior_evidence.definition_check: corr = 1.0 with the
    provided column at career end), reconstructed per row so it can be
    conditioned on the batter's hand.
    """
    both = df["is_both"].to_numpy(dtype=np.float64)
    return {
        "succ": df["control_success"].to_numpy(dtype=np.float64),
        "rev": df["is_rev_only"].to_numpy(dtype=np.float64) + both,
        "mid": df["is_mid_only"].to_numpy(dtype=np.float64) + both,
    }


def _agg(pit, hand, nums, mask):
    """Cell (pitcher, hand) and pitcher-level sums over `mask` rows only."""
    idx = np.flatnonzero(mask)
    d = pd.DataFrame({"p": pit[idx], "h": hand[idx]})
    for k, v in nums.items():
        d[k] = v[idx]
    aggs = {k: (k, "sum") for k in nums}
    cell = d.groupby(["p", "h"], sort=False).agg(n=("p", "size"), **aggs)
    pitc = d.groupby("p", sort=False).agg(n=("p", "size"), **aggs)
    return cell, pitc


def _apply(cell, pitc, pit_rows, hand_rows):
    """Shrunk conditional rates for the given rows, from the given tables."""
    mi = pd.MultiIndex.from_arrays([pit_rows, hand_rows])
    c = cell.reindex(mi)
    p = pitc.reindex(pd.Index(pit_rows))

    n_cell = c["n"].to_numpy(dtype=np.float64)
    n_cell = np.where(np.isnan(n_cell), 0.0, n_cell)
    n_pit = p["n"].to_numpy(dtype=np.float64)      # NaN <=> pitcher unseen

    out = {COUNT_COL: n_cell}
    for fname, (key, k) in RATE_SPEC.items():
        s_cell = c[key].to_numpy(dtype=np.float64)
        s_cell = np.where(np.isnan(s_cell), 0.0, s_cell)
        overall = p[key].to_numpy(dtype=np.float64) / n_pit   # NaN if unseen
        # shrink towards the pitcher's OWN unconditional rate.  n_cell == 0 ->
        # the value collapses to `overall`; pitcher unseen -> NaN (kept).
        out[fname] = (s_cell + k * overall) / (n_cell + k)
    return out


def build_season_forward_chained(df, ok, succ_season, f_excl, verbose=True):
    """Add the four platoon columns to `df` IN PLACE, season-forward-chained.

    Returns (diagnostics dict, `src_max_season` array).  `src_max_season[i]` is
    the newest season that fed row i's lookup value (NaN for 2019) and is what
    train.py asserts against the fold's validation season.
    """
    season = df["season"].to_numpy()
    pit = df[PITCHER].to_numpy()
    hand = df[HAND].to_numpy()
    nums = numerators(df)

    # rows a table may aggregate: recoverable, successor inside the window, and
    # not an exp_004-masked (pre-2023 F-league) row.
    base_ok = ok & ~f_excl
    seasons = sorted(set(season.tolist()))

    for c in ALL_COLS:
        df[c] = np.nan
    src = np.full(len(df), np.nan)

    diag = {"seasons": [int(s) for s in seasons], "per_season": {},
            "hand_codes": sorted(pd.unique(hand).tolist()),
            "rows_excluded_from_tables": {
                "not_recoverable_or_successor_outside": int((~ok).sum()),
                "exp_004_F_mask": int(f_excl.sum())}}

    for s in seasons[1:]:
        max_src = s - 1
        m = base_ok & (season <= max_src) & (succ_season <= max_src)
        cell, pitc = _agg(pit, hand, nums, m)
        rows = season == s
        vals = _apply(cell, pitc, pit[rows], hand[rows])
        for c in ALL_COLS:
            df.loc[rows, c] = vals[c]
        src[rows] = max_src

        n_cell = vals[COUNT_COL]
        diag["per_season"][str(int(s))] = {
            "table_max_season": int(max_src),
            "table_rows": int(m.sum()),
            "table_cells": int(len(cell)),
            "table_pitchers": int(len(pitc)),
            "target_rows": int(rows.sum()),
            "row_coverage_cell_nonempty_pct": round(
                100.0 * float(np.mean(n_cell > 0)), 3),
            "row_coverage_pitcher_seen_pct": round(
                100.0 * float(np.mean(~np.isnan(vals["pit_success_rate_vs_hand"]))), 3),
            "median_n_cell": float(np.median(n_cell)),
        }
        if verbose:
            print(f"  [platoon] season {s}: table<= {max_src} "
                  f"({int(m.sum()):,} rows, {len(cell):,} cells) -> "
                  f"{int(rows.sum()):,} rows, cell-hit "
                  f"{diag['per_season'][str(int(s))]['row_coverage_cell_nonempty_pct']}%",
                  flush=True)

    first = seasons[0]
    diag["season_%d_all_nan" % first] = bool(
        df.loc[season == first, list(RATE_SPEC)].isna().all().all())
    diag["nan_share_whole_file_pct"] = {
        c: round(100.0 * float(df[c].isna().mean()), 3) for c in ALL_COLS}
    return diag, src


# --------------------------------------------------------------------------- #
#  prior-evidence reproduction (config prior_evidence.measurements)
# --------------------------------------------------------------------------- #
def prelim_stats(df, ok, succ_season, f_excl, max_season, min_n=100,
                 next_season=None):
    """Re-measure the between-pitcher heterogeneity of the L/R split.

    Method (config prior_evidence.method): per pitcher take the rate against
    each batter-hand code, form the difference D.  Under "every pitcher has the
    same true D" the observed spread of D is pure binomial noise, so

        true_var = Var(D) - E[ p1(1-p1)/n1 + p2(1-p2)/n2 ]
        reliability = true_var / Var(D)
        obs/noise   = sd(D) / sd(noise)
        eb_k        = pbar(1-pbar) / true_var

    Everything here is computed on seasons <= `max_season` ONLY, i.e. inside the
    fold's training window.  Nothing looks at the validation season.
    """
    season = df["season"].to_numpy()
    pit = df[PITCHER].to_numpy()
    hand = df[HAND].to_numpy()
    nums = numerators(df)
    m = ok & ~f_excl & (season <= max_season) & (succ_season <= max_season)

    # the two most common hand codes.  The competition never states which code
    # is left and which is right, and the feature definition does not depend on
    # it, so they are reported as code1 / code2 throughout.
    vc = pd.Series(hand[m]).value_counts()
    codes = sorted(vc.index[:2].tolist())
    cell, pitc = _agg(pit, hand, nums, m)

    out = {"max_season_used": int(max_season), "n_rows": int(m.sum()),
           "hand_codes_code1_code2": [int(c) for c in codes],
           "hand_code_counts": {str(int(k)): int(v) for k, v in vc.items()},
           "min_n_per_hand": min_n, "axes": {}}

    piv_n = cell["n"].unstack()
    for fname, (key, k_cfg) in RATE_SPEC.items():
        piv_s = cell[key].unstack()
        common = [c for c in codes if c in piv_n.columns]
        if len(common) < 2:
            continue
        n1 = piv_n[common[0]].to_numpy(dtype=np.float64)
        n2 = piv_n[common[1]].to_numpy(dtype=np.float64)
        s1 = piv_s[common[0]].to_numpy(dtype=np.float64)
        s2 = piv_s[common[1]].to_numpy(dtype=np.float64)
        keep = (n1 >= min_n) & (n2 >= min_n)
        n1, n2, s1, s2 = n1[keep], n2[keep], s1[keep], s2[keep]
        p1, p2 = s1 / n1, s2 / n2
        d = p1 - p2
        uncond = (s1 + s2) / (n1 + n2)

        obs_var = float(np.var(d, ddof=1))
        noise_var = float(np.mean(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
        true_var = max(obs_var - noise_var, 0.0)
        pbar = float(np.sum(nums[key][m]) / max(1, int(m.sum())))
        eb_k = (pbar * (1 - pbar) / true_var) if true_var > 0 else None

        out["axes"][fname] = {
            "n_pitchers": int(keep.sum()),
            "pbar": round(pbar, 6),
            "true_sd_pp": round(100.0 * float(np.sqrt(true_var)), 3),
            "obs_sd_pp": round(100.0 * float(np.sqrt(obs_var)), 3),
            "noise_sd_pp": round(100.0 * float(np.sqrt(noise_var)), 3),
            "obs_over_noise": round(float(np.sqrt(obs_var / noise_var)), 3),
            "reliability": round(true_var / obs_var, 3) if obs_var > 0 else None,
            "corr_with_unconditional": round(
                float(np.corrcoef(d, uncond)[0, 1]), 3),
            "eb_k_recomputed": round(eb_k, 1) if eb_k else None,
            "eb_k_config_used_for_features": k_cfg,
            "eb_k_rel_dev_vs_config": (round(abs(eb_k - k_cfg) / k_cfg, 3)
                                       if eb_k else None),
            "eb_k_dev_over_20pct": (bool(abs(eb_k - k_cfg) / k_cfg > 0.20)
                                    if eb_k else None),
            "mean_abs_delta_pp": round(100.0 * float(np.mean(np.abs(d))), 3),
        }

    if next_season is not None:
        rows = season == next_season
        mi = pd.MultiIndex.from_arrays([pit[rows], hand[rows]])
        hit = cell.reindex(mi)["n"].to_numpy(dtype=np.float64)
        out["row_coverage_next_season_pct"] = round(
            100.0 * float(np.mean(~np.isnan(hit))), 3)
        out["next_season"] = int(next_season)
    return out
