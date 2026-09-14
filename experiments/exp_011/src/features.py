"""exp_011 -- empirical-Bayes shrinkage of the organizers' as-of rate features.

This module is the SINGLE source of truth for the feature transform.  Both the
training script (`train.py`) and the inference stub (`infer_stub.py`) import
`build_features` from here, so the train-time and test-time computations are
literally the same code path -- that is the reproducibility guarantee demanded
by the spec.

------------------------------------------------------------------------------
Why shrinkage
------------------------------------------------------------------------------
`asof_pitcher_success_rate` and friends are RAW ratios.  Their denominators
(`asof_pitcher_n`, `asof_batter_n`, `asof_pitcher_pitchmix_n`) span 0 .. 6000+,
so a rookie's 0.29 on n=86 and a veteran's 0.489 on n=3465 are handed to the
model on the same scale even though the first is mostly noise.  EDA section 0
verified `rate * n == sum of the past target` exactly, so the numerator can be
recovered and the posterior mean under a Beta(k*prior, k*(1-prior)) prior is

    shrunk = (rate * n + k * prior) / (n + k)

A depth-limited random forest cannot synthesise this from (rate, n) on its own:
it would need a deep rate x n interaction in every region of the tree.  Handing
it the combination directly is the whole hypothesis.

------------------------------------------------------------------------------
Where the prior comes from, and why it is leak-free
------------------------------------------------------------------------------
PRIOR TABLE: one scalar per (mode, feature, season).  `prior[m][f][s]` is the
league-pooled value of feature `f` estimated from rows with `season < s` ONLY.

  * A training row of season t uses seasons < t                    -> causal.
  * A validation row of season V uses seasons < V, i.e. exactly the fold's
    training window -> no valid-year information and, crucially, no *within*-
    year updating, which is what test time will look like.
  * A test row of season 2025 uses seasons <= 2024 = the whole training file ->
    a constant, shipped in `priors.json`.  Nothing about test is consulted.

Because entry `s` only ever reads seasons `< s`, ONE table computed from the
full training file is simultaneously correct for every rolling-origin fold.

Three prior modes are built; one is selected on fold_2023 (never on 2024):
  cum     pooled value over all seasons < s          (stable, lags the trend)
  last    pooled value over season s-1 only          (tracks the trend, noisier)
  trend3  log-odds linear fit over the last <=3 seasons < s, extrapolated to s
The base rate falls about 1pp/season (EDA section 1.4), so `cum` is biased high
for the most recent season -- that is exactly what `last`/`trend3` test.

The single approximation: `prior[*][2019]` has no earlier season and falls back
to season 2019 itself.  2019 is training-only in every fold (valid years are
2022/2023/2024) and the value is one global scalar over ~285k rows, so it
cannot carry per-row target information into any validation set.

`build_features` itself is strictly ROW-WISE: every operation is an elementwise
numpy expression over that row's own columns plus constants from the prior
table.  No groupby, no sort, no rolling, no reference to any other row.  That
is what makes it safe under competition rule 2-4 (an inference pass may not
look at other test rows or at the test distribution).  `infer_stub.py` proves
it numerically by re-running the transform one row at a time.
"""

import json

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #

TARGET = "control_success"

# (short name, rate column, denominator column, prior kind, family)
#   kind "target" -> the rate measures the target itself, so the pooled league
#                    value IS the base rate and can be computed directly.
#   kind "pooled" -> the underlying event column does not exist in train.csv;
#                    the league value is recovered from the per-entity (rate, n)
#                    pairs (see `_window_totals`).
RATE_SPECS = [
    ("p_success", "asof_pitcher_success_rate", "asof_pitcher_n", "target", "pitcher"),
    ("p_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n", "pooled", "pitcher"),
    ("p_middle", "asof_pitcher_middle_rate", "asof_pitcher_n", "pooled", "pitcher"),
    ("p_ball", "asof_pitcher_ball_rate", "asof_pitcher_n", "pooled", "pitcher"),
    ("p_strike", "asof_pitcher_strike_rate", "asof_pitcher_n", "pooled", "pitcher"),
    ("b_success", "asof_batter_success_rate", "asof_batter_n", "target", "batter"),
    ("b_middle", "asof_batter_middle_rate", "asof_batter_n", "pooled", "batter"),
    ("mix_fastball", "asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", "pooled", "mix"),
    ("mix_breaking", "asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", "pooled", "mix"),
    ("mix_offspeed", "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "pooled", "mix"),
]

RAW_RATE_COLS = [spec[1] for spec in RATE_SPECS]
FAMILIES = ["pitcher", "batter", "mix"]
PRIOR_MODES = ("cum", "last", "trend3")

# The as-of counters are career-cumulative with no season reset (EDA section 0),
# so inside any window the row where an entity's counter is maximal carries that
# entity's whole event total for the window.
ENTITY_OF_DENOM = {
    "asof_pitcher_n": "pitcher_id",
    "asof_batter_n": "batter_id",
    "asof_pitcher_pitchmix_n": "pitcher_id",
}

PREV_COLS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]

SHRUNK_COLS = [f"sh_{name}" for name, *_ in RATE_SPECS]
DERIVED_COLS = [
    "eb_matchup_logodds",
    "eb_p_fail",
    "eb_prev1_success_dev",
    "eb_prev3_success_dev",
    "eb_prev5_success_dev",
    "eb_prev1_middle_dev",
    "eb_prev_game_missing",
]

EPS = 1e-6
CLIP_LO, CLIP_HI = 1e-4, 1.0 - 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
# Prior table
# --------------------------------------------------------------------------- #


def _window_totals(win):
    """(numerator, denominator) league totals for every rate spec over `win`.

    For "target" specs this is exact (sum of the label, row count).  For
    "pooled" specs each entity's max-counter row inside the window carries that
    entity's running total, so summing `rate * n` over those rows recovers the
    league numerator.  At most one pitch per entity is lost to the as-of
    shift(1) (~800 of >1e6 rows) -- below the 6th decimal of the result.
    """
    totals = {}
    if len(win) == 0:
        return {name: (0.0, 0.0) for name, *_ in RATE_SPECS}

    for name, _rate, _denom, kind, _fam in RATE_SPECS:
        if kind == "target":
            totals[name] = (float(win[TARGET].sum()), float(len(win)))

    # group the pooled specs by denominator so each entity idxmax is done once
    by_denom = {}
    for name, rate_col, denom_col, kind, _fam in RATE_SPECS:
        if kind == "pooled":
            by_denom.setdefault(denom_col, []).append((name, rate_col))

    for denom_col, specs in by_denom.items():
        entity = ENTITY_OF_DENOM[denom_col]
        cols = [entity, denom_col] + [rc for _n, rc in specs]
        sub = win.loc[win[denom_col] > 0, cols]
        if sub.empty:
            for name, _rc in specs:
                totals[name] = (0.0, 0.0)
            continue
        last = sub.loc[sub.groupby(entity)[denom_col].idxmax()]
        n_vec = last[denom_col].to_numpy(dtype=float)
        for name, rate_col in specs:
            r_vec = last[rate_col].to_numpy(dtype=float)
            ok = np.isfinite(r_vec)
            totals[name] = (
                float(np.sum(r_vec[ok] * n_vec[ok])),
                float(np.sum(n_vec[ok])),
            )
    return totals


def _safe_ratio(num, den):
    return float(num / den) if den and den > 0 else float("nan")


def _trend_extrapolate(seasons, rates, target_season, max_points=3):
    """Log-odds linear extrapolation over the last `max_points` seasons."""
    pairs = [(s, r) for s, r in zip(seasons, rates) if np.isfinite(r) and 0 < r < 1]
    if not pairs:
        return float("nan")
    pairs = pairs[-max_points:]
    if len(pairs) == 1:
        return float(pairs[0][1])
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = _logit(np.array([p[1] for p in pairs], dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    z = slope * float(target_season) + intercept
    return float(np.clip(1.0 / (1.0 + np.exp(-z)), CLIP_LO, CLIP_HI))


def build_prior_table(train_df, seasons_out):
    """Build every prior mode from `train_df`, serving the seasons in `seasons_out`.

    Returns {"prior": {mode: {name: {season_str: value}}}, "windows": ..., "per_season": ...}
    """
    seasons_avail = sorted(int(s) for s in train_df["season"].unique())
    first_season = seasons_avail[0]
    seasons_need = sorted(set(int(x) for x in seasons_out) | set(seasons_avail))
    # cutoffs: every season boundary we may have to read, plus one past the end
    cutoffs = sorted(set(seasons_need) | {max(seasons_avail) + 1})

    # cumulative totals over {season < c}
    cum = {}
    windows = {}
    for c in cutoffs:
        win = train_df.loc[train_df["season"] < c]
        cum[c] = _window_totals(win)
        windows[str(c)] = {
            "desc": f"season < {c}",
            "n_rows": int(len(win)),
            "league_target_mean": (
                round(float(win[TARGET].mean()), 8) if len(win) else None
            ),
        }

    # season-t-only totals by differencing consecutive cumulative windows
    per_season = {name: {} for name, *_ in RATE_SPECS}
    for t in seasons_avail:
        if t not in cum or (t + 1) not in cum:
            continue
        for name, *_ in RATE_SPECS:
            n0, d0 = cum[t][name]
            n1, d1 = cum[t + 1][name]
            per_season[name][str(t)] = _safe_ratio(n1 - n0, d1 - d0)

    # fallback for the very first season: that season's own value
    fallback = {}
    win0 = train_df.loc[train_df["season"] == first_season]
    tot0 = _window_totals(win0)
    for name, *_ in RATE_SPECS:
        fallback[name] = _safe_ratio(*tot0[name])

    table = {m: {name: {} for name, *_ in RATE_SPECS} for m in PRIOR_MODES}
    for s in seasons_need:
        earlier = [t for t in seasons_avail if t < s]
        for name, *_ in RATE_SPECS:
            if not earlier:
                v_cum = v_last = v_trend = fallback[name]
            else:
                v_cum = _safe_ratio(*cum[s][name]) if s in cum else _safe_ratio(*cum[max(cutoffs)][name])
                v_last = per_season[name].get(str(earlier[-1]), float("nan"))
                v_trend = _trend_extrapolate(
                    earlier, [per_season[name].get(str(t), float("nan")) for t in earlier], s
                )
                if not np.isfinite(v_last):
                    v_last = v_cum
                if not np.isfinite(v_trend):
                    v_trend = v_cum
            table["cum"][name][str(s)] = float(v_cum)
            table["last"][name][str(s)] = float(v_last)
            table["trend3"][name][str(s)] = float(v_trend)

    # ---- game_type-split table (target-kind specs only) --------------------
    # Rationale / limitation.  The F league had a documented label-regime break
    # in 2023 (EDA sec.1.4: F base rate 0.709 -> 0.473), so an F row's league
    # prior arguably should not be the R-dominated pooled value.  It can be
    # computed EXACTLY only for the two "target"-kind specs, where the league
    # value is a direct mean of the label over the restricted window.  For the
    # five "pooled" specs the numerator is recovered from each entity's max-
    # counter row, and those counters are career-cumulative across BOTH leagues
    # (EDA sec.0) -- restricting the window to one game_type would divide an
    # R+F numerator by an F-only denominator.  Those specs therefore keep the
    # combined prior under gt_split, which is also the self-consistent choice:
    # the quantity being shrunk is itself an R+F career rate.
    table_gt = {m: {name: {} for name, *_ in RATE_SPECS} for m in PRIOR_MODES}
    gtypes = sorted(str(g) for g in train_df["game_type"].unique())
    for g in gtypes:
        gdf = train_df.loc[train_df["game_type"] == g]
        cum_g, per_g = {}, {}
        for c in cutoffs:
            w = gdf.loc[gdf["season"] < c]
            cum_g[c] = (float(w[TARGET].sum()), float(len(w)))
        for t in seasons_avail:
            if t in cum_g and (t + 1) in cum_g:
                per_g[t] = _safe_ratio(cum_g[t + 1][0] - cum_g[t][0],
                                       cum_g[t + 1][1] - cum_g[t][1])
        w0 = gdf.loc[gdf["season"] == first_season]
        fb_g = _safe_ratio(float(w0[TARGET].sum()), float(len(w0)))
        for s in seasons_need:
            earlier = [t for t in seasons_avail if t < s]
            for name, _rc, _dc, kind, _fam in RATE_SPECS:
                key = f"{s}|{g}"
                if kind != "target":
                    for m in PRIOR_MODES:  # combined value, see note above
                        table_gt[m][name][key] = table[m][name][str(s)]
                    continue
                if not earlier:
                    v_cum = v_last = v_trend = fb_g
                else:
                    v_cum = _safe_ratio(*cum_g[s])
                    v_last = per_g.get(earlier[-1], float("nan"))
                    v_trend = _trend_extrapolate(
                        earlier, [per_g.get(t, float("nan")) for t in earlier], s
                    )
                    if not np.isfinite(v_last):
                        v_last = v_cum
                    if not np.isfinite(v_trend):
                        v_trend = v_cum
                table_gt["cum"][name][key] = float(v_cum)
                table_gt["last"][name][key] = float(v_last)
                table_gt["trend3"][name][key] = float(v_trend)

    return {
        "prior": table,
        "prior_gt": table_gt,
        "game_types": gtypes,
        "windows": windows,
        "per_season_rate": {k: v for k, v in per_season.items()},
    }


def save_priors(priors, path):
    with open(path, "w") as f:
        json.dump(priors, f, indent=2, sort_keys=True)


def load_priors(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Row-wise feature construction
# --------------------------------------------------------------------------- #


def _prior_vector(priors, mode, name, season_values, game_types=None):
    """Map each row's (season[, game_type]) to its prior scalar. Per-row lookup."""
    if game_types is None:
        tbl = priors["prior"][mode][name]
        keys = pd.Series(np.asarray(season_values)).astype(int).astype(str)
        latest = str(max(int(k) for k in tbl))
    else:
        tbl = priors["prior_gt"][mode][name]
        s = pd.Series(np.asarray(season_values)).astype(int).astype(str)
        g = pd.Series(np.asarray(game_types)).astype(str)
        keys = s.str.cat(g, sep="|")
        # unseen (season, game_type): fall back to the latest season of the same
        # league, else the latest key overall
        latest = None
    out = keys.map(tbl)
    if out.isna().any():
        if game_types is None:
            out = out.fillna(tbl[latest])
        else:
            gg = pd.Series(np.asarray(game_types)).astype(str)
            per_g = {}
            for g_val in gg.unique():
                same = [k for k in tbl if k.endswith(f"|{g_val}")]
                if same:
                    per_g[g_val] = tbl[max(same, key=lambda k: int(k.split("|")[0]))]
                else:
                    per_g[g_val] = tbl[max(tbl, key=lambda k: int(k.split("|")[0]))]
            out = out.fillna(gg.map(per_g))
    return out.to_numpy(dtype=float)


def _shrink(rate, n, prior, k):
    """(events + k*prior) / (n + k).  n == 0 or a missing rate -> the prior.

    This is a posterior mean, not a sentinel fill: with zero observations the
    Beta(k*prior, k*(1-prior)) posterior mean IS the prior.  Rows in that state
    are still identifiable downstream because `asof_*_n` stays in the feature
    matrix (and `eb_prev_game_missing` flags the prev-game family, which has no
    counter column of its own).
    """
    r = np.asarray(rate, dtype=float)
    nn = np.asarray(n, dtype=float)
    ok = np.isfinite(r) & np.isfinite(nn) & (nn > 0)
    # probe.py P1: rate * n lands within 7.7e-3 of an integer for 100% of rows
    # (the CSV stores the ratio to ~6 significant digits), so rounding recovers
    # the exact event count rather than merely approximating it.
    events = np.where(ok, np.round(np.nan_to_num(r) * np.nan_to_num(nn)), 0.0)
    denom_n = np.where(ok, nn, 0.0)
    return (events + k * prior) / (denom_n + k)


def build_features(df, priors, k_by_family, mode="cum", add_derived=True, gt_split=False):
    """Return (new_df, added_cols).  Strictly row-wise.

    df          : any frame carrying the 48 test.csv columns (train or test)
    priors      : dict from build_prior_table / priors.json
    k_by_family : {"pitcher": k, "batter": k, "mix": k} shrinkage strengths
    mode        : prior mode, one of PRIOR_MODES
    add_derived : also build the matchup / form-deviation block
    gt_split    : look the prior up per (season, game_type) instead of season
    """
    if mode not in PRIOR_MODES:
        raise ValueError(f"unknown prior mode {mode!r}")
    out = df.copy()
    season = out["season"].to_numpy()
    gtv = out["game_type"].to_numpy() if gt_split else None
    added = []
    prior_vecs = {}

    for name, rate_col, denom_col, _kind, fam in RATE_SPECS:
        pri = _prior_vector(priors, mode, name, season, gtv)
        prior_vecs[name] = pri
        col = f"sh_{name}"
        out[col] = _shrink(
            out[rate_col].to_numpy(), out[denom_col].to_numpy(), pri, float(k_by_family[fam])
        )
        added.append(col)

    if add_derived:
        # Matchup prior on the log-odds scale (Log5 / odds-ratio form): pitcher
        # skill + batter skill minus the league baseline both already contain.
        # A depth-10 tree cannot assemble this 3-term combination from marginals.
        out["eb_matchup_logodds"] = (
            _logit(out["sh_p_success"].to_numpy())
            + _logit(out["sh_b_success"].to_numpy())
            - _logit(prior_vecs["p_success"])
        )
        # EDA section 1.1: success + reverse + middle ~= 0.90, corr(success,
        # reverse) = -0.874 -> the two failure modes summed are the sharpest
        # single failure proxy available.
        out["eb_p_fail"] = out["sh_p_reverse"].to_numpy() + out["sh_p_middle"].to_numpy()
        # Recent form RELATIVE to the pitcher's own stabilised career level.
        for tag, src in [
            ("eb_prev1_success_dev", "asof_pitcher_prev1_game_success_rate"),
            ("eb_prev3_success_dev", "asof_pitcher_prev3_game_success_rate"),
            ("eb_prev5_success_dev", "asof_pitcher_prev5_game_success_rate"),
        ]:
            out[tag] = out[src].to_numpy() - out["sh_p_success"].to_numpy()
        out["eb_prev1_middle_dev"] = (
            out["asof_pitcher_prev1_game_middle_rate"].to_numpy()
            - out["sh_p_middle"].to_numpy()
        )
        # Explicit missingness flag: the base pipeline median-imputes the prev
        # columns, which destroys "this pitcher is inside his first game".
        # Project rule: never fill a missing value without a separate flag.
        out["eb_prev_game_missing"] = (
            out["asof_pitcher_prev1_game_success_rate"].isna().to_numpy().astype(float)
        )
        added += DERIVED_COLS

    return out, added


# NOTE on the prev{1,3,5}-game rates -----------------------------------------
# The spec listed them as shrinkage targets.  They have NO denominator column,
# so the numerator cannot be recovered.  Two fallbacks were considered:
#   (a) infer the denominator from the decimal expansion (smallest b for which
#       rate*b is integral).  `probe.py` P4 measures how often this is even
#       identifiable; it is systematically biased for reducible fractions
#       (0.5 coming from 15/30 infers b=2), which would over-shrink exactly the
#       rows the shrinkage is meant to fix.
#   (b) assume a constant effective n per window.  Then
#           shrunk = (n_eff*rate + k*prior) / (n_eff + k)
#       is an AFFINE map of `rate` with a per-season intercept -- for a tree
#       ensemble that is a no-op up to the season term, i.e. no new information.
# Neither is used as a feature.  The `eb_prev*_dev` deviations above are the
# substantive alternative: they are differences against a *shrunk* baseline,
# which no monotone transform of the raw column can reproduce.
#
# NOTE on the "experience bucket" feature the spec also listed ---------------
# A bucketised `asof_pitcher_n` is a monotone step function of a column that is
# ALREADY in the base 47 features, so for an axis-aligned tree ensemble it adds
# exactly nothing (any threshold on the bucket is reachable as a threshold on
# n).  The same holds for log1p(n) and for the reliability weight n/(n+k).  The
# non-trivial part of "experience" is the rate x n interaction, and that is
# precisely what `sh_*` already encodes.  Omitted deliberately; see report.
