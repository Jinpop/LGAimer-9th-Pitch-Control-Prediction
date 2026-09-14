"""exp_016 — row-level failure-MODE label RECOVERY (design decision D1).

WHAT THIS FILE DOES
-------------------
The organisers never hand out a per-row failure-mode label.  They hand out
per-pitcher *cumulative* rates, as of the row (`asof_pitcher_{success,reverse,
middle,ball,strike}_rate`) together with the cumulative denominator
`asof_pitcher_n`.  EDA §0 verified on all 12 sampled pitchers that

    asof_pitcher_success_rate[t] * asof_pitcher_n[t] == sum(control_success[<t])

exactly, and that `asof_pitcher_n` increases by exactly +1 per row within a
pitcher in FILE ORDER (100% of rows).  Therefore the cumulative event count is

    cnt_X[t] = round(rate_X[t] * n[t])          (0 when n[t] == 0)

and the event on row t is recovered by differencing with the SAME pitcher's
NEXT row in file order:

    is_X[t] = cnt_X[t+1] - cnt_X[t]     in {0, 1}

======================================================================
!!  LEAK BOUNDARY  --  READ BEFORE TOUCHING ANYTHING IN THIS FILE  !!
======================================================================
The recovered `is_*` columns use row t+1.  They are therefore FUTURE
information relative to row t.

  * They are legal as a TRAINING TARGET.  A target is only ever consumed at
    fit time; at inference the model is handed 47 provided columns and emits a
    probability.  Nothing recovered here is ever an input.
  * They are ILLEGAL as a FEATURE.  Putting any `is_*` (or anything derived
    from it, including its own lag) into the design matrix injects the future
    into the present and invalidates the experiment.

`assert_no_recovered_in_features()` below is the machine-checked form of that
boundary and is called from train.py before every single lgb.Dataset build.
======================================================================

Limits (stated in config.yaml D1):
  * The LAST row of every pitcher has no successor -> unrecoverable (792 rows).
  * The provided rate columns are rounded, so `rate * n` is not exactly an
    integer.  The magnitude of that slack, and the share of rows whose diff
    falls outside {0, 1}, are measured by the D2 gate (src/gate.py) and are
    reported whatever they turn out to be.
"""

import numpy as np
import pandas as pd

# Modes the organisers give a cumulative rate for.  `success` is recovered too,
# purely so it can be checked against the ground-truth `control_success`
# column -- that check is the whole reason to believe the other three.
RATE_COLS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}

# Everything this module can ever put on the dataframe.  train.py refuses to
# build a Dataset whose columns intersect this set.
RECOVERED_PREFIXES = ("is_", "cnt_", "cntnext_", "draw_", "recov_")
RECOVERED_COLS = (
    [f"is_{k}" for k in RATE_COLS]
    + [f"cnt_{k}" for k in RATE_COLS]
    + [f"cntnext_{k}" for k in RATE_COLS]
    + [f"draw_{k}" for k in RATE_COLS]
    + ["is_wayoff", "is_mid_only", "is_rev_only", "is_both", "is_wayoff_true",
       "recov_ok", "recov_has_next", "mode_class"]
)

# The five mutually exclusive cells the models are trained on.  Order matters:
# index 0 is `success`, which is the class the competition scores.
CELLS = ["is_success", "is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
FAIL_CELLS = CELLS[1:]

PITCHER = "pitcher_id"
N_COL = "asof_pitcher_n"
TARGET = "control_success"


def recover(df, verbose=True):
    """Add the recovered per-row event columns to `df` IN PLACE.

    Returns a diagnostics dict (consumed by both gate.py and train.py so the
    two can never disagree about which rows are usable).

    Assumes `df` is in the organisers' file order, which EDA §0 established is
    the exact time axis the asof columns were built on.
    """
    diag = {}

    # ---- the +1 assumption, re-verified here rather than trusted ---------- #
    g = df.groupby(PITCHER, sort=False)
    n = df[N_COL].to_numpy(dtype=np.float64)
    n_next = g[N_COL].shift(-1).to_numpy(dtype=np.float64)
    has_next = ~np.isnan(n_next)
    step = n_next[has_next] - n[has_next]
    diag["n_step_all_plus_one"] = bool(np.all(step == 1.0))
    diag["n_step_violations"] = int(np.sum(step != 1.0))
    diag["n_rows"] = int(len(df))
    diag["n_rows_without_successor"] = int((~has_next).sum())
    diag["asof_pitcher_n_max"] = int(np.nanmax(n))

    # ---- cumulative counts, then the difference --------------------------- #
    # rate is NaN exactly when n == 0 (EDA §2: 792 rows, the first pitch of
    # each pitcher).  count is 0 there by definition, not by imputation --
    # this is arithmetic, not a sentinel fill.
    max_frac = {}
    for key, col in RATE_COLS.items():
        raw = df[col].to_numpy(dtype=np.float64) * n
        raw = np.where(n == 0, 0.0, raw)
        cnt = np.round(raw)
        # how far the provided (rounded) rate puts us from an integer count
        max_frac[key] = float(np.nanmax(np.abs(raw - cnt)))
        df[f"cnt_{key}"] = cnt
    diag["max_abs_distance_from_integer_count"] = max_frac

    for key in RATE_COLS:
        nxt = g[f"cnt_{key}"].shift(-1).to_numpy(dtype=np.float64)
        d = nxt - df[f"cnt_{key}"].to_numpy(dtype=np.float64)
        df[f"cntnext_{key}"] = nxt
        df[f"draw_{key}"] = d                       # may be NaN / 2 / -1 ...
        df[f"is_{key}"] = d                          # cleaned below

    # ---- per-row validity ------------------------------------------------- #
    ok = has_next.copy()
    bad_counts = {}
    for key in RATE_COLS:
        d = df[f"draw_{key}"].to_numpy(dtype=np.float64)
        in01 = np.isin(d, [0.0, 1.0])
        bad_counts[key] = int(np.sum(has_next & ~in01))
        ok &= in01
    diag["rows_with_diff_outside_01_by_mode"] = bad_counts
    diag["rows_with_diff_outside_01_any_pct"] = round(
        100.0 * float(np.sum(has_next & ~ok) / max(1, has_next.sum())), 4)

    # ---- does the recovered success match the ground truth? --------------- #
    y = df[TARGET].to_numpy(dtype=np.float64)
    rec_s = df["draw_success"].to_numpy(dtype=np.float64)
    cmp_mask = has_next & np.isin(rec_s, [0.0, 1.0])
    agree = rec_s[cmp_mask] == y[cmp_mask]
    diag["success_recovery"] = {
        "n_compared": int(cmp_mask.sum()),
        "n_agree": int(agree.sum()),
        "agree_pct": round(100.0 * float(agree.mean()), 6) if cmp_mask.sum() else None,
    }

    df["recov_has_next"] = has_next
    df["recov_ok"] = ok
    for key in RATE_COLS:
        df[f"is_{key}"] = np.where(ok, df[f"draw_{key}"], np.nan)

    # ---- way-off (failure case 2) is the arithmetic complement ------------ #
    # config D1: is_wayoff = 1 - is_success - is_middle - is_reverse, valid
    # only if the three modes are mutually exclusive and exhaustive.  That
    # assumption is exactly what gate item (3) tests; the column is created
    # regardless so the gate can measure how often it comes out negative.
    df["is_wayoff"] = (1.0 - df["is_success"] - df["is_middle"]
                       - df["is_reverse"])

    # ---- the partition that the gate actually measured -------------------- #
    # D2 gate result (gate.json): success never co-occurs with middle or
    # reverse (0 rows out of 1,474,300), but middle and reverse co-occur on
    # 50,266 rows (3.41%).  So the three modes are NOT mutually exclusive and
    # `is_wayoff` above is NOT a rate -- it is -1 on every overlap row.
    #
    # The five cells below ARE mutually exclusive and exhaustive, so they are
    # what the models are actually trained on:
    #   success | middle-only | reverse-only | middle&reverse | way-off
    # `is_wayoff_true` is the honest case-2 indicator ("missed the zone by a
    # lot"): a failure that is neither middle nor reverse.
    s_ = df["is_success"]
    m_ = df["is_middle"]
    r_ = df["is_reverse"]
    df["is_mid_only"] = ((m_ == 1) & (r_ == 0)).astype(float).where(ok)
    df["is_rev_only"] = ((r_ == 1) & (m_ == 0)).astype(float).where(ok)
    df["is_both"] = ((m_ == 1) & (r_ == 1)).astype(float).where(ok)
    df["is_wayoff_true"] = (
        (s_ == 0) & (m_ == 0) & (r_ == 0)).astype(float).where(ok)

    part = (df.loc[ok, ["is_success", "is_mid_only", "is_rev_only", "is_both",
                        "is_wayoff_true"]].to_numpy(dtype=np.float64).sum(axis=1))
    diag["partition_sums_to_one"] = bool(np.all(part == 1.0))
    diag["partition_violations"] = int(np.sum(part != 1.0))
    assert diag["partition_sums_to_one"], (
        "the five cells are not a partition -- everything downstream assumes "
        "they are")

    if verbose:
        print("[labels] recovery diagnostics:", flush=True)
        for k, v in diag.items():
            print(f"  {k}: {v}", flush=True)
    return diag


def assert_no_recovered_in_features(feat_cols, where=""):
    """Machine-checked leak boundary.  Called before every Dataset build.

    A recovered column reaching the design matrix is not a degradation, it is
    a wrong experiment: it hands the model row t+1's outcome.
    """
    bad = [c for c in feat_cols
           if c in RECOVERED_COLS or c.startswith(RECOVERED_PREFIXES)]
    assert not bad, (
        f"LEAK BOUNDARY VIOLATED at {where}: recovered/derived-from-future "
        f"columns found in the feature list: {bad}.  These are built from "
        f"row t+1 and may only ever be used as a TARGET."
    )
    return True
