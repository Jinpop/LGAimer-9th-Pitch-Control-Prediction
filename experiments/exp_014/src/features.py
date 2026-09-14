"""exp_014 feature construction — shared by train.py and test_stub_check.py.

EVERY feature here is a pure ROW-WISE function of columns that exist in
test.csv's 48 columns.  No groupby, no sort, no aggregation, no reference to
any other row.  That is what makes rule 2-4 (no cross-row / no test-distribution
reference at inference) structurally impossible to violate, and it is verified
empirically by src/test_stub_check.py.

Missing-value policy (CLAUDE.md: no sentinel filling):
  a ratio whose denominator is exactly 0 (or whose inputs are NaN) becomes NaN
  and is handed to LightGBM as NaN.  Nothing is filled with 0 / -1 / a mean.
"""

import numpy as np

P_S = "asof_pitcher_success_rate"
P_R = "asof_pitcher_reverse_rate"
P_M = "asof_pitcher_middle_rate"
B_S = "asof_batter_success_rate"
B_M = "asof_batter_middle_rate"


def safe_div(num, den):
    """Element-wise num/den, NaN wherever den == 0 or either side is NaN.

    Deliberately NOT clipped and NOT filled: a pitcher with 3 career pitches
    genuinely has an undefined / wild failure-shape, and hiding that behind a
    sentinel would let the model read the sentinel as a value.
    """
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.full(num.shape, np.nan, dtype="float64")
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[ok] = num[ok] / den[ok]
    return out


def logit(v):
    """log(v/(1-v)); NaN outside the open interval (0,1) rather than clipped.

    A career share of exactly 0 or 1 only happens at tiny sample sizes; mapping
    it to +-13.8 would invent a value, so it stays missing.
    """
    v = np.asarray(v, dtype="float64")
    out = np.full(v.shape, np.nan, dtype="float64")
    ok = np.isfinite(v) & (v > 0.0) & (v < 1.0)
    out[ok] = np.log(v[ok] / (1.0 - v[ok]))
    return out


# --------------------------------------------------------------------------- #
# arm feature groups (cumulative: A1 = CORE_A1, A2 = CORE_A1 + CORE_A2, ...)
# --------------------------------------------------------------------------- #
CORE_A1 = ["fail_shape"]
CORE_A2 = ["residual"]
EXT_A3 = [
    "middle_share_of_fail",
    "recent_shape_prev1",
    "recent_shape_prev3",
    "recent_shape_prev5",
    "shape_drift",
]
EXT_A4 = ["batter_shape", "matchup_shape"]

ALL_NEW = CORE_A1 + CORE_A2 + EXT_A3 + EXT_A4

ARM_FEATURES = {
    "A0": [],
    "A1": CORE_A1,
    "A2": CORE_A1 + CORE_A2,
    "A3": CORE_A1 + CORE_A2 + EXT_A3,
    "A4": CORE_A1 + CORE_A2 + EXT_A3 + EXT_A4,
}


def add_shape_features(df):
    """Add every exp_014 derived column to `df` in place. Returns the df.

    fail_shape           reverse / (reverse + middle)      share of the *shaped*
                         failures that are 'wrong side' rather than 'over the
                         middle'.  Two pitchers with identical success_rate but
                         different fail_shape fail in different ways.
    residual             1 - (success + reverse + middle)  the ~0.10 that the
                         EDA identity leaves over: the rate of failures that are
                         neither reverse nor middle (a third failure mode).
    middle_share_of_fail middle / (1 - success)            middle's share of ALL
                         failures (denominator is the true failure rate, not the
                         reverse+middle subtotal).
    recent_shape_prevN   prevN_middle / (1 - prevN_success) the same shape over
                         the last N games -> current form.
    shape_drift          recent_shape_prev5 - middle_share_of_fail  is this
                         pitcher drifting towards leaving balls over the middle?
    batter_shape         batter_middle / (1 - batter_success)  the batter side
                         (test.csv has no batter reverse rate, so only the
                         middle share is constructible).
    matchup_shape        logit(middle_share_of_fail) + logit(batter_shape) —
                         additive-in-log-odds interaction of the two shapes.
    """
    p_s = df[P_S].to_numpy(dtype="float64")
    p_r = df[P_R].to_numpy(dtype="float64")
    p_m = df[P_M].to_numpy(dtype="float64")

    df["fail_shape"] = safe_div(p_r, p_r + p_m)
    df["residual"] = 1.0 - (p_s + p_r + p_m)

    df["middle_share_of_fail"] = safe_div(p_m, 1.0 - p_s)
    for n in (1, 3, 5):
        s = df[f"asof_pitcher_prev{n}_game_success_rate"].to_numpy(dtype="float64")
        m = df[f"asof_pitcher_prev{n}_game_middle_rate"].to_numpy(dtype="float64")
        df[f"recent_shape_prev{n}"] = safe_div(m, 1.0 - s)
    df["shape_drift"] = (
        df["recent_shape_prev5"].to_numpy() - df["middle_share_of_fail"].to_numpy()
    )

    b_s = df[B_S].to_numpy(dtype="float64")
    b_m = df[B_M].to_numpy(dtype="float64")
    df["batter_shape"] = safe_div(b_m, 1.0 - b_s)
    df["matchup_shape"] = logit(df["middle_share_of_fail"].to_numpy()) + logit(
        df["batter_shape"].to_numpy()
    )
    return df
