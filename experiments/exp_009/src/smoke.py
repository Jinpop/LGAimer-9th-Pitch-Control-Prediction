"""Smoke test for exp_009: exercises build_model's passthrough-NaN branch and
fit_arm end to end on synthetic data, so a typo in the P4 branch does not
surface 30 minutes into the real run. Not part of the experiment."""

import numpy as np
import pandas as pd

import train as T

rng = np.random.default_rng(0)
n = 4000
X = pd.DataFrame({
    "top_bottom": rng.choice(["T", "B"], n),
    "game_type": rng.choice(["R", "F"], n),
    "base_state": rng.choice(["000", "100", "110"], n),
    "num_a": rng.normal(size=n),
    "num_b": rng.normal(size=n),
    "career_n_pitches": rng.normal(size=n),
    "recent_n_pitches": rng.normal(size=n),
    "last_season_gap": rng.integers(0, 3, n).astype(float),
})
miss = rng.random(n) < 0.2
for c in ("career_n_pitches", "recent_n_pitches", "last_season_gap"):
    X.loc[miss, c] = np.nan
X[T.FLAG_PROFILE] = X["career_n_pitches"].notna().astype("int8")
X[T.FLAG_RECENT] = X["recent_n_pitches"].notna().astype("int8")
y = (X["num_a"] + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()

prof_cols = ["career_n_pitches", "recent_n_pitches", "last_season_gap"]
is_r = (X["game_type"] == "R").to_numpy()
cold = rng.random(n) < 0.2
meta = (is_r, cold, X[T.FLAG_PROFILE].to_numpy().astype(bool),
        X[T.FLAG_RECENT].to_numpy().astype(bool))

for arm, raw, mf in [("smokeP2", [], None), ("smokeP0b", [], 3), ("smokeP4", prof_cols, None)]:
    cols = list(X.columns)
    out = T.fit_arm(arm, "fold_smoke", X, y, X, y, meta, cols, raw, mf)
    assert out["valid_R"]["n"] > 0 and out["profile_importance_share"] >= 0
    print(arm, "OK  brier", out["valid_R"]["brier"],
          "raw_nan", out["n_raw_nan_features"],
          "max_feat", out["max_features_effective"])
print("SMOKE PASS")
