"""exp_014 rule 2-4 check — the derived features must be reproducible at test
time from a SINGLE row, with no reference to any other test row or to the test
distribution.

Three things are asserted:
  1. every source column the derivation touches exists in test.csv's 48 columns;
  2. computing the features on the whole 5-row stub gives bit-identical results
     to computing them one row at a time (row independence);
  3. the same holds after shuffling the row order (no order dependence).

If any of these failed the feature would be inadmissible under rule 2-4.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ALL_NEW, add_shape_features, P_S, P_R, P_M, B_S, B_M  # noqa

TEST_PATH = os.environ.get(
    "EXP014_TEST", os.path.expanduser("~/SungjinPark/data/test.csv"))

SOURCE_COLS = [P_S, P_R, P_M, B_S, B_M] + [
    f"asof_pitcher_prev{n}_game_{k}_rate"
    for n in (1, 3, 5) for k in ("success", "middle")
]


def main():
    t = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    out = {"test_path": TEST_PATH, "test_shape": list(t.shape),
           "n_test_columns": int(t.shape[1])}

    missing = [c for c in SOURCE_COLS if c not in t.columns]
    out["source_columns_required"] = SOURCE_COLS
    out["source_columns_missing_from_test"] = missing
    assert not missing, f"test.csv is missing required source columns: {missing}"

    full = add_shape_features(t.copy())[ALL_NEW].to_numpy(dtype="float64")

    per_row = np.vstack([
        add_shape_features(t.iloc[[i]].copy())[ALL_NEW].to_numpy(dtype="float64")
        for i in range(len(t))
    ])
    d_row = np.nanmax(np.abs(np.nan_to_num(full, nan=-9e9)
                             - np.nan_to_num(per_row, nan=-9e9)))
    nan_same_row = bool((np.isnan(full) == np.isnan(per_row)).all())

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(t))
    shuf = add_shape_features(t.iloc[perm].copy())[ALL_NEW].to_numpy(dtype="float64")
    inv = np.argsort(perm)
    shuf = shuf[inv]
    d_shuf = np.nanmax(np.abs(np.nan_to_num(full, nan=-9e9)
                              - np.nan_to_num(shuf, nan=-9e9)))
    nan_same_shuf = bool((np.isnan(full) == np.isnan(shuf)).all())

    out["max_abs_diff_batch_vs_single_row"] = float(d_row)
    out["nan_pattern_identical_single_row"] = nan_same_row
    out["max_abs_diff_batch_vs_shuffled"] = float(d_shuf)
    out["nan_pattern_identical_shuffled"] = nan_same_shuf
    out["values"] = {
        c: [None if not np.isfinite(v) else round(float(v), 6)
            for v in full[:, i]]
        for i, c in enumerate(ALL_NEW)
    }
    out["passed"] = bool(d_row == 0.0 and d_shuf == 0.0
                         and nan_same_row and nan_same_shuf)

    assert out["passed"], out
    print(json.dumps(out, indent=2, ensure_ascii=False))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "test_stub_check.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
