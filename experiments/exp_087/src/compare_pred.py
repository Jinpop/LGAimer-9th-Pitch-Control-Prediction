#!/usr/bin/env python3
"""exp_087 부가 측정 — how far apart the n-seed and 3-seed F branches actually put
the SUBMITTED number.

Both sides are produced by running the shipped `script.py` on the same frame, so
this compares the deliverable through the deployed code path rather than through
a reimplementation of the branch arithmetic.  The F branch reaches only
`game_type=='F'` rows (weight 0.5) and does not touch R rows at all, so the R
rows are reported separately: they are the control, and any movement there would
mean something other than the F seeds moved.
"""
import json, sys
import numpy as np
import pandas as pd


def main(csv_a, csv_b, test_csv, out):
    a = pd.read_csv(csv_a, float_precision="round_trip")
    b = pd.read_csv(csv_b, float_precision="round_trip")
    assert list(a.columns) == list(b.columns), (a.columns, b.columns)
    idc, pc = a.columns[0], a.columns[1]
    assert (a[idc].to_numpy() == b[idc].to_numpy()).all(), "row ids differ"
    pa = a[pc].to_numpy(dtype=np.float64)
    pb = b[pc].to_numpy(dtype=np.float64)
    t = pd.read_csv(test_csv, usecols=[idc, "game_type"], float_precision="round_trip")
    t = t.set_index(idc).loc[a[idc]].reset_index()
    isF = (t["game_type"].to_numpy() == "F")
    d = pb - pa
    res = {"a": csv_a, "b": csv_b, "n_rows": int(len(pa)),
           "n_F": int(isF.sum()), "n_R": int((~isF).sum())}
    for name, m in (("all", np.ones(len(pa), bool)), ("F", isF), ("R", ~isF)):
        if m.sum() == 0:
            continue
        x, y = pa[m], pb[m]
        res[name] = {
            "n": int(m.sum()),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]) if m.sum() > 1 else None,
            "max_abs_diff": float(np.abs(y - x).max()),
            "mean_abs_diff": float(np.abs(y - x).mean()),
            "rms_diff": float(np.sqrt(np.mean((y - x) ** 2))),
            "n_rows_moved": int((y != x).sum()),
        }
    res["🔴_R_rows_must_not_move"] = bool(res["R"]["n_rows_moved"] == 0) if "R" in res else None
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "a" and k != "b"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
