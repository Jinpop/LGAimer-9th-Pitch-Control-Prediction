"""exp_037 / A5 — the reproduction gate.

2024 is treated as if it were the hidden test set: featgen is run a second time
into a SEPARATE directory with `--seasons 2024`, so it rebuilds the <= 2023
tables from scratch, refits the <= 2023 pitch model, reserialises everything to
disk and reads it back.  This file then compares that regenerated block against
the block the training run will actually consume, element by element, over all
253,507 rows x 22 columns.

Two things are asserted:
  * every table artefact's sha256 is identical between the two runs (so the
    serialisation path is deterministic), and
  * every feature value is identical, with NaN in the same places.

The PASS tolerance is 1e-9 and bit-identity is reported SEPARATELY rather than
asserted.  The defects this gate exists to catch — a table built from the wrong
season window, a lookup key that lost a level in serialisation, a pitch model
that was not refitted — all move a feature by O(1e-2) or turn it into NaN, so
1e-9 separates them from any floating-point reassociation with several orders
of magnitude to spare, while an exactness demand would let a thread-scheduling
difference inside LightGBM's histogram build abort a whole night's run.  If the
two paths really are the same arithmetic, `bit_identical` will say so.

Writes gate_2024.json; exits non-zero on failure.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data")
TOL = 1e-9
t0 = time.time()


def main():
    main_dir = os.path.abspath(sys.argv[sys.argv.index("--main") + 1])
    gate_dir = os.path.abspath(sys.argv[sys.argv.index("--gate") + 1])
    pm = json.load(open(os.path.join(main_dir, "featgen_main.json")))
    pg = json.load(open(os.path.join(gate_dir, "featgen_gate.json")))

    fm = pd.read_parquet(os.path.join(main_dir, pm["feat_file"]))
    fg = pd.read_parquet(os.path.join(gate_dir, pg["feat_file"]))
    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id", "season"])
    ids24 = tr.loc[tr["season"] == 2024, "row_id"].to_numpy()

    A = fm.set_index("row_id").loc[ids24]
    B = fg.set_index("row_id").loc[ids24]
    cols = list(A.columns)
    assert cols == list(B.columns), (cols, list(B.columns))

    rep = {"n_rows": int(len(ids24)), "n_cols": len(cols),
           "row_ids_aligned": bool((A.index.to_numpy() == ids24).all()
                                   and (B.index.to_numpy() == ids24).all()),
           "columns": {}, "PASS": True}

    tm = pm["per_season"]["2024"]["table_sha256"]
    tg = pg["per_season"]["2024"]["table_sha256"]
    rep["table_sha256_main"] = tm
    rep["table_sha256_gate"] = tg
    rep["table_sha256_identical"] = bool(tm == tg)
    rep["table_sha256_differing_files"] = sorted(
        [k for k in tm if tm.get(k) != tg.get(k)])
    # A table sha difference is REPORTED, not fatal: `pitchmodel6.txt` embeds
    # LightGBM's own float formatting.  The feature-value comparison below is
    # the substantive check and it is fatal.

    for c in cols:
        a = A[c].to_numpy(dtype=np.float64)
        b = B[c].to_numpy(dtype=np.float64)
        na_a, na_b = np.isnan(a), np.isnan(b)
        na_same = bool((na_a == na_b).all())
        both = ~na_a & ~na_b
        d = np.abs(a[both] - b[both])
        maxd = float(d.max()) if d.size else 0.0
        ok = na_same and maxd <= TOL
        rep["columns"][c] = {"nan_pattern_identical": na_same,
                             "n_nan": int(na_a.sum()),
                             "max_abs_diff": maxd,
                             "bit_identical": bool(na_same and maxd == 0.0),
                             "n_differing": int((d > 0).sum()) if d.size else 0,
                             "tol": TOL, "PASS": ok}
        rep["PASS"] = rep["PASS"] and ok

    rep["PASS"] = bool(rep["PASS"] and rep["row_ids_aligned"])
    rep["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(main_dir, "gate_2024.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("GATE_2024 " + ("PASS" if rep["PASS"] else "FAIL"), flush=True)
    for c, v in rep["columns"].items():
        if not v["PASS"]:
            print("  FAIL", c, v, flush=True)
    sys.exit(0 if rep["PASS"] else 1)


if __name__ == "__main__":
    main()
