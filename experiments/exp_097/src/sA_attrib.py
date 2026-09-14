"""exp_097 stageA A3b: WHERE do the >1e-6 rows come from?

Both float32 sites move most rows by ~1e-9 (rounding) but a handful by ~1e-5.
A handful means a DISCRETE flip, not rounding.  Two candidates are testable
from files the run already wrote:
  * mate/output/v6aux.csv  nn_nact -- an INTEGER count of active hidden units.
    A unit sitting exactly at the ReLU boundary flips with the dtype, and the
    CA axis multiplies that integer by u = 3.5931e-04.
  * mate/output/submission.csv -- the partner arm itself (a tree split flip
    would show up here).
"""
import json
import os

import numpy as np
import pandas as pd

B = "/home/<user>/SungjinPark/exp097"
U_NACT = 0.00035931005680782873
OUT = os.path.join(B, "out/sA_attrib.json")
res = {"u_nact": U_NACT}


def rd(v, rel, col):
    p = os.path.join(B, "runs", v, rel)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
    return d["row_id"].to_numpy(), d[col].to_numpy(np.float64)


id0, p0 = rd("V0", "output/submission.csv", "control_success")

for v in ("V1", "V2", "V3", "V4"):
    if not os.path.exists(os.path.join(B, "runs", v, "output/submission.csv")):
        res[v] = "MISSING"
        continue
    e = {}
    idv, pv = rd(v, "output/submission.csv", "control_success")
    assert (idv == id0).all()
    dp = pv - p0
    big = np.where(np.abs(dp) > 1e-6)[0]
    e["n_rows_over_1e-6"] = int(big.size)
    e["big_row_ids"] = [str(x) for x in id0[big][:30]]
    e["big_dp"] = [float(x) for x in dp[big][:30]]

    # --- nn_nact (integer) -------------------------------------------------
    r = rd(v, "mate/output/v6aux.csv", "nn_nact")
    r0 = rd("V0", "mate/output/v6aux.csv", "nn_nact")
    if r and r0:
        d = r[1] - r0[1]
        nzi = np.where(d != 0)[0]
        e["nn_nact"] = {"n_rows_changed": int(nzi.size),
                        "unique_deltas": sorted(set(float(x) for x in d[nzi]))[:10],
                        "implied_dlogit_max": float(np.abs(d).max() * U_NACT),
                        "implied_dp_max_at_p0.5":
                            float(0.25 * np.abs(d).max() * U_NACT),
                        "rows_changed_ids": [str(x) for x in id0[nzi][:30]]}
        e["big_rows_all_have_nact_flip"] = bool(
            big.size and set(big.tolist()) <= set(nzi.tolist()))

    # --- partner arm -------------------------------------------------------
    q = rd(v, "mate/output/submission.csv", "control_success")
    q0 = rd("V0", "mate/output/submission.csv", "control_success")
    if q and q0:
        dq = q[1] - q0[1]
        e["partner_arm"] = {"max_abs": float(np.abs(dq).max()),
                            "n_over_1e-6": int((np.abs(dq) > 1e-6).sum()),
                            "mean_abs": float(np.abs(dq).mean())}

    # --- failtype logits ---------------------------------------------------
    f = rd(v, "mate/output/failtype.csv", "lg_mr")
    f0 = rd("V0", "mate/output/failtype.csv", "lg_mr")
    if f and f0:
        df = f[1] - f0[1]
        e["failtype_lg_mr"] = {"max_abs": float(np.abs(df).max()),
                               "n_over_1e-6": int((np.abs(df) > 1e-6).sum())}
    res[v] = e
    print(v, json.dumps(e, ensure_ascii=False)[:900])

json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print("WROTE", OUT)
