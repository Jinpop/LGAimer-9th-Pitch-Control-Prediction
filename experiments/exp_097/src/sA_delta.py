"""exp_097 stageA A3: what does each float32 site cost in FINAL probability?

Every variant is the same package on the same frame; only the dtype of the named
site differs.  The comparison is against V0 (shipped, unmodified).

  V0  shipped
  VD  V0 + dtype prints only        -> must be BIT-IDENTICAL to V0 (control)
  V1  mate/nn_infer.py   in float64  (v6 NN: mu/sd, X, embeddings -> f64 GEMM)
  V2  mate/deep_infer.py in float64  (deep arm standardisation)
  V3  inseason.py + mate/inseason.py return float64 (16 ins_* columns)
  V4  V1 + V2 + V3
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

K = 400309.19882517256
B = "/home/<user>/SungjinPark/exp097"
OUT = os.path.join(B, "out/sA_delta.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(v):
    p = os.path.join(B, "out", "sub_%s.csv" % v)
    if not os.path.exists(p):
        return None, None
    d = pd.read_csv(p, encoding="utf-8", float_precision="round_trip")
    return d, sha(p)


base, sha0 = load("V0")
assert base is not None, "V0 missing"
ids = base["row_id"].to_numpy()
p0 = base["control_success"].to_numpy(np.float64)
res = {"K": K, "n_rows": int(len(p0)),
       "frame": "build_BX/frames/cov (245,789 rows, ins_p_n>0 on 245,709)",
       "V0_sha256": sha0,
       "p0_stats": {"mean": float(p0.mean()), "min": float(p0.min()),
                    "max": float(p0.max())}, "variants": {}}

for v in ("VD", "V1", "V2", "V3", "V4"):
    d, s = load(v)
    if d is None:
        res["variants"][v] = {"status": "MISSING"}
        print(v, "MISSING")
        continue
    assert (d["row_id"].to_numpy() == ids).all(), v + ": row order differs"
    p = d["control_success"].to_numpy(np.float64)
    dp = p - p0
    a = np.abs(dp)
    nz = int((dp != 0.0).sum())
    r = {"sha256": s, "byte_identical_to_V0": s == sha0,
         "n_rows_moved": nz,
         "frac_rows_moved": nz / len(p0),
         "max_abs_dp": float(a.max()),
         "mean_abs_dp": float(a.mean()),
         "rms_dp": float(np.sqrt((dp ** 2).mean())),
         "K_mean_dp2_score_units": float(K * (dp ** 2).mean()),
         "mean_dp_signed": float(dp.mean()),
         "q": {q: float(np.quantile(a, q))
               for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
         "over_1e-6_rows": int((a > 1e-6).sum()),
         "over_1e-9_rows": int((a > 1e-9).sum()),
         "over_1e-12_rows": int((a > 1e-12).sum())}
    res["variants"][v] = r
    print("%-3s moved=%-7d max|dp|=%.3e  mean|dp|=%.3e  K*mean(dp^2)=%.6f"
          "  rows>1e-6=%d" % (v, nz, r["max_abs_dp"], r["mean_abs_dp"],
                              r["K_mean_dp2_score_units"], r["over_1e-6_rows"]))

# additivity check: does V4 equal V1+V2+V3 in the delta?
try:
    d1 = load("V1")[0]["control_success"].to_numpy(np.float64) - p0
    d2 = load("V2")[0]["control_success"].to_numpy(np.float64) - p0
    d3 = load("V3")[0]["control_success"].to_numpy(np.float64) - p0
    d4 = load("V4")[0]["control_success"].to_numpy(np.float64) - p0
    r = d4 - (d1 + d2 + d3)
    res["additivity_V4_minus_sum"] = {"max_abs": float(np.abs(r).max()),
                                      "rms": float(np.sqrt((r ** 2).mean()))}
    print("additivity  max|V4-(V1+V2+V3)| = %.3e" % np.abs(r).max())
except Exception as e:
    res["additivity_V4_minus_sum"] = {"error": str(e)}

json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT)
