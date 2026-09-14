"""exp_097 stageA: are the SHIPPED standardisation constants float32-accumulated?

exp_096 found our own measurement-surface prep did `nanmean/nanstd(axis=0)` on a
float32 (1.1M, 99) array, which accumulates into a float32 buffer row by row and
loses the mean entirely on a large-offset column (`season`: sd 1.4238 -> 19.616).

Here we ask the same of the DEPLOYED constants:
  * mate/model/deep/prep.json          mu/sd over 94 NUM cols
  * mate/model/nn_a{42,777,2024,101}_meta.json  mu/sd over 73 DENSE cols

Reference frame: exp064/cache/frame99_ship.parquet (train season<=2024,
1,475,092 rows) -- the same frame prep.json names.
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq

BASE = "/home/<user>/SungjinPark"
PKG = os.path.join(BASE, "exp097/pkg/CH")
FRAME = os.path.join(BASE, "exp064/cache/frame99_ship.parquet")
OUT = os.path.join(BASE, "exp097/out/sA_musd.json")

res = {"frame": FRAME}


def load_cols(cols):
    t = pq.read_table(FRAME, columns=cols)
    X = np.empty((t.num_rows, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
    return X


def variants(X64):
    """Return {label: (mu, sd)} for the plausible ways to compute the constants."""
    X32 = X64.astype(np.float32)
    out = {}
    with np.errstate(invalid="ignore"):
        out["f64_axis0_ddof0"] = (np.nanmean(X64, axis=0), np.nanstd(X64, axis=0))
        out["f64_axis0_ddof1"] = (np.nanmean(X64, axis=0),
                                  np.nanstd(X64, axis=0, ddof=1))
        out["f32_axis0_ddof0"] = (np.nanmean(X32, axis=0).astype(np.float64),
                                  np.nanstd(X32, axis=0).astype(np.float64))
        # what a float32 array with a float64 accumulator would give
        out["f32in_f64acc"] = (np.nanmean(X32, axis=0, dtype=np.float64),
                               np.nanstd(X32, axis=0, dtype=np.float64))
    return out


def compare(name, cols, mu_ship, sd_ship):
    X64 = load_cols(cols)
    n_nan = np.isnan(X64).sum(axis=0)
    V = variants(X64)
    rows = []
    fit = {}
    for lab, (mu, sd) in V.items():
        # a float32 STORE of the constants is expected either way, so compare
        # after rounding the candidate to float32
        d_mu = np.abs(mu.astype(np.float32).astype(np.float64) - mu_ship)
        d_sd = np.abs(sd.astype(np.float32).astype(np.float64) - sd_ship)
        rel_sd = d_sd / np.maximum(np.abs(sd_ship), 1e-30)
        fit[lab] = {"max_abs_mu": float(np.nanmax(d_mu)),
                    "max_abs_sd": float(np.nanmax(d_sd)),
                    "max_rel_sd": float(np.nanmax(rel_sd)),
                    "n_cols_rel_sd_gt_1e-4": int((rel_sd > 1e-4).sum()),
                    "worst_col": cols[int(np.nanargmax(rel_sd))]}
    mu64, sd64 = V["f64_axis0_ddof0"]
    rel_sd64 = np.abs(sd64 - sd_ship) / np.maximum(np.abs(sd64), 1e-30)
    rel_mu64 = np.abs(mu64 - mu_ship) / np.maximum(np.abs(sd64), 1e-30)
    order = np.argsort(-np.maximum(rel_sd64, rel_mu64))
    for k in order[:15]:
        rows.append({"col": cols[k], "n_nan": int(n_nan[k]),
                     "mu_shipped": float(mu_ship[k]), "mu_f64": float(mu64[k]),
                     "sd_shipped": float(sd_ship[k]), "sd_f64": float(sd64[k]),
                     "sd_ratio_shipped_over_f64": float(sd_ship[k] / sd64[k])
                     if sd64[k] else None,
                     "mu_err_in_sd_units": float(rel_mu64[k]),
                     "sd_rel_err": float(rel_sd64[k])})
    res[name] = {"n_cols": len(cols), "which_recipe_fits": fit,
                 "worst_15_cols_vs_f64": rows,
                 "n_cols_sd_rel_err_gt_1pct": int((rel_sd64 > 0.01).sum()),
                 "n_cols_sd_rel_err_gt_0.1pct": int((rel_sd64 > 0.001).sum()),
                 "n_cols_mu_err_gt_0.01sd": int((rel_mu64 > 0.01).sum())}
    print(name, "-> recipe fits:", json.dumps(fit, indent=1)[:1200])
    print(name, "cols with sd rel err > 1%:", int((rel_sd64 > 0.01).sum()),
          " > 0.1%:", int((rel_sd64 > 0.001).sum()))
    for r in rows[:8]:
        print("   %-32s sd_ship=%.8g sd_f64=%.8g ratio=%.6f mu_err/sd=%.3g"
              % (r["col"], r["sd_shipped"], r["sd_f64"],
                 r["sd_ratio_shipped_over_f64"] or float("nan"),
                 r["mu_err_in_sd_units"]))
    del X64


P = json.load(open(os.path.join(PKG, "mate/model/deep/prep.json")))
compare("deep_prep", P["NUM"], np.array(P["mu"], np.float64),
        np.array(P["sd"], np.float64))

for s in (42, 777, 2024, 101):
    M = json.load(open(os.path.join(PKG, "mate/model/nn_a%d_meta.json" % s)))
    compare("nn_a%d" % s, M["DENSE"], np.array(M["mu"], np.float64),
            np.array(M["sd"], np.float64))

json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT)
