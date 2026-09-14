"""exp_097 stageB0: is our OOT standardisation defect ONE column or many?

exp_096 reported `season` only (mu 2040.479 vs 2021.057, sd 19.616 vs 1.4238).
Here all 94 NUM columns are laid out, on the SAME frame and the SAME train mask
exp_096 used, so the two numbers are comparable line by line.
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
E60 = f"{H}/exp060_nn"
W89 = f"{H}/exp089/work"
OUT = f"{H}/exp097/out/sB0_musd_oot.json"

P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
NUM = P["NUM"]
df = pd.read_parquet(f"{E60}/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUM + ["row_id"])))
Xraw = df[NUM].astype("float32").to_numpy()
trm = np.load(f"{W89}/trm_oot23.npy")

mu64 = np.nanmean(Xraw[trm].astype(np.float64), axis=0)
sd64 = np.nanstd(Xraw[trm].astype(np.float64), axis=0)
mu32 = np.nanmean(Xraw[trm], axis=0).astype(np.float64)
sd32 = np.nanstd(Xraw[trm], axis=0).astype(np.float64)

sd_ratio = sd32 / np.where(sd64 == 0, np.nan, sd64)
mu_err_sd = np.abs(mu32 - mu64) / np.where(sd64 == 0, np.nan, sd64)
rows = []
for j, c in enumerate(NUM):
    rows.append({"col": c, "mu_f32": float(mu32[j]), "mu_f64": float(mu64[j]),
                 "sd_f32": float(sd32[j]), "sd_f64": float(sd64[j]),
                 "sd_ratio_f32_over_f64": float(sd_ratio[j]),
                 "mu_err_in_true_sd_units": float(mu_err_sd[j])})
rows.sort(key=lambda r: -max(abs(r["sd_ratio_f32_over_f64"] - 1.0)
                             if np.isfinite(r["sd_ratio_f32_over_f64"]) else 0,
                             r["mu_err_in_true_sd_units"]
                             if np.isfinite(r["mu_err_in_true_sd_units"]) else 0))
bad_sd = [r for r in rows if np.isfinite(r["sd_ratio_f32_over_f64"])
          and abs(r["sd_ratio_f32_over_f64"] - 1.0) > 0.01]
bad_mu = [r for r in rows if np.isfinite(r["mu_err_in_true_sd_units"])
          and r["mu_err_in_true_sd_units"] > 0.01]
R = {"exp_id": "exp_097", "stage": "sB0_musd_oot",
     "frame": f"{E60}/cache/frame99.parquet",
     "mask": f"{W89}/trm_oot23.npy", "n_train_rows": int(trm.sum()),
     "n_cols": len(NUM),
     "n_cols_sd_off_gt_1pct": len(bad_sd),
     "n_cols_mu_off_gt_0.01sd": len(bad_mu),
     "cols_sd_off_gt_1pct": [r["col"] for r in bad_sd],
     "cols_mu_off_gt_0.01sd": [r["col"] for r in bad_mu],
     "all_cols": rows}
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
print("n cols with |sd ratio - 1| > 1%%: %d / %d" % (len(bad_sd), len(NUM)))
print("n cols with mu error > 0.01 true sd: %d / %d" % (len(bad_mu), len(NUM)))
print("%-32s %12s %12s %8s %10s" % ("col", "sd_f32", "sd_f64", "ratio",
                                    "mu_err/sd"))
for r in rows[:25]:
    print("%-32s %12.6g %12.6g %8.4f %10.4g"
          % (r["col"], r["sd_f32"], r["sd_f64"],
             r["sd_ratio_f32_over_f64"], r["mu_err_in_true_sd_units"]))
print("WROTE", OUT)
