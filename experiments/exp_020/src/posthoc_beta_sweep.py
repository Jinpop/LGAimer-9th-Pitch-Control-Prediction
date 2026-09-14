"""exp_020 post-hoc: Brier of E1 as a function of beta, on fold_2024.

Run AFTER train.py.  Nothing here selected the reported beta (that is the
pre-registered pooled OOF slope, 0.541136); this only shows how flat the loss
is in beta, which is the sensitivity the DEV2 deviation calls for.
"""
import json, os, numpy as np, pandas as pd

EXP = os.path.expanduser("~/SungjinPark/experiments/exp_020")
DATA = os.path.expanduser("~/SungjinPark/data/train.csv")
BETAS = {"exp019_prereg_pooled_2020_2023_on_A0": 0.450350,
         "exp019_D5_2021_2023_on_A0": 0.470000,
         "exp020_beta_A0_own_argmin": 0.525220,
         "exp020_beta_B3_PREREG": 0.541136,
         "exp020_beta_B3_P4_chained": 0.564661,
         "ORACLE_on_valid_2024_DISPLAY_ONLY": 0.652631}

df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=["season", "control_success"])
y = df.loc[df.season == 2024, "control_success"].to_numpy(float)
p0 = np.load(os.path.join(EXP, "pred/pred_E0_B3_P4iter_seed42_fold_2024.npy"))
p1 = np.load(os.path.join(EXP, "pred/pred_E1_B3_P4iter_plus_offset_seed42_fold_2024.npy"))
d = (p1 - p0) / 0.541136                       # exact: E1 = clip(p0 + beta*d)
b0 = float(np.mean((p0 - y) ** 2))
out = {"note": "POST-HOC display only; nothing here selected the reported beta",
       "n": int(len(y)), "E0_brier": round(b0, 9),
       "E_d_s_squared_valid_2024": float(np.mean(d ** 2)), "sweep": {}}
for tag, b in BETAS.items():
    p = np.clip(p0 + b * d, 1e-6, 1 - 1e-6)
    br = float(np.mean((p - y) ** 2))
    out["sweep"][tag] = {"beta": b, "brier": round(br, 6),
                         "delta_vs_E0": round(br - b0, 9)}
path = os.path.join(EXP, "diagnostics.json")
dg = json.load(open(path))
dg["D5b_beta_sweep_POSTHOC"] = out
json.dump(dg, open(path, "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2))
