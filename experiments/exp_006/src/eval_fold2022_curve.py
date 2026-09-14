"""exp_006 addendum — valid-R Brier curve for fold_2022 (train<=2021 -> valid 2022).

Purpose: policy P4 ("carry previous fold's valid-R Brier argmin") needs the
fold_2022 R-Brier argmin as its input for simulation B (valid 2023). The main
run only evaluated curves for folds 2023/2024. Reuses the saved 1000-round
booster (identical trees, same seed) — no retraining.
"""

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_006")
TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
]
ITERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 45, 48, 50, 60,
         75, 100, 150, 200, 300, 400, 600, 800, 1000]

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
feat_cols = [c for c in df.columns if c not in DROP_COLS]
for c in CAT_COLS:
    df[c] = df[c].astype("category")

va = df[df["season"] == 2022]
X_va = va[feat_cols]
y = va[TARGET].to_numpy().astype(float)
is_r = (va["game_type"] == "R").to_numpy()

booster = lgb.Booster(model_file=os.path.join(OUT_DIR, "model_fold_2022_1000.txt"))
booster.params["num_threads"] = 8

curve = {}
t0 = time.time()
for k in ITERS:
    p = booster.predict(X_va, num_iteration=k)
    b_r = float(np.mean((p[is_r] - y[is_r]) ** 2))
    curve[str(k)] = round(b_r, 6)
argmin_k = min(curve, key=curve.get)
out = {
    "fold": "fold_2022",
    "valid_year": 2022,
    "n_valid_R": int(is_r.sum()),
    "curve_valid_R_brier": curve,
    "curve_argmin_valid_R": {"iter": int(argmin_k), "brier": curve[argmin_k]},
    "eval_time_sec": round(time.time() - t0, 1),
}
with open(os.path.join(OUT_DIR, "fold2022_curve.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
