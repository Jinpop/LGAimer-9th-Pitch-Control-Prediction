"""exp_097b stageA — 출하 딥 팔 표준화 상수(mu/sd) 94열 전수 대조.

질문: 배포 prep.json 의 mu/sd 가 exp_096 이 찾은 float32 누산 결함을 갖고 있는가?
방법: 배포가 선언한 학습 프레임(train season<=2024)에서 f64 로 다시 계산해 대조한다.
      f32 누산 판본도 함께 계산해 '어느 레시피가 출하값과 맞는가'로 판정한다.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

PKG = sys.argv[1]
FRAME = sys.argv[2]
OUT = sys.argv[3]

P = json.load(open(os.path.join(PKG, "mate/model/deep/prep.json")))
NUM = P["NUM"]
mu_ship = np.array(P["mu"], dtype=np.float64)
sd_ship = np.array(P["sd"], dtype=np.float64)

df = pd.read_parquet(FRAME)
print("frame rows", len(df), "cols", df.shape[1], flush=True)
if "season" in df.columns:
    m = df["season"].to_numpy() <= 2024
    df = df.loc[m]
print("rows<=2024", len(df), flush=True)

miss = [c for c in NUM if c not in df.columns]
assert not miss, ("프레임에 없는 NUM 열", miss)

rows = []
for j, c in enumerate(NUM):
    v64 = df[c].to_numpy(dtype=np.float64)
    v32 = v64.astype(np.float32)
    mu64 = float(np.nanmean(v64))
    sd64 = float(np.nanstd(v64))
    mu32 = float(np.nanmean(v32))          # f32 배열 -> f32 누산 (exp_096 결함형)
    sd32 = float(np.nanstd(v32))
    rows.append({
        "j": j, "col": c,
        "mu_ship": float(mu_ship[j]), "sd_ship": float(sd_ship[j]),
        "mu_f64": mu64, "sd_f64": sd64,
        "mu_f32acc": mu32, "sd_f32acc": sd32,
        "sd_ratio_ship_over_f64": (sd_ship[j] / sd64) if sd64 else float("nan"),
        "sd_ratio_f32acc_over_f64": (sd32 / sd64) if sd64 else float("nan"),
        "mu_err_ship_in_truesd": (abs(mu_ship[j] - mu64) / sd64) if sd64 else float("nan"),
        "mu_err_f32acc_in_truesd": (abs(mu32 - mu64) / sd64) if sd64 else float("nan"),
        "n_finite": int(np.isfinite(v64).sum()),
    })

bad_sd = [r for r in rows if abs(r["sd_ratio_ship_over_f64"] - 1.0) > 0.01]
bad_mu = [r for r in rows if r["mu_err_ship_in_truesd"] > 0.01]
worst = sorted(rows, key=lambda r: -abs(r["sd_ratio_ship_over_f64"] - 1.0))[:10]

res = {
    "frame": FRAME, "pkg": PKG, "n_rows_used": int(len(df)), "n_cols": len(NUM),
    "n_cols_sd_off_gt_1pct": len(bad_sd),
    "cols_sd_off_gt_1pct": [(r["col"], r["sd_ratio_ship_over_f64"]) for r in bad_sd],
    "n_cols_mu_off_gt_0.01sd": len(bad_mu),
    "cols_mu_off_gt_0.01sd": [(r["col"], r["mu_err_ship_in_truesd"]) for r in bad_mu],
    "worst10_by_sd_ratio": [{k: r[k] for k in
                             ("col", "sd_ship", "sd_f64", "sd_f32acc",
                              "sd_ratio_ship_over_f64", "sd_ratio_f32acc_over_f64",
                              "mu_err_ship_in_truesd", "mu_err_f32acc_in_truesd")}
                            for r in worst],
    "recipe_fit": {
        "max_abs_rel_sd_ship_vs_f64": max(abs(r["sd_ratio_ship_over_f64"] - 1) for r in rows),
        "max_abs_rel_sd_ship_vs_f32acc": max(
            abs(r["sd_ship"] / r["sd_f32acc"] - 1) if r["sd_f32acc"] else 0.0 for r in rows),
        "max_mu_err_ship_in_truesd": max(r["mu_err_ship_in_truesd"] for r in rows),
        "max_mu_err_f32acc_in_truesd": max(r["mu_err_f32acc_in_truesd"] for r in rows),
    },
    "rows": rows,
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                 ensure_ascii=False, indent=1)[:3000])
