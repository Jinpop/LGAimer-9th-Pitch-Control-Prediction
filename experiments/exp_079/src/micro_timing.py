"""exp_079 stage3 보조 — CatBoost 추론 시간의 트리 수 선형성 실측.

단독 실행 전제.  253,507행 × 99열(대회 test 245,789행보다 3.1% 큼 = 보수적)에서
ntree_end 를 바꿔가며 predict_proba 시간을 잰다.  출하 경로와 같게 thread_count
는 지정하지 않는다(v6_infer.py 가 지정하지 않는다).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import json         # noqa: E402
import sys          # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import catboost as cb  # noqa: E402

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp079"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
MPATH = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/models/oot_s42_i1600_t6.cbm"

FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
df = pd.read_parquet(FRAME, columns=sorted(set(FE + ["season"])))
vam = np.where(df["season"].to_numpy() == 2024)[0]
Xv = df[FE].iloc[vam].copy()
for c in CATC:
    Xv[c] = Xv[c].astype(str)
del df
n = len(Xv)

m = cb.CatBoostClassifier()
m.load_model(MPATH)
m.predict_proba(Xv.iloc[:2000])                      # 워밍업
REP = 3
rows = []
for k in (354, 600, 900, 1200, 1600):
    if k > m.tree_count_:
        continue
    ts = []
    for _ in range(REP):
        t = time.time()
        m.predict_proba(Xv, ntree_end=k)
        ts.append(time.time() - t)
    rows.append({"trees": k, "sec_min": min(ts), "sec_median": float(np.median(ts)),
                 "sec_all": ts})
    print(f"  {k:5d}그루  {min(ts):7.3f}초 (중앙 {np.median(ts):.3f})", flush=True)

x = np.array([r["trees"] for r in rows], float)
y = np.array([r["sec_min"] for r in rows])
a, b = np.polyfit(x, y, 1)
R = {"model": MPATH, "n_rows": int(n), "n_features": len(FE),
     "test_rows_real": 245789, "row_ratio_frame_over_test": n / 245789,
     "repeats": REP, "rows": rows,
     "linear_fit_sec": {"slope_per_tree": float(a), "intercept": float(b),
                        "r2": float(1 - np.sum((y - (a * x + b)) ** 2)
                                    / np.sum((y - y.mean()) ** 2))},
     "note": "출하 v6-Cat 은 시드 4개를 순차 예측하므로 팔 전체 비용은 4배다."}
json.dump(R, open(f"{OUT}/out/micro_timing.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\n기울기 {a*1000:.4f} ms/그루 · 절편 {b:.3f}초 · R2 "
      f"{R['linear_fit_sec']['r2']:.6f}", flush=True)
print(f"기록 {OUT}/out/micro_timing.json", flush=True)
