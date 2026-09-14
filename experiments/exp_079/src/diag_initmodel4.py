"""exp_079 진단 4 — 분리 시험을 파일로 남긴다 (진단 2 는 save 예외로 dump 미도달).

weights=[1.0, **0.0**] 로 새 트리 기여를 0 으로 만든 뒤 접두를 본다.
그래도 어긋나면 원인은 추가된 트리가 아니라 CTR 표 병합이다.
"""
import os
THREADS = 6
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS)
import json, time                                            # noqa: E402
import numpy as np, pandas as pd, catboost as cb             # noqa: E402

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
OUT = f"{H}/exp079"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
T0 = time.time()
FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
df = pd.read_parquet(f"{H}/exp071/cache/frame071.parquet",
                     columns=sorted(set(FE + ["season"])))
vam = np.where(df["season"].to_numpy() == 2024)[0]
Xv = df[FE].iloc[vam].copy()
for c in CATC:
    Xv[c] = Xv[c].astype(str)
del df
m0 = cb.CatBoostClassifier(); m0.load_model(f"{MODEL}/cat_42.cbm")
mN = cb.CatBoostClassifier(); mN.load_model(f"{OUT}/models/diag_newonly_s42_i346.cbm")
p0 = m0.predict_proba(Xv, thread_count=THREADS)[:, 1].astype(np.float64)

def prob(m, ntree_end=0):
    r = np.asarray(m.predict(Xv, prediction_type="Probability",
                             ntree_end=ntree_end, thread_count=THREADS), np.float64)
    return r[:, 1] if r.ndim == 2 else r

R = {"exp_id": "exp_079", "part": "diag_initmodel4_isolation",
     "설계": "sum_models weights=[1.0, 0.0] — 새 트리의 기여가 정확히 0 이다. "
            "그래도 접두 354그루 예측이 배포본과 어긋나면 원인은 '추가된 트리'가 "
            "아니라 '두 모델을 합치는 행위(CTR 표 병합)' 다.",
     "m0_trees": int(m0.tree_count_), "mN_trees": int(mN.tree_count_)}
for w in ([1.0, 0.0], [1.0, 1.0]):
    for pol in ("KeepAllTables", "IntersectingCountersAverage",
                "LeaveMostDiversifiedTable"):
        ms = cb.sum_models([m0, mN], weights=w, ctr_merge_policy=pol)
        pp = prob(ms, 354); d = np.abs(pp - p0)
        k = f"w{w[0]:g}_{w[1]:g}__{pol}"
        R[k] = {"weights": w, "policy": pol, "tree_count": int(ms.tree_count_),
                "prefix_max_abs_diff": float(d.max()),
                "prefix_n_diff_rows": int((d > 0).sum()),
                "prefix_bitwise_equal": bool(np.array_equal(pp, p0)),
                "prefix_rms_diff": float(np.sqrt(np.mean(d ** 2)))}
        print(f"  {k}: |Δ| {d.max():.6e} 행 {int((d>0).sum())} 동일 "
              f"{R[k]['prefix_bitwise_equal']}", flush=True)
R["결론"] = ("weights=[1,0] 인데도 IntersectingCountersAverage 에서 접두가 "
           "어긋나면 CTR 표 병합이 원인임이 확정된다.  fit(init_model=...) 은 "
           "이 정책을 기본으로 쓴다.")
R["n_rows"] = int(len(Xv)); R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/diag_initmodel4.json", "w"), indent=1, ensure_ascii=False)
print(f"기록 {OUT}/out/diag_initmodel4.json", flush=True)
