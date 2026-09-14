"""exp_079 진단 2 — sum_models 정책별 접두 항등 + 분리 시험.

1차 진단 결과 (out/diag_initmodel.json 은 B 단계에서 API 오류로 미완):
  🔴 iterations=100 의 접두 편차가 iterations=346 과 **자릿수까지 같다**
     (max|Δ| 2.382827e-02, 다른 행 252,237).  편차가 추가 트리 수에 따라 자라지
     않는다 = 누적 표류가 아니라 **일회성 이산 변화**다.

분리 시험 (이번 파일의 핵심):
  weights=[1.0, 0.0] 로 합치면 새 트리의 기여가 0 이다.  그래도 접두가 어긋나면
  원인은 '추가된 트리'가 아니라 **합치는 행위 자체(CTR 표 병합)** 다.
"""
import os

THREADS = 6
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS)

import json         # noqa: E402
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
SRC = f"{MODEL}/cat_42.cbm"
NEWP = f"{OUT}/models/diag_newonly_s42_i346.cbm"
T0 = time.time()
R = {"exp_id": "exp_079", "part": "diag_initmodel2",
     "1차_관측": {"iterations_100": {"max_abs_diff": 2.382827e-02,
                                   "n_diff_rows": 252237, "trees": 454},
                "iterations_346": {"max_abs_diff": 0.0238282684767398,
                                   "n_diff_rows": 252237, "trees": 700},
                "🔴": "두 편차가 같다 = 추가 트리 수와 무관한 일회성 변화"}}

FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
ci = [FE.index(c) for c in CATC]
df = pd.read_parquet(FRAME, columns=sorted(set(FE + ["season", "game_type",
                                                     "control_success"])))
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
trm = np.where(~((s <= 2022) & (g == "F")))[0]
vam = np.where(s == 2024)[0]
y = df["control_success"].to_numpy(np.int64)
Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
Xtr, ytr, Xv = Xc.iloc[trm], y[trm], Xc.iloc[vam]
del Xc
print(f"프레임 {time.time()-T0:.1f}s", flush=True)

m0 = cb.CatBoostClassifier()
m0.load_model(SRC)
p0 = m0.predict_proba(Xv, thread_count=THREADS)[:, 1].astype(np.float64)


def prob(m, ntree_end=0):
    r = m.predict(Xv, prediction_type="Probability", ntree_end=ntree_end,
                  thread_count=THREADS)
    r = np.asarray(r, np.float64)
    return r[:, 1] if r.ndim == 2 else r


def cmp_prefix(m, tag, ntree_end=354):
    pp = prob(m, ntree_end)
    d = np.abs(pp - p0)
    rec = {"tree_count": int(m.tree_count_), "ntree_end": ntree_end,
           "max_abs_diff": float(d.max()), "n_diff_rows": int((d > 0).sum()),
           "bitwise_equal": bool(np.array_equal(pp, p0)),
           "rms_diff": float(np.sqrt(np.mean(d ** 2)))}
    print(f"  [{tag}] trees {rec['tree_count']} 접두|Δ| "
          f"{rec['max_abs_diff']:.6e} 다른행 {rec['n_diff_rows']} "
          f"동일 {rec['bitwise_equal']}", flush=True)
    return rec


P = dict(learning_rate=0.06, depth=6, l2_leaf_reg=10, loss_function="Logloss",
         bootstrap_type="Bernoulli", subsample=0.8, has_time=False,
         one_hot_max_size=2, border_count=64, boosting_type="Plain",
         verbose=0, thread_count=THREADS)

if os.path.exists(NEWP):
    mN = cb.CatBoostClassifier()
    mN.load_model(NEWP)
    print("새 모델 재사용", flush=True)
else:
    t = time.time()
    pool_tr = cb.Pool(Xtr, ytr, cat_features=ci)
    pool_tr.set_baseline(m0.predict(pool_tr, prediction_type="RawFormulaVal"))
    mN = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                               iterations=346, **P)
    mN.fit(pool_tr)
    mN.save_model(NEWP)
    print(f"새 모델 346그루 적합 {time.time()-t:.0f}초", flush=True)
R["new_model_trees"] = int(mN.tree_count_)

print("\n=== 대조 0: m0 자기 자신 (sum_models 없이) ===", flush=True)
R["ctrl_m0_self"] = cmp_prefix(m0, "m0")

print("\n=== 분리 시험: weights=[1.0, 0.0] — 새 트리 기여 0 ===", flush=True)
for pol in ("KeepAllTables", "IntersectingCountersAverage"):
    try:
        ms = cb.sum_models([m0, mN], weights=[1.0, 0.0], ctr_merge_policy=pol)
        R[f"isolate_w1_0_{pol}"] = cmp_prefix(ms, f"w[1,0]·{pol}")
    except Exception as e:                                   # noqa: BLE001
        R[f"isolate_w1_0_{pol}"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  [w[1,0]·{pol}] 실패 {e}", flush=True)

print("\n=== 정책별 접두 항등 (weights=[1,1]) ===", flush=True)
for pol in ("KeepAllTables", "IntersectingCountersAverage",
            "LeaveMostDiversifiedTable", "FailIfCtrsIntersects"):
    try:
        ms = cb.sum_models([m0, mN], weights=[1.0, 1.0], ctr_merge_policy=pol)
    except Exception as e:                                   # noqa: BLE001
        R[f"sum_{pol}"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  [{pol}] 실패 {type(e).__name__}: {str(e)[:140]}", flush=True)
        continue
    R[f"sum_{pol}"] = cmp_prefix(ms, pol)
    R[f"sum_{pol}"]["full_pred_mean"] = float(prob(ms).mean())
    if R[f"sum_{pol}"]["bitwise_equal"]:
        ms.save_model(f"{OUT}/models/sum_{pol}_s42_T700.cbm")
        print(f"    🟢 {pol} 로 접두 항등 회복 — 모델 저장", flush=True)

R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/diag_initmodel2.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\n기록 {OUT}/out/diag_initmodel2.json · {R['wall_min']:.2f}분", flush=True)
