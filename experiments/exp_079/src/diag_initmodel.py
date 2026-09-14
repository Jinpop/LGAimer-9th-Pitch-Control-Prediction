"""exp_079 진단 — init_model 접두 항등이 왜 20그루에서는 성립하고 346그루에서 깨지는가.

관측:
  stage0b G0a  iterations=20   → 접두 354그루 예측이 배포본과 **정확히 0.0**
  stage2       iterations=346  → max|Δ| 0.0238, 253,507행 중 252,237행이 다름

가설: CatBoost 의 학습 이어붙이기는 내부적으로 `sum_models([init, new])` 이고
기본 `ctr_merge_policy` 가 **IntersectingCountersAverage** 다.  두 모델의 범주형
카운터(CTR) 표가 겹치면 **평균**되므로, 앞 354그루가 보는 CTR 값 자체가 바뀐다.
새 트리가 CTR 표를 만들지 않을 만큼 짧으면(20그루) 겹칠 표가 없어 항등이 성립한다.

시험:
  A  fit(init_model=...) 를 iterations 100 으로 → 어디서 깨지는가
  B  baseline 수동 적합 + sum_models(ctr_merge_policy='KeepAllTables') → 항등 회복?
  C  같은 쌍에 기본 정책(IntersectingCountersAverage) → A 재현?
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
T0 = time.time()
R = {"exp_id": "exp_079", "part": "diag_initmodel",
     "관측": {"iterations_20_prefix_max_abs_diff": 0.0,
             "iterations_346_prefix_max_abs_diff": 0.0238282684767398,
             "iterations_346_n_diff_rows": 252237, "n_rows": 253507}}

FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
ci = [FE.index(c) for c in CATC]
df = pd.read_parquet(FRAME, columns=sorted(set(FE + ["season", "game_type",
                                                     "control_success"])))
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
trm = np.where(~((s <= 2022) & (g == "F")))[0]
vam = np.where(s == 2024)[0]
assert len(trm) == 1369784 and len(vam) == 253507
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

P = dict(learning_rate=0.06, depth=6, l2_leaf_reg=10, loss_function="Logloss",
         bootstrap_type="Bernoulli", subsample=0.8, has_time=False,
         one_hot_max_size=2, border_count=64, boosting_type="Plain",
         verbose=0, thread_count=THREADS)


def cmp_prefix(m, tag):
    pp = m.predict_proba(Xv, ntree_end=354,
                         thread_count=THREADS)[:, 1].astype(np.float64)
    d = np.abs(pp - p0)
    rec = {"tree_count": int(m.tree_count_), "max_abs_diff": float(d.max()),
           "n_diff_rows": int((d > 0).sum()),
           "bitwise_equal": bool(np.array_equal(pp, p0)),
           "rms_diff": float(np.sqrt(np.mean(d ** 2)))}
    print(f"  [{tag}] trees {rec['tree_count']} 접두|Δ| "
          f"{rec['max_abs_diff']:.6e} 다른행 {rec['n_diff_rows']} "
          f"동일 {rec['bitwise_equal']}", flush=True)
    return rec


# ── A  fit(init_model=...) 을 100 그루로 ────────────────────────────
print("\n=== A: fit(init_model) iterations=100 ===", flush=True)
t = time.time()
mA = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                           iterations=100, **P)
mA.fit(Xtr, ytr, cat_features=ci, init_model=SRC)
R["A_fit_init_model_100"] = cmp_prefix(mA, "A100")
R["A_fit_init_model_100"]["sec"] = time.time() - t

# ── B  수동 baseline + sum_models 정책 비교 ─────────────────────────
print("\n=== B: baseline 수동 적합 346그루 ===", flush=True)
t = time.time()
pool_tr = cb.Pool(Xtr, ytr, cat_features=ci)
base_tr = m0.predict(pool_tr, prediction_type="RawFormulaVal")
pool_tr.set_baseline(base_tr)
mN = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                           iterations=346, **P)
mN.fit(pool_tr)
R["B_new_model_only"] = {"tree_count": int(mN.tree_count_),
                         "sec": time.time() - t}
print(f"  새 모델 단독 트리 {mN.tree_count_} · {time.time()-t:.0f}초", flush=True)

for pol in ("KeepAllTables", "IntersectingCountersAverage",
            "LeaveMostDiversifiedTable"):
    try:
        ms = cb.sum_models([m0, mN], weights=[1.0, 1.0], ctr_merge_policy=pol)
    except Exception as e:                                   # noqa: BLE001
        R[f"B_sum_models_{pol}"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  [{pol}] 실패 {e}", flush=True)
        continue
    R[f"B_sum_models_{pol}"] = cmp_prefix(ms, pol)
    if pol == "KeepAllTables" and R[f"B_sum_models_{pol}"]["bitwise_equal"]:
        ms.save_model(f"{OUT}/models/diag_sum_keepall_s42_T700.cbm")

R["결론"] = (
    "A100 과 B 의 정책별 결과로 판정한다.  KeepAllTables 에서 접두가 비트 "
    "동일이면 원인은 CTR 표 병합이고 설계는 그 정책으로 복구된다. "
    "세 정책 전부 깨지면 이어붙이기 자체가 접두를 보존하지 않는 것이다.")
R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/diag_initmodel.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\n기록 {OUT}/out/diag_initmodel.json · {R['wall_min']:.2f}분", flush=True)
