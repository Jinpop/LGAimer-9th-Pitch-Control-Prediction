"""exp_079 stage2 (교정본) — 배포 cat_{seed}.cbm 을 접두 보존하며 연장.

사용:  fit_deploy_ext3.py <seed> <target_trees>

🔴 왜 교정이 필요했나 (out/diag_initmodel{,2,3}.json):
   `fit(init_model=...)` 은 내부적으로 두 모델을 합치는데 기본 CTR 표 병합 정책이
   **IntersectingCountersAverage** 다.  두 모델의 범주형 카운터 표가 겹치면 평균
   되므로 **앞 354그루가 보는 값 자체가 바뀐다.**  분리 시험이 이를 확정했다 —
   weights=[1.0, **0.0**] 로 새 트리 기여를 0 으로 만들어도 접두 편차가 그대로
   나온다(max|Δ| 2.382827e-02, 252,237/253,507행).  즉 편차의 원인은 추가된
   트리가 아니라 **합치는 행위**다.

   정책 4종 실측: KeepAllTables 접두 0.0 이나 `.cbm` 저장 실패("Some CTR data
   are lost") · **LeaveMostDiversifiedTable 접두 0.0 이고 저장 성공** ·
   IntersectingCountersAverage 접두 깨짐 · FailIfCtrsIntersects 는 1.2.10 에
   없는 이름.

   → 수동 baseline 적합 + sum_models(LeaveMostDiversifiedTable) 로 간다.
     변인은 그때 정확히 하나 — 추가된 트리.
"""
import os
import sys

THREADS = 6
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS)

import hashlib      # noqa: E402
import json         # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import catboost as cb  # noqa: E402

SEED = int(sys.argv[1])
TARGET = int(sys.argv[2])
BASE_TREES = 354
POLICY = "LeaveMostDiversifiedTable"

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp079"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
TAG = f"depext3_s{SEED}_T{TARGET}_t{THREADS}"
DONE = f"{OUT}/cache/{TAG}.json"
if os.path.exists(DONE):
    print(f"[skip] {TAG} 이미 존재", flush=True)
    sys.exit(0)

t0 = time.time()
FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
assert len(FE) == 99
ci = [FE.index(c) for c in CATC]
assert ci == [4, 10, 13, 14, 45], ci

need = sorted(set(FE + ["season", "game_type", "control_success"]))
df = pd.read_parquet(FRAME, columns=need)
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
trm = np.where(~((s <= 2022) & (g == "F")))[0]
assert len(trm) == 1369784, len(trm)
vam = np.where(s == 2024)[0]
assert len(vam) == 253507, len(vam)
y = df["control_success"].to_numpy(np.int64)
Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
Xtr, ytr, Xv = Xc.iloc[trm], y[trm], Xc.iloc[vam]
del Xc
t_frame = time.time() - t0

SRC = f"{MODEL}/cat_{SEED}.cbm"
m0 = cb.CatBoostClassifier()
m0.load_model(SRC)
assert m0.tree_count_ == BASE_TREES, m0.tree_count_
p0 = m0.predict_proba(Xv, thread_count=THREADS)[:, 1].astype(np.float64)

P = dict(iterations=TARGET - BASE_TREES, learning_rate=0.06, depth=6,
         l2_leaf_reg=10, loss_function="Logloss", bootstrap_type="Bernoulli",
         subsample=0.8, has_time=False, one_hot_max_size=2, border_count=64,
         boosting_type="Plain", verbose=200, thread_count=THREADS)

t1 = time.time()
pool_tr = cb.Pool(Xtr, ytr, cat_features=ci)
pool_tr.set_baseline(m0.predict(pool_tr, prediction_type="RawFormulaVal"))
mN = cb.CatBoostClassifier(random_seed=SEED, allow_writing_files=False, **P)
mN.fit(pool_tr)
t_fit = time.time() - t1
assert mN.tree_count_ == TARGET - BASE_TREES

ms = cb.sum_models([m0, mN], weights=[1.0, 1.0], ctr_merge_policy=POLICY)
assert ms.tree_count_ == TARGET, ms.tree_count_


def prob(m, ntree_end=0):
    r = np.asarray(m.predict(Xv, prediction_type="Probability",
                             ntree_end=ntree_end, thread_count=THREADS),
                   np.float64)
    return r[:, 1] if r.ndim == 2 else r


pre_mem = prob(ms, BASE_TREES)
full_mem = prob(ms)
anchor_mem = {"max_abs_diff": float(np.abs(pre_mem - p0).max()),
              "bitwise_equal": bool(np.array_equal(pre_mem, p0)),
              "n_diff_rows": int((pre_mem != p0).sum())}
assert anchor_mem["bitwise_equal"], anchor_mem

MP = f"{OUT}/models/{TAG}.cbm"
ms.save_model(MP)

# 저장/재적재 왕복 게이트 — CTR 표가 직렬화에서 유실되지 않았는가
mr = cb.CatBoostClassifier()
mr.load_model(MP)
pre_rt = prob(mr, BASE_TREES)
full_rt = prob(mr)
rt = {"tree_count": int(mr.tree_count_),
      "prefix_max_abs_diff_vs_deploy": float(np.abs(pre_rt - p0).max()),
      "prefix_bitwise_equal_vs_deploy": bool(np.array_equal(pre_rt, p0)),
      "full_max_abs_diff_vs_inmemory": float(np.abs(full_rt - full_mem).max()),
      "full_bitwise_equal_vs_inmemory": bool(np.array_equal(full_rt, full_mem))}
assert rt["prefix_bitwise_equal_vs_deploy"], rt
assert rt["full_bitwise_equal_vs_inmemory"], rt

np.save(f"{OUT}/cache/p_{TAG}_insample2024.npy", full_rt)
rec = {"seed": SEED, "target_trees": TARGET, "base_trees": BASE_TREES,
       "added_trees": TARGET - BASE_TREES, "threads": THREADS,
       "ctr_merge_policy": POLICY, "n_train": int(len(trm)),
       "tree_count": int(ms.tree_count_), "init_model": SRC,
       "init_model_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
       "model_path": MP,
       "model_sha256": hashlib.sha256(open(MP, "rb").read()).hexdigest(),
       "model_bytes": os.path.getsize(MP),
       "identity_anchor_prefix354_inmemory": anchor_mem,
       "save_reload_roundtrip": rt,
       "insample2024_pred_mean_base": float(p0.mean()),
       "insample2024_pred_mean_ext": float(full_rt.mean()),
       "insample2024_max_abs_shift": float(np.abs(full_rt - p0).max()),
       "insample2024_rms_shift": float(np.sqrt(np.mean((full_rt - p0) ** 2))),
       "learn_best_error_new_trees":
           float(mN.get_best_score()["learn"]["Logloss"])
           if mN.get_best_score() else None,
       "sec_frame": t_frame, "sec_fit": t_fit, "sec_total": time.time() - t0}
json.dump(rec, open(DONE + ".tmp", "w"), indent=1)
os.replace(DONE + ".tmp", DONE)
print(f"[done] {TAG}  fit {t_fit/60:.2f}분  접두항등(메모리/왕복) "
      f"{anchor_mem['bitwise_equal']}/{rt['prefix_bitwise_equal_vs_deploy']}  "
      f"전체 {(time.time()-t0)/60:.2f}분", flush=True)
