"""exp_079 stage1 워커 — ≤2023 마스크 CatBoost 1시드를 1600그루까지 적합.

사용:  fit_oot.py <seed> <iterations>

exp_077 fit_one.py 의 `P` 딕셔너리를 값 그대로 쓴다.  달라지는 것은 iterations
하나뿐(354 → 1600).  thread_count 는 6 고정 (S17 — 스레드는 시드급 모델 변수).

트리 수 격자마다 ntree_end 로 잘라 예측을 저장한다.  gradient boosting 에서
앞 k 그루는 'k 라운드까지 학습한 모델' 그 자체이므로 이 절단이 학습량 축이다
(stage0 G0e 가 짧은 적합과의 동일성을 실측으로 확인한다).
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
ITERS = int(sys.argv[2])

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp079"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
GRID = sorted({354} | set(range(400, 1601, 50)))
GRID = [k for k in GRID if k <= ITERS]
TAG = f"oot_s{SEED}_i{ITERS}_t{THREADS}"
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
trm = np.where((s <= 2023) & ~((s <= 2022) & (g == "F")))[0]
assert len(trm) == 1116277, len(trm)
vam = np.where(s == 2024)[0]
assert len(vam) == 253507, len(vam)
y = df["control_success"].to_numpy(np.float64)
Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
t_frame = time.time() - t0

P = dict(iterations=ITERS, learning_rate=0.06, depth=6, l2_leaf_reg=10,
         loss_function="Logloss", bootstrap_type="Bernoulli", subsample=0.8,
         has_time=False, one_hot_max_size=2, border_count=64,
         boosting_type="Plain", verbose=200, thread_count=THREADS)

t1 = time.time()
m = cb.CatBoostClassifier(random_seed=SEED, allow_writing_files=False, **P)
m.fit(Xc.iloc[trm], y[trm], cat_features=ci)
t_fit = time.time() - t1

Xv = Xc.iloc[vam]
del Xc
t2 = time.time()
preds = {}
for k in GRID:
    p = m.predict_proba(Xv, ntree_end=k,
                        thread_count=THREADS)[:, 1].astype(np.float64)
    fp = f"{OUT}/cache/p_{TAG}_k{k}.npy"
    np.save(fp + ".tmp.npy", p)
    os.replace(fp + ".tmp.npy", fp)
    preds[k] = {"mean": float(p.mean()), "min": float(p.min()),
                "max": float(p.max())}
t_pred = time.time() - t2

MP = f"{OUT}/models/{TAG}.cbm"
m.save_model(MP)
rec = {"seed": SEED, "iterations": ITERS, "threads": THREADS,
       "n_train": int(len(trm)), "n_pred": int(len(vam)),
       "tree_count": int(m.tree_count_),
       "learn_best_error": float(m.get_best_score()["learn"]["Logloss"])
       if m.get_best_score() else None,
       "model_path": MP,
       "model_sha256": hashlib.sha256(open(MP, "rb").read()).hexdigest(),
       "grid": GRID, "pred_stats": {str(k): v for k, v in preds.items()},
       "sec_frame": t_frame, "sec_fit": t_fit, "sec_pred_all_grid": t_pred,
       "sec_total": time.time() - t0}
json.dump(rec, open(DONE + ".tmp", "w"), indent=1)
os.replace(DONE + ".tmp", DONE)
print(f"[done] {TAG}  fit {t_fit/60:.2f}분  전체 {(time.time()-t0)/60:.2f}분",
      flush=True)
