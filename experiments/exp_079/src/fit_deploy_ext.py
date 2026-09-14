"""exp_079 stage2 워커 — 배포 cat_{seed}.cbm 을 init_model 로 목표 그루수까지 연장.

사용:  fit_deploy_ext.py <seed> <target_trees>

변인은 정확히 하나 — **추가된 트리**.  앞 354그루는 배포본에서 그대로 복사되므로
항등 앵커가 정확하다(stage0b G0a 가 접두 비트 동일을 실측).

⚠️ 라벨은 int64 로 넘긴다.  배포본의 class_names 가 정수 [0, 1] 이라 float 라벨은
   `Cannot sum models with different class params` 로 거부된다.  stage0b G0a3 가
   라벨 dtype 이 예측을 바꾸지 않음을 비트 동일로 확인했다.
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

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp079"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
TAG = f"dep_ext_s{SEED}_T{TARGET}_t{THREADS}"
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
t_frame = time.time() - t0

SRC = f"{MODEL}/cat_{SEED}.cbm"
m0 = cb.CatBoostClassifier()
m0.load_model(SRC)
assert m0.tree_count_ == BASE_TREES, m0.tree_count_
p0 = m0.predict_proba(Xc.iloc[vam], thread_count=THREADS)[:, 1].astype(np.float64)

P = dict(iterations=TARGET - BASE_TREES, learning_rate=0.06, depth=6,
         l2_leaf_reg=10, loss_function="Logloss", bootstrap_type="Bernoulli",
         subsample=0.8, has_time=False, one_hot_max_size=2, border_count=64,
         boosting_type="Plain", verbose=200, thread_count=THREADS)

t1 = time.time()
m = cb.CatBoostClassifier(random_seed=SEED, allow_writing_files=False, **P)
m.fit(Xc.iloc[trm], y[trm], cat_features=ci, init_model=SRC)
t_fit = time.time() - t1
assert m.tree_count_ == TARGET, (m.tree_count_, TARGET)

Xv = Xc.iloc[vam]
del Xc
p_pref = m.predict_proba(Xv, ntree_end=BASE_TREES,
                         thread_count=THREADS)[:, 1].astype(np.float64)
anchor = {"prefix_max_abs_diff": float(np.abs(p_pref - p0).max()),
          "prefix_bitwise_equal": bool(np.array_equal(p_pref, p0)),
          "prefix_n_diff_rows": int((p_pref != p0).sum())}
assert anchor["prefix_bitwise_equal"], anchor

p_full = m.predict_proba(Xv, thread_count=THREADS)[:, 1].astype(np.float64)
np.save(f"{OUT}/cache/p_{TAG}_insample2024.npy", p_full)
np.save(f"{OUT}/cache/p_dep354_s{SEED}_insample2024.npy", p0)

MP = f"{OUT}/models/{TAG}.cbm"
m.save_model(MP)
rec = {"seed": SEED, "target_trees": TARGET, "base_trees": BASE_TREES,
       "added_trees": TARGET - BASE_TREES, "threads": THREADS,
       "n_train": int(len(trm)), "tree_count": int(m.tree_count_),
       "init_model": SRC,
       "init_model_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
       "model_path": MP,
       "model_sha256": hashlib.sha256(open(MP, "rb").read()).hexdigest(),
       "model_bytes": os.path.getsize(MP),
       "identity_anchor_prefix354": anchor,
       "insample2024_pred_mean_base": float(p0.mean()),
       "insample2024_pred_mean_ext": float(p_full.mean()),
       "insample2024_max_abs_shift": float(np.abs(p_full - p0).max()),
       "learn_best_error": float(m.get_best_score()["learn"]["Logloss"])
       if m.get_best_score() else None,
       "sec_frame": t_frame, "sec_fit": t_fit, "sec_total": time.time() - t0}
json.dump(rec, open(DONE + ".tmp", "w"), indent=1)
os.replace(DONE + ".tmp", DONE)
print(f"[done] {TAG}  fit {t_fit/60:.2f}분  접두항등 "
      f"{anchor['prefix_bitwise_equal']}  전체 {(time.time()-t0)/60:.2f}분",
      flush=True)
