"""exp_077 CatBoost 시드 1개 적합 워커.

사용:  fit_one.py <seed> <oot|deploy> <threads> <save_model:0|1>

팀원 ootbuild.py 의 `P` 딕셔너리를 값 그대로 쓴다.  달라지는 것은 셋뿐이고
전부 metrics 에 기록한다:
  1) thread_count  2(팀원, 코어 2개) -> 6(우리, config 지시)
  2) save_snapshot True -> 없음 (9분 Bash 한도가 우리에겐 없다)
  3) allow_writing_files True -> False (catboost_info 디렉터리 경합 방지)
"""
import os
import sys

THREADS = int(sys.argv[3])
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import json
import time

import numpy as np
import pandas as pd
import catboost as cb

SEED = int(sys.argv[1])
MASK = sys.argv[2]
SAVE_MODEL = bool(int(sys.argv[4]))

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BN/pkg_BN/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp077"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
TAG = f"{MASK}_s{SEED}_t{THREADS}"
PRED = f"{OUT}/cache/_cat_{TAG}.npy"
if os.path.exists(PRED):
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
if MASK == "oot":
    trm = np.where((s <= 2023) & ~((s <= 2022) & (g == "F")))[0]
    assert len(trm) == 1116277, len(trm)
elif MASK == "deploy":
    trm = np.where(~((s <= 2022) & (g == "F")))[0]
    assert len(trm) == 1369784, len(trm)
else:
    raise ValueError(MASK)
vam = np.where(s == 2024)[0]
assert len(vam) == 253507, len(vam)
y = df["control_success"].to_numpy(np.float64)

Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
t_frame = time.time() - t0

P = dict(iterations=354, learning_rate=0.06, depth=6, l2_leaf_reg=10,
         loss_function="Logloss", bootstrap_type="Bernoulli", subsample=0.8,
         has_time=False, one_hot_max_size=2, border_count=64,
         boosting_type="Plain", verbose=100, thread_count=THREADS)

t1 = time.time()
m = cb.CatBoostClassifier(random_seed=SEED, allow_writing_files=False, **P)
m.fit(Xc.iloc[trm], y[trm], cat_features=ci)
t_fit = time.time() - t1

t2 = time.time()
p = m.predict_proba(Xc.iloc[vam])[:, 1].astype(np.float64)
t_pred = time.time() - t2
np.save(PRED + ".tmp.npy", p)
os.replace(PRED + ".tmp.npy", PRED)
if SAVE_MODEL:
    m.save_model(f"{OUT}/cache/_cat_{TAG}.cbm")
    np.save(f"{OUT}/cache/_leaf_{TAG}.npy",
            np.asarray(m.get_leaf_values(), np.float64))

json.dump({"seed": SEED, "mask": MASK, "threads": THREADS,
           "n_train": int(len(trm)), "n_pred": int(len(vam)),
           "tree_count": int(m.tree_count_),
           "learn_best_error": float(m.get_best_score()["learn"]["Logloss"])
           if m.get_best_score() else None,
           "pred_mean": float(p.mean()), "pred_min": float(p.min()),
           "pred_max": float(p.max()),
           "sec_frame": t_frame, "sec_fit": t_fit, "sec_pred": t_pred,
           "sec_total": time.time() - t0},
          open(f"{OUT}/cache/_cat_{TAG}.json", "w"), indent=1)
print(f"[done] {TAG}  fit {t_fit/60:.2f}분  전체 {(time.time()-t0)/60:.2f}분  "
      f"pmean {p.mean():.8f}", flush=True)
