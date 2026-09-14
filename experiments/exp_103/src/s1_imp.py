"""exp_103 stage1 부록 — 열별 중요도 + 개별 열 기여(1시드, 진단용).

config 이 요구한 "열별 중요도 · 어느 열이 사는지" 를 채운다. 판정에는 쓰지 않는다
(판정은 s1_screen2.py 의 3판본 대조 + 사전 등록 문턱).
"""
import json, os, time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool

H = "/home/<user>/SungjinPark"; B = f"{H}/exp103"; W97 = f"{H}/exp097/work"
REF = 0.2498069
DEV = os.environ.get("GPU103", "0")
X = np.load(f"{W97}/X_oot23.npy"); C = np.load(f"{W97}/codes_oot23.npy")
y = np.load(f"{W97}/y.npy").astype(np.float64)
trm = np.load(f"{W97}/trm_oot23.npy"); vam = np.load(f"{W97}/valid_mask.npy")
F = np.load(f"{B}/work/batform2.npy")
FCOLS = json.load(open(f"{B}/work/batform2_cols.json"))["cols"]
NUM94 = json.load(open(f"{H}/audit_deep/pkg/DEEP/mate/model/deep/prep.json",
                       encoding="utf-8"))["NUM"]
CATN = [f"c{j}" for j in range(5)]
d = {NUM94[j]: X[:, j] for j in range(94)}
for j in range(5):
    d[CATN[j]] = C[:, j].astype(np.int32)
for j, c in enumerate(FCOLS):
    d[c] = F[:, j]
D = pd.DataFrame(d)
m = CatBoostClassifier(iterations=3000, depth=6, learning_rate=0.05,
                       loss_function="Logloss", task_type="GPU", devices=DEV,
                       random_seed=42, verbose=0, thread_count=3,
                       l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8)
t0 = time.time()
m.fit(Pool(D[trm], y[trm], cat_features=CATN))
p = m.predict_proba(Pool(D[vam], cat_features=CATN))[:, 1]
imp = m.get_feature_importance(Pool(D[vam], y[vam], cat_features=CATN), type="LossFunctionChange")
names = list(D.columns)
rk = sorted(zip(names, [float(v) for v in imp]), key=lambda t: -t[1])
R = {"fit_sec": round(time.time() - t0, 1),
     "score_1seed": 100000.0 * (1 - float(np.mean((p - y[vam]) ** 2)) / REF),
     "top30_all": rk[:30],
     "new_cols_ranked": sorted([(n, v) for n, v in rk if n in FCOLS],
                               key=lambda t: -t[1]),
     "new_cols_rank_positions": {n: int([q[0] for q in rk].index(n)) + 1
                                 for n in FCOLS},
     "n_features": len(names)}
json.dump(R, open(f"{B}/out/s1_imp.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
