"""exp_104 축B1 — TabPFN 에 **열을 줄여** 먹인다. exp_101 은 94열을 통째로 먹였다.

두 관문을 동시에 미는 유일한 손잡이라는 것이 config 의 논거다: 열이 줄면 ① TabPFN v2 의
열 수 약점이 완화되고 ② 질의 시간이 준다. **둘 다 측정한다.**

🔴 열 순위는 **≤2023 학습 행에서만** 낸다 — 2024(평가면)를 안 본다.
   LightGBM gain 중요도(고정 시드·고정 부분표본). 순위 산출은 한 번이고 그 뒤 동결된다.
🔴 시간은 채점 서버 L4 환산까지 적는다: A5000 -> L4 보수 계수 **2.94배**(S164), 예산 여유 484초.
🔴 문맥은 train(<=2023)에서 고정 시드로 뽑아 동결하고, 질의·문맥 교집합 0 을 assert 한다
   (exp_101 규칙 2-4 구조 승계).
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--n_ctx", type=int, default=20000)
ap.add_argument("--n_est", type=int, default=1)
ap.add_argument("--chunk", type=int, default=40000)
ap.add_argument("--ckpt", default="ckpt/tabpfn-v2-classifier-v2_default.ckpt")
ap.add_argument("--topk", default="20,30,40,50")
ap.add_argument("--rank_only", type=int, default=0)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(a.threads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
import numpy as np
import torch

torch.set_num_threads(a.threads)
from scipy.special import logit

H = "/home/<user>/SungjinPark"
B101, B = f"{H}/exp101", f"{H}/exp104"
os.makedirs(f"{B}/work", exist_ok=True)
os.makedirs(f"{B}/out", exist_ok=True)
X = np.load(f"{H}/exp096/work/X_f64_oot23.npy", mmap_mode="r")
y = np.load(f"{H}/exp096/workb/y.npy")
trm = np.load(f"{H}/exp089/work/trm_oot23.npy")
vam = np.load(f"{H}/exp089/work/valid_mask.npy")
tr, va = np.flatnonzero(trm), np.flatnonzero(vam)
NUM = json.load(open(f"{H}/audit_deep/pkg/DEEP/mate/model/deep/prep.json",
                     encoding="utf-8"))["NUM"]
assert X.shape[1] == 94 == len(NUM)

# ── 열 순위: ≤2023 만. 한 번 내고 동결한다. ─────────────────────────────
RANK_P = f"{B}/work/colrank_le2023.json"
if os.path.exists(RANK_P):
    RK = json.load(open(RANK_P))
else:
    import lightgbm as lgb
    rs = np.random.default_rng(20260826)
    sub = np.sort(rs.choice(tr, 300000, replace=False))
    ds = lgb.Dataset(np.ascontiguousarray(X[sub]).astype(np.float32),
                     label=y[sub].astype(np.float64), feature_name=list(NUM))
    bst = lgb.train({"objective": "binary", "learning_rate": 0.05,
                     "num_leaves": 63, "min_data_in_leaf": 200, "verbose": -1,
                     "seed": 42, "num_threads": a.threads,
                     "feature_fraction": 1.0, "bagging_fraction": 1.0},
                    ds, num_boost_round=300)
    gain = bst.feature_importance("gain")
    ordr = np.argsort(-gain).tolist()
    RK = {"method": "LightGBM gain, 300 rounds, 300k sampled <=2023 rows, seed 42",
          "🔴_no_2024": "순위 산출에 평가면(2024) 행·라벨을 쓰지 않았다",
          "n_sub": int(len(sub)), "order_idx": ordr,
          "order_name": [NUM[i] for i in ordr],
          "gain": [float(gain[i]) for i in ordr]}
    json.dump(RK, open(RANK_P, "w"), ensure_ascii=False, indent=1)
print("rank top10:", RK["order_name"][:10], flush=True)
if a.rank_only:
    raise SystemExit("RANK_ONLY_DONE")
from tabpfn import TabPFNClassifier                                # noqa: E402

rng = np.random.default_rng(2026)
ci = np.sort(rng.permutation(tr)[:a.n_ctx])          # exp_101 과 같은 문맥 뽑기
assert len(np.intersect1d(ci, va)) == 0, "문맥·질의 교집합이 0 이 아니다"
yq = y[va].astype(np.int64)
REP = {"exp_id": "exp_104", "stage": "sB1_cols", "n_ctx": a.n_ctx,
       "n_est": a.n_est, "chunk": a.chunk, "device": a.device,
       "n_query": int(len(va)), "colrank": RANK_P,
       "L4_factor": 2.94, "L4_budget_sec": 484.0, "rows": []}
for k in [int(x) for x in a.topk.split(",")]:
    cols = sorted(RK["order_idx"][:k])
    Xc = np.ascontiguousarray(X[np.ix_(ci, cols)]).astype(np.float32)
    Xq = np.ascontiguousarray(X[np.ix_(va, cols)]).astype(np.float32)
    clf = TabPFNClassifier(device=a.device, model_path=f"{B101}/{a.ckpt}",
                           n_estimators=a.n_est, ignore_pretraining_limits=True,
                           random_state=42)
    t0 = time.time()
    clf.fit(Xc, y[ci].astype(np.int64))
    t_fit = time.time() - t0
    t1 = time.time()
    p = np.concatenate([clf.predict_proba(Xq[s:s + a.chunk])[:, 1]
                        for s in range(0, len(Xq), a.chunk)])
    t_pred = time.time() - t1
    np.save(f"{B}/work/p_tabB1_top{k}.npy", p.astype(np.float64))
    br = float(np.mean((p - yq) ** 2))
    tot = t_fit + t_pred
    sc245 = tot * 245789.0 / len(va)
    r = {"topk": k, "cols_idx": cols, "cols_name": [NUM[i] for i in cols],
         "brier": br, "alone_score_raw": 100000.0 * (1 - br / 0.2498069),
         "z_sd": float(np.std(logit(np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)))),
         "t_fit_sec": round(t_fit, 2), "t_pred_sec": round(t_pred, 2),
         "t_total_sec": round(tot, 2),
         "a5000_sec_245789rows": round(sc245, 2),
         "L4_conservative_sec": round(sc245 * 2.94, 1),
         "fits_L4_budget_484s": bool(sc245 * 2.94 <= 484.0)}
    REP["rows"].append(r)
    print(json.dumps({q: r[q] for q in r if not q.startswith("cols")},
                     ensure_ascii=False), flush=True)
json.dump(REP, open(f"{B}/out/sB1_cols.json", "w"), ensure_ascii=False, indent=1)
print("SB1_DONE")
