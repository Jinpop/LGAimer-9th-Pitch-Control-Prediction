"""exp_101 관문B — TabPFN 추론 시간 실측.

규칙 2-4: 문맥(in-context 학습 표)은 **train(<=2023) 행에서만** 고정 seed 로 뽑는다.
질의 행은 문맥에 절대 들어가지 않으며, 질의 청크는 서로를 참조하지 않는다(별도 검증: gateB_sep.py).
"""
import os, sys, json, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--n_ctx", type=int, default=4096)
ap.add_argument("--n_est", type=int, default=1)
ap.add_argument("--n_query", type=int, default=245789)
ap.add_argument("--chunk", type=int, default=20000)
ap.add_argument("--ckpt", default="ckpt/tabpfn-v2-classifier-v2_default.ckpt")
ap.add_argument("--budget_sec", type=float, default=0.0, help=">0 이면 그 시각에 중단하고 완료 행수를 기록")
ap.add_argument("--tag", default="run")
ap.add_argument("--save_p", default="")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = str(a.threads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import numpy as np, torch
torch.set_num_threads(a.threads)
from tabpfn import TabPFNClassifier

H = "/home/<user>/SungjinPark"
B = f"{H}/exp101"
X = np.load(f"{H}/exp096/work/X_f64_oot23.npy", mmap_mode="r")
y = np.load(f"{H}/exp096/workb/y.npy")
trm = np.load(f"{H}/exp089/work/trm_oot23.npy")     # <=2023
vam = np.load(f"{H}/exp089/work/valid_mask.npy")    # 2024

tr_idx = np.flatnonzero(trm)
va_idx = np.flatnonzero(vam)
rng = np.random.default_rng(a.seed)
ctx_idx = np.sort(rng.choice(tr_idx, size=min(a.n_ctx, tr_idx.size), replace=False))
q_idx = va_idx[: a.n_query]
assert np.intersect1d(ctx_idx, q_idx).size == 0, "문맥과 질의가 겹친다 (규칙 2-4)"

Xc = np.ascontiguousarray(X[ctx_idx]).astype(np.float32)
yc = y[ctx_idx].astype(np.int64)
Xq = np.ascontiguousarray(X[q_idx]).astype(np.float32)
yq = y[q_idx].astype(np.int64)

clf = TabPFNClassifier(device=a.device, model_path=f"{B}/{a.ckpt}",
                       n_estimators=a.n_est, auto_scale_n_estimators=False,
                       ignore_pretraining_limits=True, random_state=a.seed)

t0 = time.time(); clf.fit(Xc, yc); t_fit = time.time() - t0

P = np.empty((Xq.shape[0], 2), dtype=np.float64)
done = 0; t_pred = 0.0; aborted = False
tp0 = time.time()
for s in range(0, Xq.shape[0], a.chunk):
    e = min(s + a.chunk, Xq.shape[0])
    P[s:e] = clf.predict_proba(Xq[s:e])
    done = e
    t_pred = time.time() - tp0
    print(f"[{a.tag}] rows={done} t_pred={t_pred:.1f}s rate={done/t_pred:.1f}/s", flush=True)
    if a.budget_sec > 0 and t_pred > a.budget_sec:
        aborted = True; break

res = dict(tag=a.tag, device=a.device, threads=a.threads, n_ctx=int(ctx_idx.size),
           n_est=a.n_est, n_query=int(Xq.shape[0]), chunk=a.chunk, ckpt=a.ckpt,
           t_fit_sec=round(t_fit, 3), t_pred_sec=round(t_pred, 3),
           rows_done=int(done), aborted=aborted,
           t_total_sec=round(t_fit + t_pred, 3),
           rows_per_sec=round(done / t_pred, 2) if t_pred > 0 else None)
if not aborted:
    p1 = np.clip(P[:, 1], 1e-15, 1 - 1e-15)
    res["brier"] = float(np.mean((p1 - yq) ** 2))
    res["score_conv"] = round(100000 * (1 - res["brier"] / 0.2498069), 5)
    res["logloss"] = float(-np.mean(yq * np.log(p1) + (1 - yq) * np.log(1 - p1)))
    if a.save_p:
        np.save(f"{B}/work/{a.save_p}", p1.astype(np.float64))
os.makedirs(f"{B}/out", exist_ok=True)
json.dump(res, open(f"{B}/out/gateB_{a.tag}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
