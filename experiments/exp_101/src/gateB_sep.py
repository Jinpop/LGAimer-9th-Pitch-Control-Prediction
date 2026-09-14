"""exp_101 — 규칙 2-4 내용 분리 검사 (TabPFN in-context 경로).

주장: 문맥은 train(<=2023) 동결표에서만 오고, 질의 행의 예측은 *다른 질의 행의 내용* 에
      의존하지 않는다.
검사: KEEP 행 8개를 고정하고, 나머지 질의 행을 통째로 다른 행으로 갈아치운 뒤
      KEEP 행의 예측이 정확히 0.0 만큼 움직이는지 본다(S28/AL·AM 계열 내용/위치 분리).
추가: 같은 KEEP 행을 단독(배치 1)으로도 예측해 비교한다.
"""
import os, json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--n_ctx", type=int, default=20000)
ap.add_argument("--n_est", type=int, default=1)
ap.add_argument("--batch", type=int, default=2000)
ap.add_argument("--ckpt", default="ckpt/tabpfn-v2-classifier-v2_default.ckpt")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "6"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
import numpy as np, torch
torch.set_num_threads(6)
from tabpfn import TabPFNClassifier

H = "/home/<user>/SungjinPark"; B = f"{H}/exp101"
X = np.load(f"{H}/exp096/work/X_f64_oot23.npy", mmap_mode="r")
y = np.load(f"{H}/exp096/workb/y.npy")
trm = np.load(f"{H}/exp089/work/trm_oot23.npy"); vam = np.load(f"{H}/exp089/work/valid_mask.npy")
tr = np.flatnonzero(trm); va = np.flatnonzero(vam)
rng = np.random.default_rng(42)
ci = np.sort(rng.choice(tr, size=a.n_ctx, replace=False))
Xc = np.ascontiguousarray(X[ci]).astype(np.float32); yc = y[ci].astype(np.int64)

KEEP = va[:8]                      # 고정할 8행
FILL_A = va[8:a.batch]             # 프레임 A 의 나머지
FILL_B = va[-(a.batch - 8):]       # 프레임 B 의 나머지 — A 와 한 행도 겹치지 않는다
assert np.intersect1d(FILL_A, FILL_B).size == 0
assert np.intersect1d(KEEP, FILL_B).size == 0

clf = TabPFNClassifier(device=a.device, model_path=f"{B}/{a.ckpt}", n_estimators=a.n_est,
                       ignore_pretraining_limits=True, random_state=42)
clf.fit(Xc, yc)
def run(idx):
    return clf.predict_proba(np.ascontiguousarray(X[idx]).astype(np.float32))[:, 1]

fa = np.concatenate([KEEP, FILL_A]); fb = np.concatenate([KEEP, FILL_B])
pa, pb = run(fa), run(fb)
solo = np.array([run(KEEP[i:i+1])[0] for i in range(len(KEEP))])
shuf = np.concatenate([FILL_A[:50], KEEP, FILL_A[50:]])   # KEEP 위치만 이동
ps = run(shuf); ps_keep = ps[50:58]

R = {"n_ctx": int(a.n_ctx), "n_est": a.n_est, "device": a.device, "batch": a.batch,
     "content_test_max_abs_diff": float(np.max(np.abs(pa[:8] - pb[:8]))),
     "position_test_max_abs_diff": float(np.max(np.abs(pa[:8] - ps_keep))),
     "solo_vs_batch_max_abs_diff": float(np.max(np.abs(pa[:8] - solo))),
     "keep_p_frameA": [float(v) for v in pa[:8]],
     "keep_p_frameB": [float(v) for v in pb[:8]],
     "PASS_content_separation": bool(np.max(np.abs(pa[:8] - pb[:8])) == 0.0)}
json.dump(R, open(f"{B}/out/gateB_sep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
