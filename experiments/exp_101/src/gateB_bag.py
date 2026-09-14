"""문맥 배깅 — 서로 겹치지 않는 train(<=2023) 문맥 B 개의 로짓 평균.
TabPFN 이 1.1M 행 중 20k 만 본다는 것이 약점인지 시험한다. 비용은 B 배로 정확히 선형.
"""
import os, json, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--n_ctx", type=int, default=20000); ap.add_argument("--n_est", type=int, default=1)
ap.add_argument("--bags", type=int, default=5); ap.add_argument("--chunk", type=int, default=40000)
ap.add_argument("--ckpt", default="ckpt/tabpfn-v2-classifier-v2_default.ckpt")
ap.add_argument("--tag", default="bag"); ap.add_argument("--save_p", default="")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(a.threads)
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"]="1"
import numpy as np, torch
torch.set_num_threads(a.threads)
from scipy.special import logit, expit
from tabpfn import TabPFNClassifier
H="/home/<user>/SungjinPark"; B=f"{H}/exp101"
X=np.load(f"{H}/exp096/work/X_f64_oot23.npy",mmap_mode="r"); y=np.load(f"{H}/exp096/workb/y.npy")
trm=np.load(f"{H}/exp089/work/trm_oot23.npy"); vam=np.load(f"{H}/exp089/work/valid_mask.npy")
tr=np.flatnonzero(trm); va=np.flatnonzero(vam)
Xq=np.ascontiguousarray(X[va]).astype(np.float32); yq=y[va].astype(np.int64)
rng=np.random.default_rng(2026)
perm=rng.permutation(tr)                      # 배깅 문맥은 서로 겹치지 않게 자른다
Z=np.zeros(len(va)); per=[]
t0=time.time()
for b in range(a.bags):
    ci=np.sort(perm[b*a.n_ctx:(b+1)*a.n_ctx])
    clf=TabPFNClassifier(device=a.device,model_path=f"{B}/{a.ckpt}",n_estimators=a.n_est,
                         ignore_pretraining_limits=True,random_state=42+b)
    clf.fit(np.ascontiguousarray(X[ci]).astype(np.float32), y[ci].astype(np.int64))
    p=np.concatenate([clf.predict_proba(Xq[s:s+a.chunk])[:,1] for s in range(0,len(Xq),a.chunk)])
    zb=logit(np.clip(p.astype(np.float64),1e-6,1-1e-6)); Z+=zb
    br=float(np.mean((p-yq)**2))
    per.append({"bag":b,"brier":br,"score":round(100000*(1-br/0.2498069),5),"sec":round(time.time()-t0,1)})
    print(json.dumps(per[-1]),flush=True)
Z/=a.bags; P=expit(Z)
br=float(np.mean((P-yq)**2))
R={"tag":a.tag,"n_ctx":a.n_ctx,"n_est":a.n_est,"bags":a.bags,"device":a.device,
   "total_context_rows":int(a.n_ctx*a.bags),"wall_sec":round(time.time()-t0,1),
   "per_bag":per,"bag_mean_brier":br,"bag_mean_score":round(100000*(1-br/0.2498069),5),
   "n_query":int(len(va))}
if a.save_p: np.save(f"{B}/work/{a.save_p}", P)
json.dump(R,open(f"{B}/out/gateB_{a.tag}.json","w"),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
