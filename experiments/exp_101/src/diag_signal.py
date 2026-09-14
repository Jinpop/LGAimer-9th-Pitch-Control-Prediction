"""문맥 크기에 따른 TabPFN 신호 진단 + 같은 문맥의 로지스틱 회귀 대조."""
import os, sys, json, time, argparse
ap=argparse.ArgumentParser()
ap.add_argument("--device",default="cuda:0"); ap.add_argument("--threads",type=int,default=6)
ap.add_argument("--n_query",type=int,default=30000); ap.add_argument("--n_est",type=int,default=4)
ap.add_argument("--ctxs",default="1024,4096,10000")
ap.add_argument("--ckpt",default="ckpt/tabpfn-v2-classifier-v2_default.ckpt")
a=ap.parse_args()
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]=str(a.threads)
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"]="1"
import numpy as np, torch
torch.set_num_threads(a.threads)
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
H="/home/<user>/SungjinPark"; B=f"{H}/exp101"
X=np.load(f"{H}/exp096/work/X_f64_oot23.npy",mmap_mode="r"); y=np.load(f"{H}/exp096/workb/y.npy")
trm=np.load(f"{H}/exp089/work/trm_oot23.npy"); vam=np.load(f"{H}/exp089/work/valid_mask.npy")
tr=np.flatnonzero(trm); va=np.flatnonzero(vam)[:a.n_query]
Xq=np.ascontiguousarray(X[va]).astype(np.float32); yq=y[va].astype(np.int64)
base=float(np.mean((yq.mean()-yq)**2))
out={"base_brier":base,"base_rate":float(yq.mean()),"n_query":int(len(va)),"rows":[]}
S=lambda b: round(100000*(1-b/0.2498069),5)
for nc in [int(x) for x in a.ctxs.split(",")]:
    rng=np.random.default_rng(42)
    ci=np.sort(rng.choice(tr,size=nc,replace=False))
    Xc=np.ascontiguousarray(X[ci]).astype(np.float32); yc=y[ci].astype(np.int64)
    clf=TabPFNClassifier(device=a.device,model_path=f"{B}/{a.ckpt}",n_estimators=a.n_est,
                         ignore_pretraining_limits=True,random_state=42)
    t0=time.time(); clf.fit(Xc,yc)
    P=np.concatenate([clf.predict_proba(Xq[s:s+20000]) for s in range(0,len(Xq),20000)])[:,1]
    dt=time.time()-t0
    lr=LogisticRegression(max_iter=2000,C=1.0).fit(Xc,yc); pl=lr.predict_proba(Xq)[:,1]
    r={"n_ctx":nc,"n_est":a.n_est,"sec":round(dt,1),
       "tabpfn_brier":float(np.mean((P-yq)**2)),"tabpfn_auc":float(roc_auc_score(yq,P)),
       "lr_brier":float(np.mean((pl-yq)**2)),"lr_auc":float(roc_auc_score(yq,pl))}
    r["tabpfn_score"]=S(r["tabpfn_brier"]); r["lr_score"]=S(r["lr_brier"]); r["base_score"]=S(base)
    out["rows"].append(r); print(json.dumps(r),flush=True)
json.dump(out,open(f"{B}/out/diag_signal.json","w"),indent=1)
print(json.dumps(out,indent=1))
