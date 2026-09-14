"""exp_071 부록 — k 민감도 표를 읽을 수 있게 절대 점수까지 채운다.
(Δ 만 있으면 '나쁜 NN 인데 Δ 가 크다'가 왜 그런지 안 보인다 — 기준점이 함께 움직인다.)"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"
import json, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
H="/home/<user>/SungjinPark"; E=f"{H}/exp071"
OOT,OOT3,OUT=f"{H}/exp069/oot",f"{H}/exp070/oot3",f"{E}/out"
EPS=1e-6; REF=0.2498069
DEP=np.array([0.5,-0.10,0.03,0.7,0.25244,0.04756])
lg=lambda p: logit(np.clip(np.asarray(p,np.float64),EPS,1-EPS))
conv=lambda b:100000.0*(1.0-b/REF)
tr=pd.read_csv(f"{H}/data/train.csv",usecols=["row_id","season","control_success"])
tr=tr[tr.season==2024].sort_values("row_id",kind="stable").reset_index(drop=True)
y=tr.control_success.to_numpy(np.float64); ybar=float(y.mean())
V={k:np.load(f"{OOT}/{f}").astype(np.float64) for k,f in
   [("p_lgb","p_lgb_oot2024.npy"),("p_cat","p_cat_oot2024.npy"),
    ("p_leaf","p_leaf_oot2024.npy"),("p_q","p_q_oot2024.npy"),
    ("p_v6_3arm","p_v6_3arm_renorm_oot2024.npy"),("off","off_oot2024.npy")]}
V["p_ebm"]=np.load(f"{OOT3}/p_ebm_oot2024.npy").astype(np.float64)
AA=(42,7,2024)
L_aa=np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy").astype(np.float64)) for s in AA],0)
L_t70=np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy").astype(np.float64)) for s in AA],0)
L_q,L_ebm,off=lg(V["p_q"]),lg(V["p_ebm"]),V["off"]
def rc(z,c0=0.0):
    for _ in range(40):
        p=expit(z+c0); mu=float(p.mean())
        if abs(mu-ybar)<1e-14: break
        d=(mu-ybar)/float((p*(1-p)).mean()); c0-=d
        if abs(d)<1e-14: break
    return c0
def sz(z):
    c=rc(z); return conv(float(np.mean((expit(z+c)-y)**2)))
def vm(th):
    wv,we,be,wo,ca,ct=th
    return np.array([wo*(1-we)*wv,wo*(1-we)*(1-wv),wo*we,wo*be,ca,ct])
def pack(L_v6):
    X=np.column_stack([L_v6,L_q,L_ebm,off,L_aa,L_t70])
    s50=sz(X@vm(DEP)); th=DEP.copy(); th[0]=0.70; s70=sz(X@vm(th))
    f=lambda x:(lambda t:(np.mean((expit(X@vm(t)+rc(X@vm(t)))-y)**2)))(np.r_[x,DEP[1:]])
    r=minimize_scalar(f,bounds=(0.,1.),method="bounded",options={"xatol":1e-6})
    return {"v6팔_단독점수":sz(L_v6),"블렌드_w_v6=0.50":s50,"블렌드_w_v6=0.70":s70,
            "Δ_0.70":s70-s50,"조건부_argmax":float(r.x),
            "조건부_argmax_점수":conv(float(r.fun)),"조건부_argmax_Δ":conv(float(r.fun))-s50}
W=(0.05,0.68,0.15,0.12)
EP={s:np.load(f"{E}/cache/p24_epochs_seed{s}.npy").astype(np.float64) for s in (42,777,2024,101)}
TI=json.load(open(f"{OUT}/train071.json")); be={int(a):b for a,b in TI["best_epoch_per_seed"].items()}
R={"3종판":pack(lg(V["p_v6_3arm"]))}
def v6of(pnn):
    return lg(expit(W[0]*lg(V["p_lgb"])+W[1]*lg(V["p_cat"])+W[2]*lg(pnn)+W[3]*lg(V["p_leaf"])))
for kk in [2,4,6,8,10,12,14,16,20,25]:
    pk=expit(np.mean([lg(EP[s][kk-1]) for s in (42,777,2024,101)],0))
    R[f"k={kk}"]=dict(pack(v6of(pk)),nn팔_단독점수=sz(lg(pk)))
pb=expit(np.mean([lg(EP[s][be[s]-1]) for s in (42,777,2024,101)],0))
R["best_epoch판"]=dict(pack(v6of(pb)),nn팔_단독점수=sz(lg(pb)),epochs=be)
R["🔴읽는법"]=("Δ 는 '같은 곡선 위 두 점의 차'다. NN 이 나빠지면 **기준점(w_v6=0.50)이 먼저 내려가서** "
              "Δ 가 커 보일 수 있다. 절대 점수 열을 함께 보라 — 4팔 블렌드의 절대 점수는 k 와 무관하게 "
              "3종판(928.18)보다 낮다.")
json.dump(R,open(f"{OUT}/addendum.json","w"),indent=1,ensure_ascii=False)
for k_,v in R.items():
    if isinstance(v,dict):
        print(f"{k_:>14} nn단독 {v.get('nn팔_단독점수',float('nan')):8.2f} "
              f"v6팔 {v['v6팔_단독점수']:8.2f}  w.50 {v['블렌드_w_v6=0.50']:8.3f} "
              f"w.70 {v['블렌드_w_v6=0.70']:8.3f}  Δ {v['Δ_0.70']:+7.3f}  "
              f"argmax {v['조건부_argmax']:.4f} → {v['조건부_argmax_점수']:8.3f}",flush=True)
print("DONE_ADDENDUM")
