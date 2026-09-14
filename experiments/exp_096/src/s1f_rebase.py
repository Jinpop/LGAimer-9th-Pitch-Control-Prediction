"""exp_096 s1f — 🔴 기준선을 바꿔 다시 판정 (post-hoc 진단, 사전 등록 판정 아님 — S8).

정직면의 z_deep 이 float32 표준화 결함 위에 있고 **배포 딥 팔은 그렇지 않다**는 것이
이 실험에서 밝혀졌다.  그러면 물어야 할 것은 바뀐다:

    (기존)  결함 있는 딥 팔 옆에 새 팔을 꽂으면 오르는가            -> +7.29 (rank), +8.37 (mlp64)
    (진짜)  **결함을 고친 딥 팔** 옆에 새 팔을 꽂으면 오르는가       <- 이 파일

z_deep 자리에 mlp64(= 같은 레시피·같은 5시드, float64 표준화)를 넣고 격자를 다시 돈다.
2팔 기준점도 그 면에서 새로 구한다 (w2=0 단면의 argmax).
"""
import json, os, time
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="2"
import numpy as np, pandas as pd
from scipy.special import expit, logit
H="/home/<user>/SungjinPark"; B90,B96=f"{H}/exp090",f"{H}/exp096"
OOT,OOT3,OOT4=f"{H}/exp069/oot",f"{H}/exp070/oot3",f"{H}/exp071/out"
EPS,REF=1e-6,0.2498069; K=100000.0/REF
S5=[42,777,2024,101,7]; AA=(42,7,2024)
W4={"LGB":0.05,"CAT":0.68,"NN":0.15,"LEAF":0.12}
BO=dict(alpha=1.07911,beta_skew=-0.059151,c_aa=0.25244,c_T70=0.04756,w_out=0.7,
        w_v6_inner=0.6668397489,w_ebm=-0.1,beta_ins=0.03)
lg=lambda p: logit(np.clip(np.asarray(p,np.float64),EPS,1-EPS))
pts=lambda b: 100000.0*(1.0-b/REF); t0=time.time()
tro=pd.read_csv(f"{H}/data/train.csv",usecols=["row_id","season","pitcher_id","control_success"])
tro=tro[tro.season==2024].sort_values("row_id",kind="stable")
y=tro.control_success.to_numpy(np.float64); pit=tro.pitcher_id.to_numpy(); YB=float(y.mean())
V={k:np.load(p).astype(np.float64) for k,p in {
 "p_lgb":f"{OOT}/p_lgb_oot2024.npy","p_cat":f"{OOT}/p_cat_oot2024.npy",
 "p_leaf":f"{OOT}/p_leaf_oot2024.npy","p_nn":f"{OOT4}/p_nn_oot2024.npy",
 "p_q":f"{OOT}/p_q_oot2024.npy","p_ebm":f"{OOT3}/p_ebm_oot2024.npy",
 "off":f"{OOT}/off_oot2024.npy"}.items()}
LV=sum(W4[t]*lg(V[k]) for t,k in (("LGB","p_lgb"),("CAT","p_cat"),("NN","p_nn"),("LEAF","p_leaf")))
L_aa=np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy")) for s in AA],0)
L_t70=np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy")) for s in AA],0)
Z_MATE=((1-BO["w_ebm"])*(BO["w_v6_inner"]*LV+(1-BO["w_v6_inner"])*lg(V["p_q"]))
        +BO["w_ebm"]*lg(V["p_ebm"])+BO["beta_ins"]*V["off"])
FIX=BO["c_aa"]*L_aa+BO["c_T70"]*L_t70
def pin(z,f):
    c=0.0
    for _ in range(200):
        p=f(z,c); g=float(p.mean())-YB
        if abs(g)<1e-15: break
        c-=g/float((p*(1-p)).mean())
    return c
F_ABD=lambda z,c: expit(BO["alpha"]*z+BO["beta_skew"]*z*z+c)
prob=lambda z: F_ABD(z,pin(z,F_ABD))
sc=lambda z: pts(float(np.mean((prob(z)-y)**2)))
rng=np.random.default_rng(20260824)
uq,inv=np.unique(pit,return_inverse=True); nb=len(uq)
order=np.argsort(inv,kind="stable")
st=np.searchsorted(inv[order],np.arange(nb)); en=np.append(st[1:],len(order))
BS=(en-st).astype(np.float64); PK=np.stack([rng.integers(0,nb,nb) for _ in range(1000)])
IDX=[order[st[j]:en[j]] for j in range(nb)]
def boot(d):
    b_=np.array([d[IDX[j]].sum() for j in range(nb)]); bs=b_[PK].sum(1)/BS[PK].sum(1)
    se=float(bs.std(ddof=1)); ob=float(d.mean())
    return {"gain_pts":-ob*K,"t":float(-ob/se) if se>0 else None,
            "p_share_new_worse":float((bs>0).mean())}
ld=lambda tag: np.mean([np.load(f"{B96}/out/preds/z_{tag}_oot23_s{s}.npy") for s in S5],0)
DEFECT=np.mean([np.load(f"{B90}/out/preds/z_K1632_oot23_s{s}.npy") for s in S5],0)
BASE64=ld("mlp64")
z0=BO["w_out"]*Z_MATE+FIX; b0=sc(z0)
def root(w,w2,dz,nz): return BO["w_out"]*((1-w-w2)*Z_MATE+w*dz+w2*nz)+FIX
WG=[round(x,4) for x in np.arange(0.0,0.4401,0.02)]
W2G=[round(x,4) for x in np.arange(0.0,0.4001,0.02)]
R={"exp_id":"exp_096","stage":"s1f_rebaselined",
   "🔴":"post-hoc 진단이다. 사전 등록 판정(S8)은 out/s1_arch_partial.json 이 정본이다.",
   "질문":"결함을 고친 딥 팔(mlp64)을 기준선으로 놓아도 새 팔이 +2.5 를 넘는가",
   "baseline_no_deep":b0}
for bname,BZ in (("deep_defective_f32",DEFECT),("deep_fixed_f64",BASE64)):
    cut=[{"w":w,"gain":sc(root(w,0.0,BZ,BZ))-b0} for w in WG]
    ref=max(cut,key=lambda c:c["gain"])
    P_ref=prob(root(ref["w"],0.0,BZ,BZ)); SQr=(P_ref-y)**2
    blk={"two_arm_ref":ref,"two_arm_curve_argmax_w":ref["w"],"arms":{}}
    for a in ("rank","rankT816","mlp64","cos","ctrl35","seqA"):
        if a==bname.split("_")[-1]: continue
        try: NZ=ld(a)
        except Exception: continue
        G={}
        for w in WG:
            for w2 in W2G:
                if w+w2>0.9: continue
                G[(w,w2)]=sc(root(w,w2,BZ,NZ))-b0
        bw=max(G,key=lambda t:G[t]); resid=G[bw]-ref["gain"]
        bt=boot((prob(root(bw[0],bw[1],BZ,NZ))-y)**2-SQr)
        blk["arms"][a]={"argmax_w":bw[0],"argmax_w2":bw[1],"gain":G[bw],
                        "residual_vs_two_arm_ref":resid,"t":bt["t"],
                        "p_share_new_worse":bt["p_share_new_worse"],
                        "PASS_same_rule":bool(resid>=2.5 and bt["t"] is not None and bt["t"]>=2.0)}
        print(bname,a,"resid %+.4f t %s"%(resid,("%.4f"%bt["t"]) if bt["t"] else "None"),flush=True)
    R[bname]=blk
R["🔴_읽기"]=("두 블록을 나란히 본다. 결함 있는 기준선에서 크던 잔여가 고친 기준선에서 "
              "줄어들면, 그 이득은 '새 팔'이 아니라 '기준 팔의 결함'이었다는 뜻이다.")
R["wall_sec"]=round(time.time()-t0,1)
json.dump(R,open(f"{B96}/out/s1f_rebase.json","w"),ensure_ascii=False,indent=1)
print(json.dumps({k:v for k,v in R.items() if k in ("deep_defective_f32","deep_fixed_f64")},
                 ensure_ascii=False,indent=1))
