# exp_111 stage F — 2축 동시해 · 지속성 진단 · k 격자 민감도 · 귀무 유효자유도
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="2"
import glob, json, time
import numpy as np, pandas as pd
from scipy.special import logit
H="/home/<user>/SungjinPark"; OUT=f"{H}/exp111"
REF,EPS=0.2498069,1e-6; K_PTS=100000.0/REF; ALPHA=1.07911
BETA={"R":0.597044,"F":0.492041}; K_PERS=400
lg=lambda p: logit(np.clip(np.asarray(p,np.float64),EPS,1-EPS))
tr=pd.read_csv(f"{H}/data/train.csv",encoding="utf-8-sig",
   usecols=["row_id","season","game_type","control_success","pitcher_id","batter_id",
            "batter_hand","pitcher_hand","batter_team_id"])
row=tr[tr.season==2024].reset_index(drop=True); N=len(row)
r_oot=np.load(f"{OUT}/r_oot.npy"); r67=np.load(f"{OUT}/r67.npy"); r_cs3=np.load(f"{OUT}/r_cs3.npy")
p_oot=np.load(f"{OUT}/p_oot.npy"); W=np.load(f"{OUT}/dir_w.npy")
c_pre=ALPHA*p_oot*(1-p_oot)
AXN=["batter_id","batter_id_x_pitcher_hand","batter_team"]
def vtx(d,res):
    d=d-d.mean(); G=float(np.mean(d*d)); L=float(np.mean(d*res))
    return dict(K_L=K_PTS*L,K_G=K_PTS*G,vertex_pts=K_PTS*L*L/G if G>0 else 0.,u_star=L/G if G>0 else 0.)
def joint(vs,res):
    Vm=np.array([v-v.mean() for v in vs]); L=K_PTS*(Vm@res)/N; G=K_PTS*(Vm@Vm.T)/N
    w,Q=np.linalg.eigh(G); keep=w>w.max()*1e-10
    u=Q[:,keep]@(Q[:,keep].T@L/w[keep]); return float(2*L@u-u@G@u),int(keep.sum())
F={}
pair=[c_pre*W[0],c_pre*W[1]]
jg,rk=joint(pair,r_oot); jg67,_=joint(pair,r67)
F["joint_two_surviving"]={"axes":AXN[:2],"ourfaceB_joint_vertex_pts":jg,"rank":rk,
  "naive_sum":vtx(pair[0],r_oot)["vertex_pts"]+vtx(pair[1],r_oot)["vertex_pts"],
  "rho":float(np.corrcoef(W[0],W[1])[0,1]),"ref67_joint_vertex_pts":jg67}
# ── 지속성 진단 : <=2023 표 vs 2024 타자평균 잔차 (가중 상관) ──
bat=row.batter_id.to_numpy(); ub,bi=np.unique(bat,return_inverse=True); nb=np.bincount(bi).astype(float)
tabs=[]
for s in (2022,2023):
    p=np.mean([np.load(x) for x in sorted(glob.glob(
        f"{H}/experiments/exp_030/pred/pred_C5B3_u{s}_seed*_st2.npy"))],axis=0)
    cur=tr[tr.season==s].copy(); past=tr[tr.season<=s-1]
    ov=past.groupby("pitcher_id").control_success.mean()
    cell=past.groupby(["pitcher_id","batter_hand"]).control_success.agg(["sum","count"])
    sm,nm=cell["sum"].to_dict(),cell["count"].to_dict()
    key=list(zip(cur.pitcher_id,cur.batter_hand)); hit=np.array([k in nm for k in key])
    o=np.nan_to_num(cur.pitcher_id.map(ov).to_numpy())
    s_=np.array([sm.get(k,0.0) for k in key],float); n_=np.array([nm.get(k,0) for k in key],float)
    d_s=(s_-n_*o)/(n_+K_PERS)
    f_=past.groupby(["pitcher_hand","batter_hand"]).control_success.mean()
    c_=past.groupby("pitcher_hand").control_success.mean()
    Lh=np.nan_to_num(np.array([f_.get((a,b),np.nan)-c_.get(a,np.nan) for a,b in zip(cur.pitcher_hand,cur.batter_hand)]))
    off=np.where(hit,d_s,Lh); bt=np.array([BETA.get(g,0.59072) for g in cur.game_type])
    r=cur.control_success.to_numpy(float)-np.clip(p+bt*off,EPS,1-EPS)
    cur["resid"]=r-r.mean(); tabs.append(cur[["batter_id","resid"]])
TAB=pd.concat(tabs,ignore_index=True)
k0=json.load(open(f"{OUT}/sB2.json"))["prereg_k"]["batter_id"]["k"]
uid,idx=np.unique(TAB.batter_id.to_numpy(),return_inverse=True)
n_=np.bincount(idx,minlength=len(uid)).astype(float); s_=np.bincount(idx,weights=TAB.resid.to_numpy(),minlength=len(uid))
tabval=dict(zip(uid.tolist(),(s_/(n_+k0)).tolist()))
wb=np.array([tabval.get(b,0.0) for b in ub])
PER={}
for nm2,rv in (("A_67col_AA_arm_OOT",r67),("B_honest_full_blend_OOT",r_oot),("C_CS3_calframe_insample",r_cs3)):
    m=np.bincount(bi,weights=rv)/nb
    cw=float(np.average(wb*m,weights=nb)); sw=float(np.sqrt(np.average(wb**2,weights=nb)*np.average(m**2,weights=nb)))
    PER[nm2]={"weighted_cov_table_vs_2024batter_mean":cw,
              "weighted_corr":cw/sw if sw>0 else None,
              "weighted_sd_2024_batter_mean":float(np.sqrt(np.average(m**2,weights=nb)))}
PER["🔑"]="표(<=2023)와 2024 타자평균 잔차의 가중 공분산. 잔차 총분산은 거의 안 변하는데 이 공분산이 줄면 = 모델이 ★예측 가능한★ 타자 성분만 흡수한 것."
F["persistence"]=PER
# ── k 격자 민감도 (선택 아님 · 정직면 B) ──
KG={}
for i,name in enumerate(AXN):
    cols={"batter_id":["batter_id"],"batter_id_x_pitcher_hand":["batter_id","pitcher_hand"],
          "batter_team":["batter_team_id"]}[name]
    tab2=[]
    for s in (2022,2023):
        pass
    KG[name]={}
KG["note"]="stage B 에서 CS3 면 격자를 이미 냈다(sB.json). 정직면 격자는 batter_id 만 아래에 낸다."
tabg=TAB.batter_id.to_numpy()
for kk in [20,60,200,600,2000,k0]:
    val=s_/(n_+kk); pos=pd.Index(uid).get_indexer(row.batter_id.to_numpy())
    raw=np.where(pos>=0,val[np.maximum(pos,0)],0.0)
    if raw.std()>0:
        KG.setdefault("batter_id_ourfaceB_vertex_by_k",{})[("prereg_%.1f"%kk) if abs(kk-k0)<1e-9 else str(kk)]=\
            vtx(c_pre*(raw-raw.mean())/raw.std(),r_oot)["vertex_pts"]
F["k_grid_sensitivity"]=KG
# ── 귀무 유효자유도 ──
sC=json.load(open(f"{OUT}/sC.json"))
unit=K_PTS*float(np.mean(r_oot**2))/N
F["null_effective_df"]={"one_df_expected_vertex_pts":unit}
for t in sC:
    for n in sC[t]:
        F["null_effective_df"][f"{t}|{n}"]={"null_mean_pts":sC[t][n]["B_honest_full_blend_OOT"]["mean"],
            "implied_eff_df":sC[t][n]["B_honest_full_blend_OOT"]["mean"]/unit}
json.dump(F,open(f"{OUT}/sF.json","w"),ensure_ascii=False,indent=1)
print(json.dumps({"joint2":F["joint_two_surviving"],"persistence":PER,
                  "k_sens":KG.get("batter_id_ourfaceB_vertex_by_k"),
                  "eff_df":F["null_effective_df"]},ensure_ascii=False,indent=1))
print("STAGE_F_OK")
