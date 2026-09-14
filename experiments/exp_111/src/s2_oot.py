# exp_111 stage B2/C/D/E — ★out-of-time 정직면★ 기반 재측정 + 귀무 + 타자 교차적합 + 직교성.
# 캘프레임(<=2024 학습 팔을 2024행에 적용)은 표본 안 기억이 크다(점수 1690 vs LB 1152).
# 배포는 out-of-time 이므로 L 은 OOT 면에서 읽어야 한다.  재학습 0 · 패키지 실행 0 · 2스레드.
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import glob, json, time
import numpy as np, pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"; OUT = f"{H}/exp111"
REF, EPS, K_PERS = 0.2498069, 1e-6, 400
K_PTS = 100000.0/REF
BETA = {"R":0.597044,"F":0.492041}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
W4 = {"LGB":0.05,"CAT":0.68,"NN":0.15,"LEAF":0.12}
S5 = [42,777,2024,101,7]; AA3 = (42,7,2024); W_DEEP = 0.40
OOT_, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
rng = np.random.default_rng(42); T0=time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
lg = lambda p: logit(np.clip(np.asarray(p,np.float64), EPS, 1-EPS))

USE = ["row_id","season","game_type","control_success","pitcher_id","batter_id",
       "batter_hand","pitcher_hand","batter_team_id","asof_pitcher_n",
       "asof_pitcher_success_rate","balls_before","strikes_before"]
tr = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig", usecols=USE)
ids = pd.read_csv(f"{H}/build_AC/work/tr2024/test.csv", encoding="utf-8-sig",
                  usecols=["row_id"])["row_id"].to_numpy()
cur24 = tr[tr.season==2024].reset_index(drop=True)
assert (cur24.row_id.to_numpy()==ids).all()
y24 = cur24.control_success.to_numpy(np.float64); N=len(y24)
YB = float(y24.mean())
past24 = tr[tr.season<=2023]
log("loaded", N)
PERM_IS_IDENTITY=None

# ─────────── 기반 셋 ───────────
def anchored(p):
    q=np.clip(p,EPS,1-EPS); g=float(np.mean(y24-q)); b=np.clip(q+g,EPS,1-EPS)
    return b, y24-b
def score(b): return 100000.0*(1.0-np.mean((b-y24)**2)/REF)

p_cs3 = np.load(f"{H}/build_CU/work/calp_PAR.npy"); base_cs3, r_cs3 = anchored(p_cs3)

def deployed_offset(cur,past):
    ov = past.groupby("pitcher_id").control_success.mean()
    cell = past.groupby(["pitcher_id","batter_hand"]).control_success.agg(["sum","count"])
    s_map,n_map = cell["sum"].to_dict(), cell["count"].to_dict()
    key=list(zip(cur.pitcher_id,cur.batter_hand))
    hit=np.array([k in n_map for k in key])
    o=np.nan_to_num(cur.pitcher_id.map(ov).to_numpy())
    s_=np.array([s_map.get(k,0.0) for k in key],float); n_=np.array([n_map.get(k,0) for k in key],float)
    d_s=(s_-n_*o)/(n_+K_PERS)
    f=past.groupby(["pitcher_hand","batter_hand"]).control_success.mean()
    c=past.groupby("pitcher_hand").control_success.mean()
    L=np.nan_to_num(np.array([f.get((a,b),np.nan)-c.get(a,np.nan)
                              for a,b in zip(cur.pitcher_hand,cur.batter_hand)]))
    return np.where(hit,d_s,L)
off24=deployed_offset(cur24,past24)
beta24=np.array([BETA.get(g,0.59072) for g in cur24.game_type])
p67=np.mean([np.load(x) for x in sorted(glob.glob(
      f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed*_at1400.npy"))],axis=0)
base67, r67 = anchored(np.clip(p67+beta24*off24,EPS,1-EPS))

# ── 정직면(OOT, <=2023 학습) 현 블렌드 복제 — exp_102/103b 면 그대로 ──
rid = np.asarray(np.load(f"{OOT_}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, np.sort(cur24.row_id.to_numpy().astype(str))) or \
       np.array_equal(rid, cur24.row_id.to_numpy().astype(str))
perm = pd.Index(rid).get_indexer(cur24.row_id.to_numpy().astype(str))
assert (perm>=0).all()
V = {k: np.load(p).astype(np.float64)[perm] for k,p in {
     "p_lgb":f"{OOT_}/p_lgb_oot2024.npy","p_cat":f"{OOT_}/p_cat_oot2024.npy",
     "p_leaf":f"{OOT_}/p_leaf_oot2024.npy","p_nn":f"{OOT4}/p_nn_oot2024.npy",
     "p_q":f"{OOT_}/p_q_oot2024.npy","p_ebm":f"{OOT3}/p_ebm_oot2024.npy",
     "off":f"{OOT_}/off_oot2024.npy"}.items()}
LV = sum(W4[t]*lg(V[k]) for t,k in (("LGB","p_lgb"),("CAT","p_cat"),("NN","p_nn"),("LEAF","p_leaf")))
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
Z_MATE = ((1-BO["w_ebm"])*(BO["w_v6_inner"]*LV+(1-BO["w_v6_inner"])*LQ)
          + BO["w_ebm"]*LE + BO["beta_ins"]*V["off"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy")) for s in AA3],0)
L_t70= np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy")) for s in AA3],0)
def base94(seed):
    for b in (f"{H}/exp097", f"{H}/exp102"):
        p=f"{b}/out/preds/z_F1632_f64o_s{seed}.npy"
        if os.path.exists(p): return p
    raise FileNotFoundError(seed)
Z_DEEP = np.mean([np.load(base94(s)) for s in S5],0)
zm = (1-W_DEEP)*Z_MATE + W_DEEP*Z_DEEP
z  = BO["w_out"]*zm + BO["c_aa"]*L_aa + BO["c_T70"]*L_t70
def pin(zz):
    c=0.0
    for _ in range(200):
        p=expit(BO["alpha"]*zz+BO["beta_skew"]*zz*zz+c); g=float(p.mean())-YB
        if abs(g)<1e-15: break
        c-=g/float((p*(1-p)).mean())
    return c
p_oot = expit(BO["alpha"]*z+BO["beta_skew"]*z*z+pin(z))
base_oot, r_oot = anchored(p_oot)
BASES = {"A_67col_AA_arm_OOT":(base67,r67), "B_honest_full_blend_OOT":(base_oot,r_oot),
         "C_CS3_calframe_insample":(base_cs3,r_cs3)}
SC = {k: float(score(b)) for k,(b,_) in BASES.items()}
log("scores", SC)

# ─────────── 전파계수 ───────────
w_sp = np.load(f"{H}/exp109/work/w_unit.npy")
c_emp = (np.load(f"{H}/exp109/work/calp_CT.npy")-np.load(f"{H}/exp109/work/calp_CR.npy"))/(-0.014*w_sp)
c_of = {"A_67col_AA_arm_OOT": np.ones(N), "B_honest_full_blend_OOT": c_emp,
        "C_CS3_calframe_insample": c_emp}

# ─────────── 표 재료 ───────────
tabs=[]
for s in (2022,2023):
    p=np.mean([np.load(x) for x in sorted(glob.glob(
        f"{H}/experiments/exp_030/pred/pred_C5B3_u{s}_seed*_st2.npy"))],axis=0)
    cur=tr[tr.season==s].copy(); past=tr[tr.season<=s-1]
    off=deployed_offset(cur,past); bt=np.array([BETA.get(g,0.59072) for g in cur.game_type])
    r=cur.control_success.to_numpy(float)-np.clip(p+bt*off,EPS,1-EPS)
    cur["resid"]=r-r.mean(); cur["seas"]=s
    tabs.append(cur[["batter_id","pitcher_hand","batter_team_id","resid","seas"]])
TAB=pd.concat(tabs,ignore_index=True)
TR_RESID = TAB.resid.to_numpy(np.float64); TR_SEAS = TAB.seas.to_numpy()
AXES={"batter_id":["batter_id"],"batter_id_x_pitcher_hand":["batter_id","pitcher_hand"],
      "batter_team":["batter_team_id"]}
def gcodes(fr,cols):
    if len(cols)==1: return fr[cols[0]].to_numpy()
    return fr[cols[0]].to_numpy().astype(np.int64)*10+fr[cols[1]].to_numpy().astype(np.int64)
def eb_k(codes,res):
    uid,idx=np.unique(codes,return_inverse=True); G,Nn=len(uid),len(res)
    n=np.bincount(idx,minlength=G).astype(float); s=np.bincount(idx,weights=res,minlength=G)
    ss=np.bincount(idx,weights=res**2,minlength=G)
    s2w=float(ss.sum()-(s**2/n).sum())/(Nn-G)
    Q=float(np.sum(n*(s/n-res.mean())**2)); den=Nn-float(np.sum(n**2))/Nn
    s2b=(Q-(G-1)*s2w)/den
    return (float(s2w/s2b) if s2b>0 else float("inf")), dict(G=int(G),sigma2_within=s2w,sigma2_between=float(s2b))
def tdir(tc,tres,rc,k):
    uid,idx=np.unique(tc,return_inverse=True)
    n=np.bincount(idx,minlength=len(uid)).astype(float); s=np.bincount(idx,weights=tres,minlength=len(uid))
    val=s/(n+k); pos=pd.Index(uid).get_indexer(rc)
    return np.where(pos>=0, val[np.maximum(pos,0)], 0.0), pos>=0
def vtx(d,res):
    d=d-d.mean(); G=float(np.mean(d*d))
    if G<=0: return dict(K_L=0.,K_G=0.,vertex_pts=0.,u_star=0.)
    L=float(np.mean(d*res)); return dict(K_L=K_PTS*L,K_G=K_PTS*G,vertex_pts=K_PTS*L*L/G,u_star=L/G)
bat24=cur24.batter_id.to_numpy()
uidb,bidx=np.unique(bat24,return_inverse=True); NB=len(uidb)
PICKS=rng.integers(0,NB,size=(1000,NB))
def boot_t(d,res):
    d=d-d.mean(); prod=d*res
    bs=np.bincount(bidx,weights=prod,minlength=NB)
    se=float(bs[PICKS].sum(1).std(ddof=1))/N
    return (float(prod.mean())/se if se>0 else 0.0), se

RES={"bases":{"scores_pts":SC,"note":"C 는 <=2024 학습 팔을 2024행에 적용 = 표본 안 기억. 배포는 OOT 이므로 B 가 대표면."},
     "prereg_k":{}, "per_axis":{}, "w_deep_used":W_DEEP}
DIR={}
for name,cols in AXES.items():
    tc=gcodes(TAB,cols); rc=gcodes(cur24,cols)
    k,kd=eb_k(tc,TR_RESID); RES["prereg_k"][name]={"k":k,**kd}
    raw,hit=tdir(tc,TR_RESID,rc,k); w=(raw-raw.mean())/raw.std()
    DIR[name]=w
    rec={"coverage_f":float(hit.mean()),"raw_sd":float(raw.std())}
    for bn,(b,res) in BASES.items():
        v=vtx(c_of[bn]*w,res); t,se=boot_t(c_of[bn]*w,res)
        rec[bn]={**v,"t_of_L_batter_block":t}
    RES["per_axis"][name]=rec
    log(name,"k=%.1f"%k, {bn:round(rec[bn]["vertex_pts"],3) for bn in BASES},
        "t_B=%.2f"%rec["B_honest_full_blend_OOT"]["t_of_L_batter_block"])
# 동시해
def joint(vs,res):
    Vm=np.array([v-v.mean() for v in vs]); L=K_PTS*(Vm@res)/N; G=K_PTS*(Vm@Vm.T)/N
    ww,Q=np.linalg.eigh(G); keep=ww>ww.max()*1e-10
    u=Q[:,keep]@(Q[:,keep].T@L/ww[keep]); return float(2*L@u-u@G@u), int(keep.sum())
RES["joint_trio"]={}
for bn,(b,res) in BASES.items():
    jg,rk=joint([c_of[bn]*DIR[n] for n in AXES],res)
    RES["joint_trio"][bn]={"joint_vertex_pts":jg,"rank":rk,
        "naive_sum":sum(RES["per_axis"][n][bn]["vertex_pts"] for n in AXES)}
RES["joint_trio"]["pairwise_rho"]=np.corrcoef(np.array([DIR[n] for n in AXES])).tolist()
log("joint", {k:round(v["joint_vertex_pts"],3) for k,v in RES["joint_trio"].items() if isinstance(v,dict) and "joint_vertex_pts" in v})
json.dump(RES,open(f"{OUT}/sB2.json","w"),ensure_ascii=False,indent=1)
np.save(f"{OUT}/dir_w.npy",np.array([DIR[n] for n in AXES]))
np.save(f"{OUT}/r_oot.npy",r_oot); np.save(f"{OUT}/r_cs3.npy",r_cs3); np.save(f"{OUT}/r67.npy",r67)
np.save(f"{OUT}/c_emp.npy",c_emp); np.save(f"{OUT}/p_oot.npy",p_oot)
print("STAGE_B2_OK")

# ─────────── stage C : 귀무 둘 x 20 draw ───────────
NDRAW=20
NUL={"N1_identity_permute":{},"N2_resid_permute_within_season":{}}
for name,cols in AXES.items():
    tc=gcodes(TAB,cols); rc=gcodes(cur24,cols); k=RES["prereg_k"][name]["k"]
    v1={bn:[] for bn in BASES}; v2={bn:[] for bn in BASES}
    uid=np.unique(np.concatenate([np.unique(tc),np.unique(rc)]))
    for d in range(NDRAW):
        rg=np.random.default_rng(1000+d)
        # N1 : 표 항목 <-> 타자 대응을 무작위 치환 (그룹 크기 분포 보존)
        pm=rg.permutation(len(uid)); mp=dict(zip(uid.tolist(), uid[pm].tolist()))
        rc_p=np.array([mp[x] for x in rc])
        raw,_=tdir(tc,TR_RESID,rc_p,k)
        if raw.std()>0:
            w=(raw-raw.mean())/raw.std()
            for bn,(b,res) in BASES.items(): v1[bn].append(vtx(c_of[bn]*w,res)["vertex_pts"])
        # N2 : 표 재료 잔차를 시즌 안에서 치환 (구조·커버리지 완전 보존)
        rp=TR_RESID.copy()
        for s in (2022,2023):
            m=TR_SEAS==s; rp[m]=rg.permutation(rp[m])
        raw2,_=tdir(tc,rp,rc,k)
        if raw2.std()>0:
            w2=(raw2-raw2.mean())/raw2.std()
            for bn,(b,res) in BASES.items(): v2[bn].append(vtx(c_of[bn]*w2,res)["vertex_pts"])
    for tag,vv in (("N1_identity_permute",v1),("N2_resid_permute_within_season",v2)):
        NUL[tag][name]={bn:{"mean":float(np.mean(a)),"p95":float(np.percentile(a,95)),
                            "max":float(np.max(a)),"n":len(a)} for bn,a in vv.items()}
    log("null",name,"N1 p95(B)=%.3f N2 p95(B)=%.3f"%(
        NUL["N1_identity_permute"][name]["B_honest_full_blend_OOT"]["p95"],
        NUL["N2_resid_permute_within_season"][name]["B_honest_full_blend_OOT"]["p95"]))
json.dump(NUL,open(f"{OUT}/sC.json","w"),ensure_ascii=False,indent=1)
print("STAGE_C_OK")

# ─────────── stage D : 타자 반분할 교차적합 (20 x 2) ───────────
CF={}
for name in AXES:
    w=DIR[name]
    for bn,(b,res) in BASES.items():
        full,half=[],[]
        for rep in range(20):
            rg=np.random.default_rng(5000+rep)
            side=rg.integers(0,2,NB)[bidx]
            for a in (0,1):
                mA, mB = side==a, side!=a
                d=c_of[bn]*w; d=d-d.mean()
                LA=float(np.mean(d[mA]*res[mA])); GA=float(np.mean(d[mA]*d[mA]))
                if GA<=0: continue
                u=LA/GA
                LB_=float(np.mean(d[mB]*res[mB])); GB=float(np.mean(d[mB]*d[mB]))
                full.append(K_PTS*(2*LB_*u-GB*u*u))
                half.append(K_PTS*(2*LB_*(0.5*u)-GB*(0.5*u)**2))
        CF.setdefault(name,{})[bn]={
          "oos_full_step":{"mean":float(np.mean(full)),"sd":float(np.std(full,ddof=1)),
                           "frac_pos":float(np.mean(np.array(full)>0)),"worst":float(np.min(full)),"n":len(full)},
          "oos_half_step":{"mean":float(np.mean(half)),"sd":float(np.std(half,ddof=1)),
                           "frac_pos":float(np.mean(np.array(half)>0)),"worst":float(np.min(half))}}
    log("crossfit",name,"B full=%.3f half=%.3f"%(
        CF[name]["B_honest_full_blend_OOT"]["oos_full_step"]["mean"],
        CF[name]["B_honest_full_blend_OOT"]["oos_half_step"]["mean"]))
json.dump(CF,open(f"{OUT}/sD.json","w"),ensure_ascii=False,indent=1)
print("STAGE_D_OK")
