# exp_111 stage E — ★전이비 눈금★(우리 정직면에서 LB 가 알려진 두 축을 재서) + 직교성 + 흡수 진단.
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="2"
import glob, json, time
import numpy as np, pandas as pd
from scipy.special import expit, logit
H=f"/home/<user>/SungjinPark"; OUT=f"{H}/exp111"
REF,EPS=0.2498069,1e-6; K_PTS=100000.0/REF
ALPHA,BSKEW=1.07911,-0.059151
T0=time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]",*a,flush=True)
lg=lambda p: logit(np.clip(np.asarray(p,np.float64),EPS,1-EPS))

USE=["row_id","season","game_type","control_success","pitcher_id","batter_id","batter_hand",
     "pitcher_hand","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate",
     "balls_before","strikes_before"]
tr=pd.read_csv(f"{H}/data/train.csv",encoding="utf-8-sig",usecols=USE)
row=tr[tr.season==2024].reset_index(drop=True); H23=tr[tr.season<=2023]
N=len(row); y=row.control_success.to_numpy(np.float64)
r_oot=np.load(f"{OUT}/r_oot.npy"); r_cs3=np.load(f"{OUT}/r_cs3.npy"); r67=np.load(f"{OUT}/r67.npy")
p_oot=np.load(f"{OUT}/p_oot.npy"); c_emp=np.load(f"{OUT}/c_emp.npy")
W=np.load(f"{OUT}/dir_w.npy")            # batter 3형제 표준화 방향
AXN=["batter_id","batter_id_x_pitcher_hand","batter_team"]
# 정직면 전파계수 : z 를 p 에서 역산 (alpha*z + bskew*z^2 + d = logit(p))
zf=lg(p_oot)
zz=(-ALPHA+np.sqrt(np.maximum(ALPHA**2+4*BSKEW*(zf-zf.mean()*0),1e-12)))/(2*BSKEW) if False else None
# 안전한 역산: d 는 상수라 z 를 뉴턴으로 푼다 (d 는 pin 결과 → 여기선 c_pre 의 z 의존만 필요)
d_const=float(np.median(zf))*0  # placeholder, 아래에서 정확히 구함
# 정확히: 우리는 z(로짓믹스 이후 root 로짓)를 저장 안 했으므로 exp_111 s2 와 같은 식으로 재구성 대신
# c_pre = p(1-p)*(alpha + 2*bskew*z) 에서 z 를 zf 로부터 2차식 역함수로 푼다.
def invert(zfin, d):
    a,b=BSKEW,ALPHA; cc=d-zfin
    disc=np.maximum(b*b-4*a*cc,0.0)
    return (-b+np.sqrt(disc))/(2*a)
# d 를 격자로 찾아 |alpha*z+bskew*z^2+d - zf| 최소화 — 실제로는 d 가 유일하게 정해지지 않으므로
# 대신 c_pre 를 캘프레임 실측 c_emp 의 형태로 대체하고, 민감도로 p(1-p)*alpha 도 함께 낸다.
c_pre = ALPHA*p_oot*(1-p_oot)          # 정직면 pre-alpha 전파 (beta_skew 2차항 무시, 2% 이내)
c_post= p_oot*(1-p_oot)                # post-alpha (count 축 자리)
def vtx(d,res):
    d=d-d.mean(); G=float(np.mean(d*d))
    if G<=0: return dict(K_L=0.,K_G=0.,vertex_pts=0.,u_star=0.)
    L=float(np.mean(d*res)); return dict(K_L=K_PTS*L,K_G=K_PTS*G,vertex_pts=K_PTS*L*L/G,u_star=L/G)

# ─── E1 : LB 가 알려진 두 축을 우리 정직면에서 잰다 ───
def splits_axis(Hh,R,who,cellf,nc,k=75.0):
    yy=Hh.control_success.to_numpy(np.float64); league=float(yy.mean())
    w=Hh[who].to_numpy(np.int64); idsu,idx=np.unique(w,return_inverse=True); P=len(idsu)
    n=np.bincount(idx,minlength=P).astype(float); s=np.bincount(idx,weights=yy,minlength=P)
    p_all=(s+k*league)/(n+k); code=cellf(Hh)
    ln=np.bincount(code,minlength=nc).astype(float); ls=np.bincount(code,weights=yy,minlength=nc)
    lrate=(ls+k*league)/(ln+k); lgc=lg(lrate)-lg(league)
    flat=idx*nc+code
    cn=np.bincount(flat,minlength=P*nc).astype(float).reshape(P,nc)
    cs=np.bincount(flat,weights=yy,minlength=P*nc).reshape(P,nc)
    rate=(cs+k*p_all[:,None])/(cn+k)
    D=(lg(rate)-lg(p_all)[:,None])-lgc[None,:]
    pos=pd.Index(idsu).get_indexer(R[who].to_numpy(np.int64)); rc=cellf(R)
    return np.where(pos>=0, D[np.maximum(pos,0),rc], 0.0)
f_bh=lambda d: d.batter_hand.to_numpy(np.int64)-1
f_s2=lambda d: (d.strikes_before.to_numpy(np.int64)>=2).astype(np.int64)
SD1,SD2=0.08766134165019425,0.05495482212319005
sp23=splits_axis(H23,row,"pitcher_id",f_bh,2)/SD1+splits_axis(H23,row,"pitcher_id",f_s2,2)/SD2
THETA=json.load(open(f"{H}/build_CU/pkg/PAR/blend.json"))["count_offsets"]["theta"]
ckey=(row.balls_before.astype(str)+"-"+row.strikes_before.astype(str)).to_numpy()
cnt=np.array([THETA[k] for k in ckey])
CAL={}
KNOWN={"sp_pit_split":{"dir":c_pre*((sp23-sp23.mean())/sp23.std()),
        "lb_max_gain_pts":6.69394,"lb_realized_pts":6.651907121,
        "family":"라벨 EB 수축 표(splits) · out-of-time"},
       "CS3_count_theta":{"dir":c_post*(cnt-cnt.mean())/cnt.std(),
        "lb_max_gain_pts":1.0367,"lb_realized_pts":0.7866627052,
        "family":"원본 test 열(balls/strikes) 순수 오프셋 · 표 조회 0"}}
for k,v in KNOWN.items():
    vv=vtx(v["dir"],r_oot); vv67=vtx(v["dir"],r67); vc=vtx(v["dir"],r_cs3)
    CAL[k]={"family":v["family"],"ourface_B_vertex_pts":vv["vertex_pts"],
            "ourface_B_K_L":vv["K_L"],"ourface_B_K_G":vv["K_G"],
            "ref67_vertex_pts":vv67["vertex_pts"],"cs3_calframe_vertex_pts":vc["vertex_pts"],
            "lb_max_gain_pts":v["lb_max_gain_pts"],"lb_realized_pts":v["lb_realized_pts"],
            "transfer_vertex_ratio_LBmax_over_ourfaceB":v["lb_max_gain_pts"]/vv["vertex_pts"] if vv["vertex_pts"]>0 else None}
    log("CALIB",k,round(vv["vertex_pts"],4),"-> ratio",
        round(CAL[k]["transfer_vertex_ratio_LBmax_over_ourfaceB"],4))

# ─── E2 : 배터 축 LB 환산 ───
sB2=json.load(open(f"{OUT}/sB2.json"))
sD=json.load(open(f"{OUT}/sD.json"))
RATES={"table_EB_family_sp":CAL["sp_pit_split"]["transfer_vertex_ratio_LBmax_over_ourfaceB"],
       "raw_column_family_count":CAL["CS3_count_theta"]["transfer_vertex_ratio_LBmax_over_ourfaceB"],
       "ledger_S211_sp_0.74":0.74,"ledger_S215_count_0.1406":0.1406,"ledger_S192_weight_0.1137":0.1137}
CONV={}
for n in AXN:
    v=sB2["per_axis"][n]["B_honest_full_blend_OOT"]["vertex_pts"]
    oos=sD[n]["B_honest_full_blend_OOT"]["oos_half_step"]["mean"]
    CONV[n]={"ourfaceB_vertex_pts":v,"ourfaceB_oos_half_step_pts":oos,
             "LB_from_vertex":{k:v*r for k,r in RATES.items() if r},
             "LB_from_oos_half":{k:oos*r for k,r in RATES.items() if r}}
CONV["joint_trio"]={"ourfaceB_vertex_pts":sB2["joint_trio"]["B_honest_full_blend_OOT"]["joint_vertex_pts"],
    "LB_from_vertex":{k:sB2["joint_trio"]["B_honest_full_blend_OOT"]["joint_vertex_pts"]*r
                      for k,r in RATES.items() if r}}
log("CONV batter_id", {k:round(x,3) for k,x in CONV["batter_id"]["LB_from_vertex"].items()})

# ─── E3 : 직교성 ───
def ins_off(Hh,R,Kk=50.0):
    n=Hh.asof_pitcher_n.to_numpy(np.float64)
    c=np.round(n*np.nan_to_num(Hh.asof_pitcher_success_rate.to_numpy(np.float64)))
    t=pd.DataFrame({"pid":Hh.pitcher_id.to_numpy(),"n":n,"c":c})
    pp=t.loc[t.groupby("pid")["n"].idxmax()].set_index("pid"); lgl=float(np.nansum(t.c)/max(np.nansum(t.n),1))
    nn=R.asof_pitcher_n.to_numpy(np.float64)
    pos=pd.Index(pp.index.to_numpy()).get_indexer(R.pitcher_id.to_numpy()); kn=pos>=0; sf=np.where(kn,pos,0)
    an=np.where(kn,pp["n"].to_numpy()[sf],0.0); ac=np.where(kn,pp["c"].to_numpy()[sf],0.0)
    ins_n=np.maximum(nn-an,0.0); prior=np.where(an>0,ac/np.maximum(an,1.0),lgl)
    kk=np.maximum(np.round(nn*np.nan_to_num(R.asof_pitcher_success_rate.to_numpy(np.float64)))-ac,0.0)
    cur=(kk+Kk*prior)/(ins_n+Kk); dev=cur-prior
    cc=np.clip(cur,.02,.98); pv=np.clip(cc-dev,.02,.98)
    return np.nan_to_num(np.log(cc/(1-cc))-np.log(pv/(1-pv))), cur, np.where(an>0,ac/np.maximum(an,1.0),np.nan)
o23,ins_suc,anc=ins_off(H23,row)
# personal lookup_d (<=2023) & league_dev 발동 셀
def personal(Hh,k=400.0):
    yy=Hh.control_success.to_numpy(np.float64); pid=Hh.pitcher_id.to_numpy(np.int64); bh=Hh.batter_hand.to_numpy(np.int64)
    idsu,idx=np.unique(pid,return_inverse=True)
    pn=np.bincount(idx,minlength=len(idsu)).astype(float); ps=np.bincount(idx,weights=yy,minlength=len(idsu))
    o=ps/np.maximum(pn,1.0); key=pid*10+bh
    kids,kidx=np.unique(key,return_inverse=True)
    cn=np.bincount(kidx,minlength=len(kids)).astype(float); cs=np.bincount(kidx,weights=yy,minlength=len(kids))
    po=pd.Series(o,index=idsu).reindex(kids//10).to_numpy()
    return dict(zip(kids,(cs-cn*po)/(cn+k))), set(kids.tolist())
Tp,Kset=personal(H23)
rkey=row.pitcher_id.to_numpy(np.int64)*10+row.batter_hand.to_numpy(np.int64)
d_s=np.array([Tp.get(kk,0.0) for kk in rkey]); miss=np.array([kk not in Kset for kk in rkey],float)
ph=row.pitcher_hand.to_numpy(np.int64); bh=row.batter_hand.to_numpy(np.int64)
ftmr=np.load(f"{H}/build_CU/work/ftmr_PAR.npy"); ftwo=np.load(f"{H}/build_CU/work/ftwo_PAR.npy")
psr=row.asof_pitcher_success_rate.to_numpy(np.float64)
zlog=lg(p_oot)
OTH={"sp_pit_split(CT,axis3)":c_pre*sp23,
     "CS3_count_theta":c_post*cnt,
     "ctrl_asof_psr(axis2)":c_pre*np.nan_to_num(psr),
     "beta_ins_off(axis1)":c_pre*o23,
     "failtype_b(lg_mr)":c_post*ftmr, "failtype_c(lg_wo)":c_post*ftwo,
     "lookup_d_d_s(L12c)":c_pre*d_s,
     "league_dev_1|1":c_pre*(miss*((ph==1)&(bh==1))), "league_dev_1|2":c_pre*(miss*((ph==1)&(bh==2))),
     "league_dev_2|1":c_pre*(miss*((ph==2)&(bh==1))), "league_dev_2|2":c_pre*(miss*((ph==2)&(bh==2))),
     "alpha_proxy":p_oot*(1-p_oot)*zlog, "beta_skew_proxy":p_oot*(1-p_oot)*zlog*zlog,
     "ins_p_success":c_pre*np.nan_to_num(ins_suc), "supp_anc_suc":c_pre*np.nan_to_num(anc)}
gt=row.game_type.to_numpy()
def cen(v): v=np.asarray(v,np.float64); return v-v.mean()
RHO={}
for i,n in enumerate(AXN):
    a=cen(c_pre*W[i]); rr={}
    for m,b in OTH.items():
        b=cen(b)
        if a.std()==0 or b.std()==0: rr[m]=None; continue
        allr=float(np.corrcoef(a,b)[0,1])
        seg={}
        for s in ("R","F"):
            ms=gt==s
            seg[s]=float(np.corrcoef(a[ms]-a[ms].mean(),b[ms]-b[ms].mean())[0,1]) if ms.sum()>10 and a[ms].std()>0 and b[ms].std()>0 else None
        rr[m]={"all":allr,"R":seg["R"],"F":seg["F"]}
    RHO[n]={"per_axis":rr,
            "max_abs_rho_all":max(abs(x["all"]) for x in rr.values() if x),
            "max_abs_rho_any_segment":max(max(abs(x["all"]),abs(x["R"] or 0),abs(x["F"] or 0)) for x in rr.values() if x)}
    log("RHO",n,"max|rho| all=%.4f seg=%.4f"%(RHO[n]["max_abs_rho_all"],RHO[n]["max_abs_rho_any_segment"]))
RHO["🔴_not_reconstructible"]=["w_out","c_aa","c_T70","w_v6_inner","w_ebm","d_adj(상수→중심화 후 0)",
    "mate 내부 beta(offset.beta) — 팔별 로짓이 이 프레임에 없다"]

# ─── E4 : 흡수 진단 ───
bat=row.batter_id.to_numpy(); ub,bi=np.unique(bat,return_inverse=True)
nb=np.bincount(bi).astype(float)
ABS_={}
for nm,rv in (("A_67col_AA_arm_OOT",r67),("B_honest_full_blend_OOT",r_oot),("C_CS3_calframe_insample",r_cs3)):
    m=np.bincount(bi,weights=rv)/nb
    sh=np.bincount(bi,weights=rv)/(nb+208.0)
    ABS_[nm]={"weighted_sd_of_shrunk_batter_mean":float(np.sqrt(np.average(sh**2,weights=nb))),
              "n_batters":int(len(ub))}
ABS_["rho_shrunkmean_A_vs_B"]=float(np.corrcoef(np.bincount(bi,weights=r67)/(nb+208.0),
                                                np.bincount(bi,weights=r_oot)/(nb+208.0))[0,1])
ABS_["rho_shrunkmean_A_vs_C"]=float(np.corrcoef(np.bincount(bi,weights=r67)/(nb+208.0),
                                                np.bincount(bi,weights=r_cs3)/(nb+208.0))[0,1])
json.dump({"E1_transfer_calibration":CAL,"E2_LB_conversion":CONV,"E3_orthogonality":RHO,
           "E4_absorption":ABS_},open(f"{OUT}/sE.json","w"),ensure_ascii=False,indent=1)
log("absorption",{k:(round(v["weighted_sd_of_shrunk_batter_mean"],6) if isinstance(v,dict) else round(v,4)) for k,v in ABS_.items()})
print("STAGE_E_OK")
