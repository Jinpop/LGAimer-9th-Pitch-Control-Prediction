"""exp_096 — fold B 부록: 척도 몫을 뗀 대조 (S25·S33).

fold B 의 팔들은 z_sd 0.355 로 크게 과분산이라 고정 척도 점수가 −1800 대로 무너진다.
그 상태의 −25.36 은 척도 몫이 지배한다.  alpha 를 자유롭게 두고 다시 재야 정보 몫이 보인다.
"""
import json, os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="2"
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize_scalar
H="/home/<user>/SungjinPark"; B=f"{H}/exp096"; W=f"{B}/workb"
S5=[42,777,2024,101,7]; REF=0.2498069; K=100000.0/REF
y=np.load(f"{W}/y.npy")[np.load(f"{W}/vam.npy")].astype(np.float64)
pit=np.load(f"{W}/pit_valid.npy",allow_pickle=True); YB=float(y.mean())
def pin(z):
    c=0.0
    for _ in range(200):
        p=expit(z+c); g=float(p.mean())-YB
        if abs(g)<1e-15: break
        c-=g/float((p*(1-p)).mean())
    return c
sc=lambda z: 100000.0*(1.0-float(np.mean((expit(z+pin(z))-y)**2))/REF)
def afree(z):
    r=minimize_scalar(lambda a:-sc(a*z),bounds=(0.02,2.0),method="bounded",
                      options={"xatol":1e-8})
    return float(-r.fun), float(r.x)
def auc(s):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n0*n1))
rng=np.random.default_rng(20260824)
uq,inv=np.unique(pit,return_inverse=True); nb=len(uq)
order=np.argsort(inv,kind="stable")
st=np.searchsorted(inv[order],np.arange(nb)); en=np.append(st[1:],len(order))
BS=(en-st).astype(np.float64)
PK=np.stack([rng.integers(0,nb,nb) for _ in range(1000)])
IDX=[order[st[j]:en[j]] for j in range(nb)]
def boot(d):
    b_=np.array([d[IDX[j]].sum() for j in range(nb)])
    bs=b_[PK].sum(1)/BS[PK].sum(1); se=float(bs.std(ddof=1)); ob=float(d.mean())
    return {"gain_pts":-ob*K,"t":float(-ob/se) if se>0 else None,
            "p_share_rank_worse":float((bs>0).mean())}
Z={}
for tag in ("std","rank"):
    m=np.mean([np.load(f"{B}/out/predsb/z_{tag}_le22_s{s}.npy") for s in S5],0)
    g,a=afree(m)
    Z[tag]={"z_sd":float(m.std()),"auc":auc(m),"fixed_scale_score":sc(m),
            "alpha_free_score":g,"alpha_star":a,"_m":m}
za=Z["rank"]["alpha_star"]*Z["rank"]["_m"]; zs=Z["std"]["alpha_star"]*Z["std"]["_m"]
Pa=expit(za+pin(za)); Ps=expit(zs+pin(zs))
R={"exp_id":"exp_096","stage":"foldb_addendum",
   "🔴_why":("fold B 팔은 z_sd 0.355 로 크게 과분산이라 고정 척도 점수가 무너진다. "
             "S25·S33 규율대로 alpha 자유 값을 함께 봐야 정보 몫이 보인다."),
   "std":{k:v for k,v in Z["std"].items() if k!="_m"},
   "rank":{k:v for k,v in Z["rank"].items() if k!="_m"},
   "delta_alpha_free_rank_minus_std":
       Z["rank"]["alpha_free_score"]-Z["std"]["alpha_free_score"],
   "paired_boot_alpha_free":boot((Pa-y)**2-(Ps-y)**2),
   "reference_foldA":{"deep(std) alone_seedmean":862.5652996988365,
                      "rank alone_seedmean":887.5233255888793,
                      "delta":+24.958025890042804,
                      "deep auc on 2024":0.55276},
   "🔴_reading":("fold A 와 fold B 의 부호가 갈린다. 단 fold B 는 신호 자체가 훨씬 약한 "
                 "환경이고(AUC 0.531 vs 0.553) 두 팔 모두 alpha* 가 0.2 아래라 "
                 "정직면 관례가 그대로 안 통한다. 유의하지도 않다.")}
json.dump(R,open(f"{B}/out/foldb_addendum.json","w"),ensure_ascii=False,indent=1)
print(json.dumps(R,ensure_ascii=False,indent=1))
