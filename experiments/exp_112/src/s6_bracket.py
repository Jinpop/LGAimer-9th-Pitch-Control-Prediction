import json,numpy as np,pandas as pd
from scipy.special import logit
O='out/'; K=100000.0/0.2498069
PK='<SCRATCH>/cw/pkg/blend.json'
bl=json.load(open(PK,encoding='utf-8'))
p=np.load(O+'offline_CW.npy'); y=np.load(O+'labels.npy'); C=np.load(O+'exp112_calcols.npz',allow_pickle=True)
lg_mr=np.load(O+'ftmr_CW_0.npy'); lg_wo=np.load(O+'ftwo_CW_0.npy')
mu_mr,mu_wo=bl['failtype_offset']['mu_mr'],bl['failtype_offset']['mu_wo']
s=p*(1-p); r=p-y; n=len(p)
def pin(v): return s*(v-(s*v).sum()/s.sum())
def kl_se(d):
    x=-K*d*r
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(n))
th=bl['count_offsets']['theta']; tab=np.zeros((4,3))
for i in range(4):
    for j in range(3): tab[i,j]=th[f'{i}-{j}']
bb=C['balls_before'].astype(float); ss=C['strikes_before'].astype(float)
ok=(bb==np.floor(bb))&(ss==np.floor(ss))&(bb>=0)&(bb<=3)&(ss>=0)&(ss<=2)
vc=np.zeros(n); vc[ok]=tab[bb[ok].astype(int),ss[ok].astype(int)]
axes={'failtype_b':pin(lg_mr-mu_mr),'failtype_c':pin(lg_wo-mu_wo),'count_CS3':pin(vc)}
res={'note':'K·L_cal 의 표본 표준오차 — 부호가 잡음인지 아닌지 가른다. 🔴 편향(표본 안 암기)은 이걸로 안 잡힌다'}
for k,d in axes.items():
    m_,se=kl_se(d); res[k]=dict(K_L_cal=m_,se=se,t=m_/se,K_G=K*float((d*d).mean()))
# --- failtype c 의 최대이득 브래킷 ---
KGb,KGc=res['failtype_b']['K_G'],res['failtype_c']['K_G']
db,dc=axes['failtype_b'],axes['failtype_c']; KGbc=K*float((db*dc).mean())
b_ship=-0.03645076320272027; LBb=2.6766924977
KLb_LB=(LBb+KGb*b_ship**2)/(2*b_ship)
KLc_cal=res['failtype_c']['K_L_cal']
br={}
for name,ratio in [('앵커_b_비율_0.1809',KLb_LB/res['failtype_b']['K_L_cal']),
                   ('절반_0.0904',0.5*KLb_LB/res['failtype_b']['K_L_cal']),
                   ('두배_0.3618',2.0*KLb_LB/res['failtype_b']['K_L_cal']),
                   ('L_c=0',0.0)]:
    KLc=KLc_cal*ratio
    cond=KLc-KGbc*b_ship
    br[name]=dict(ratio=ratio,K_L_c=KLc,
        alone_max_pts=KLc*KLc/KGc,
        u_star_alone=KLc/KGc,
        cond_on_shipped_b_max_pts=cond*cond/KGc,
        u_star_cond=cond/KGc)
res['failtype_c_bracket']=br
res['K_G_bc']=KGbc
res['🔴_why_a_bracket']='캘프레임 L 의 전이비가 앵커에 따라 +0.181(b) 과 −1.512(count) 로 갈렸다. 하나의 값을 주장할 수 없다'
json.dump(res,open(O+'s6_bracket.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
