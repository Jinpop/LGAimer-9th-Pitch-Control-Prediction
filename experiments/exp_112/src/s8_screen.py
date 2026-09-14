import json,numpy as np,pandas as pd
from scipy.special import logit
O='out/'; K=100000.0/0.2498069
PK='<SCRATCH>/cw/pkg/blend.json'
bl=json.load(open(PK,encoding='utf-8'))
p=np.load(O+'offline_CW.npy'); y=np.load(O+'labels.npy'); C=np.load(O+'exp112_calcols.npz',allow_pickle=True)
lg_mr=np.load(O+'ftmr_CW_0.npy'); lg_wo=np.load(O+'ftwo_CW_0.npy')
al,bs,da=bl['alpha'],bl['beta_skew'],bl['d_adj']
ftb=bl['failtype_offset']['b']; mu_mr=bl['failtype_offset']['mu_mr']; mu_wo=bl['failtype_offset']['mu_wo']
n=len(p); s=p*(1-p); r=p-y
th=bl['count_offsets']['theta']; tab=np.zeros((4,3))
for i in range(4):
    for j in range(3): tab[i,j]=th[f'{i}-{j}']
bb=C['balls_before'].astype(float); ss=C['strikes_before'].astype(float)
ok=(bb==np.floor(bb))&(ss==np.floor(ss))&(bb>=0)&(bb<=3)&(ss>=0)&(ss<=2)
cnt=np.zeros(n); cnt[ok]=tab[bb[ok].astype(int),ss[ok].astype(int)]
bt=bl['bat_offsets']; bk=np.array([int(x) for x in bt['theta']],dtype=np.int64)
bv=np.array([float(bt['theta'][x]) for x in bt['theta']])
bid=pd.to_numeric(pd.Series(C['batter_id']),errors='coerce').to_numpy(float)
bok=np.isfinite(bid)&(bid==np.floor(bid))
pos=pd.Index(bk).get_indexer(np.where(bok,bid,-1).astype(np.int64))
bat=np.where(pos>=0,bv[np.maximum(pos,0)],bt['theta_miss']); bat=np.where(bok,bat,0.0)
Q=logit(p)-da-ftb*(lg_mr-mu_mr)-cnt-bat
m=(-al+np.sqrt(al*al+4*bs*Q))/(2*bs)
ax1=bl['axis_offsets'][1]; raw=C['asof_pitcher_success_rate'].astype(float)
dl=np.zeros(n); fin=np.isfinite(raw); dl[fin]=(raw[fin]-ax1['center'])/ax1['scale']
def pin(v): return s*(v-(s*v).sum()/s.sum())
V={'failtype_b (출하 −0.036451)':lg_mr-mu_mr,
   'failtype_c (출하 0.0)':lg_wo-mu_wo,
   'alpha (출하 1.07911)':m,
   'beta_skew (출하 −0.059151)':m*m,
   'axis ctrl_asof_psr (출하 +0.006544)':(al+2*bs*m)*dl,
   'count_offsets 표 (출하 u=1)':cnt,
   'bat_offsets 표 (출하 u=1)':bat}
out={'🔴_읽는_법':'K·L_cal 은 표본 안 암기가 실린 값이라 ★크기★ 로 못 쓴다. |t| 가 작으면 그 프레임에 신호가 없다는 뜻이고 부호도 못 읽는다(count 축이 그 사례, t −1.08 → LB 부호가 반대였다).',
     'n':n,'K':K}
for k,v in V.items():
    d=pin(v); x=-K*d*r
    out[k]=dict(K_G=K*float((d*d).mean()),K_L_cal=float(x.mean()),
                se=float(x.std(ddof=1)/np.sqrt(n)),t=float(x.mean()/(x.std(ddof=1)/np.sqrt(n))))
json.dump(out,open(O+'s8_screen.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
