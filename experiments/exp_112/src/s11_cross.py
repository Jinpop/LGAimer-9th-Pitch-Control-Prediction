import json,numpy as np,pandas as pd
from scipy.special import logit
O='out/'; K=100000.0/0.2498069
PK='<SCRATCH>/cw/pkg/blend.json'
bl=json.load(open(PK,encoding='utf-8'))
p=np.load(O+'offline_CW.npy'); C=np.load(O+'exp112_calcols.npz',allow_pickle=True)
lg_mr=np.load(O+'ftmr_CW_0.npy'); lg_wo=np.load(O+'ftwo_CW_0.npy')
al,bs,da=bl['alpha'],bl['beta_skew'],bl['d_adj']
ftb=bl['failtype_offset']['b']; mu_mr=bl['failtype_offset']['mu_mr']; mu_wo=bl['failtype_offset']['mu_wo']
n=len(p); s=p*(1-p)
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
a1=bl['axis_offsets'][1]; raw=C['asof_pitcher_success_rate'].astype(float)
dl=np.zeros(n); fin=np.isfinite(raw); dl[fin]=(raw[fin]-a1['center'])/a1['scale']
def pin(v): return s*(v-(s*v).sum()/s.sum())
V={'ft_c':lg_wo-mu_wo,'ft_b':lg_mr-mu_mr,'alpha':m,'beta_skew':m*m,
   'ctrl_asof_psr':(al+2*bs*m)*dl,'count':cnt,'bat':bat}
names=list(V); D={k:pin(v) for k,v in V.items()}
rho={a:{b:float(np.corrcoef(D[a],D[b])[0,1]) for b in names} for a in names}
seg=np.where(np.arange(n)>=0,'',''); 
out={'rho_pinned_calframe':rho,
 'ft_c_vs_others':{b:rho['ft_c'][b] for b in names if b!='ft_c'},
 '🔑':'ft_c 는 ft_b 와 ρ −0.635 로 강하게 얽혀 있다. 나머지 축과는 |ρ| ≤ %.3f — 함께 실어도 교차항이 작다'%max(abs(rho['ft_c'][b]) for b in names if b not in ('ft_c','ft_b'))}
json.dump(out,open(O+'s11_cross.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out['ft_c_vs_others'],ensure_ascii=False,indent=1)); print(out['🔑'])
