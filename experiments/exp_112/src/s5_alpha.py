import json,numpy as np,pandas as pd
from scipy.special import logit
O='out/'; K=100000.0/0.2498069
PK='<SCRATCH>/cw/pkg/blend.json'
bl=json.load(open(PK,encoding='utf-8'))
p=np.load(O+'offline_CW.npy'); y=np.load(O+'labels.npy'); C=np.load(O+'exp112_calcols.npz',allow_pickle=True)
lg_mr=np.load(O+'ftmr_CW_0.npy')
alpha=bl['alpha']; bs=bl['beta_skew']; d_adj=bl['d_adj']
ftb=bl['failtype_offset']['b']; mu_mr=bl['failtype_offset']['mu_mr']
# count + bat shifts
th=bl['count_offsets']['theta']; tab=np.zeros((4,3))
for i in range(4):
    for j in range(3): tab[i,j]=th[f'{i}-{j}']
bb=C['balls_before'].astype(float); ss=C['strikes_before'].astype(float)
ok=np.isfinite(bb)&np.isfinite(ss)&(bb==np.floor(bb))&(ss==np.floor(ss))&(bb>=0)&(bb<=3)&(ss>=0)&(ss<=2)
cnt=np.zeros(len(p)); cnt[ok]=tab[bb[ok].astype(int),ss[ok].astype(int)]
bt=bl['bat_offsets']; bk=np.array([int(x) for x in bt['theta']],dtype=np.int64)
bv=np.array([float(bt['theta'][x]) for x in bt['theta']])
bid=pd.to_numeric(pd.Series(C['batter_id']),errors='coerce').to_numpy(float)
bok=np.isfinite(bid)&(bid==np.floor(bid))
pos=pd.Index(bk).get_indexer(np.where(bok,bid,-1).astype(np.int64))
bat=np.where(pos>=0,bv[np.maximum(pos,0)],bt['theta_miss']); bat=np.where(bok,bat,0.0)
Q=logit(p)-d_adj-ftb*(lg_mr-mu_mr)-cnt-bat          # aux_shift 미포함 (nact 미보유)
disc=alpha*alpha+4*bs*Q
m=(-alpha+np.sqrt(disc))/(2*bs)
chk=alpha*m+bs*m*m-Q
res={'m_recovery_max_resid':float(np.abs(chk).max()),
     'm_stats':dict(min=float(m.min()),max=float(m.max()),sd=float(m.std()),mean=float(m.mean())),
     '🔴_aux_shift_omitted':'aux_offsets[0] v6_nn_hidden_nact 의 raw(nact)를 이 방에서 보유하지 않아 0 으로 뒀다. u=3.5931e-04, nact in [0,64] 이므로 |aux_shift| <= 0.01494 → m 오차 <= 0.0139 (m 의 sd 대비)'}
s=p*(1-p); r=p-y
def pin(v): return s*(v-(s*v).sum()/s.sum())
def rep(v,label,ship=None):
    d=pin(v); KG=K*float((d*d).mean()); KL=-K*float((d*r).mean())
    o=dict(label=label,K_G_calframe=KG,K_L_calframe_UNTRUSTED=KL,
           K_L_needed_for_1p4_pts=float(np.sqrt(1.4*KG)),
           K_L_needed_for_2p5_pts=float(np.sqrt(2.5*KG)))
    if ship is not None:
        o['u_needed_for_1p4_pts']=float(np.sqrt(1.4/KG))
        o['shipped_value']=ship
    return o
res['alpha']=rep(m,'alpha  (dz/dalpha = logit_mix)',alpha)
res['beta_skew']=rep(m*m,'beta_skew  (dz/dbeta = logit_mix^2)',bs)
# axis ctrl_asof_psr: pre-alpha, chain = alpha + 2*beta_skew*m
ax=bl['axis_offsets'][1]
raw=C['asof_pitcher_success_rate'].astype(float)
dl=np.zeros(len(p)); fin=np.isfinite(raw); dl[fin]=(raw[fin]-ax['center'])/ax['scale']
res['axis_ctrl_asof_psr']=rep((alpha+2*bs*m)*dl,'axis_offsets[1] ctrl_asof_psr (pre-alpha, 연쇄율 적용)',ax['u'])
res['axis_ctrl_asof_psr']['n_nonfinite_raw']=int((~fin).sum())
res['axis_ctrl_asof_psr']['package_recorded_G_ii']=ax['G_ii']
res['axis_ctrl_asof_psr']['package_recorded_single_axis_vertex_u']=ax['single_axis_vertex_u']
res['axis_ctrl_asof_psr']['package_recorded_single_axis_vertex_gain_pts']=ax['single_axis_vertex_gain_pts']
res['axis_ctrl_asof_psr']['shipped_pct_of_recorded_single_axis_vertex']=100*ax['u']/ax['single_axis_vertex_u']
a0=bl['axis_offsets'][0]
res['axis_beta_ins_off_recorded_only']={k:a0[k] for k in ('u','probe_u','probe_gain_pts','G_ii','single_axis_vertex_u','single_axis_vertex_gain_pts')}
res['axis_beta_ins_off_recorded_only']['shipped_pct_of_recorded_single_axis_vertex']=100*a0['u']/a0['single_axis_vertex_u']
json.dump(res,open(O+'s5_alpha.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
