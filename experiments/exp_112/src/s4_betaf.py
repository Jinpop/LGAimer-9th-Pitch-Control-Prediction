import sys,os,json,numpy as np
P="<SCRATCH>/cw/pkg"
sys.path.insert(0,P)
import form_decomp as fd
O='out/'
K=100000.0/0.2498069
p=np.load(O+'offline_CW.npy'); y=np.load(O+'labels.npy')
C=np.load(O+'exp112_calcols.npz',allow_pickle=True)
pmap_ids=np.load(os.path.join(P,'model_form','pmap_pitcher_ids.npy'))
n_end=np.load(os.path.join(P,'model_form','n_end_2024ext.npy'))
s_end=np.load(os.path.join(P,'model_form','s_end_2024ext.npy'))
unseen=len(pmap_ids); pmap={int(v):i for i,v in enumerate(pmap_ids)}
pid=C['pitcher_id']
pcode=np.array([pmap.get(int(v),unseen) for v in pid],dtype=np.int64)
nn=C['asof_pitcher_n'].astype(np.float64); rate=C['asof_pitcher_success_rate'].astype(np.float64)
out=fd.apply_decomposition(nn,rate,pcode,n_end,s_end,100)
d=out['d_form_eb']; hf=out['has_form']
d_eff=np.where(np.isfinite(d),d,0.0)
s=p*(1-p); r=p-y
res={'K_frozen_in_form_offset_py':100,'d_form_eb_k_in_blend_json':100,
 'n_has_form':int(hf.sum()),'n':int(len(p)),
 'n_unseen_pitcher':int((pcode==unseen).sum()),
 'd_eff_stats':dict(min=float(d_eff.min()),max=float(d_eff.max()),
                    sd=float(d_eff.std()),mean=float(d_eff.mean()),
                    n_nonzero=int(np.count_nonzero(d_eff)))}
# beta_f enters in PROBABILITY space: preds = clip(p_blend + beta_f*d_eff)
for tag,dd in (('unpinned',d_eff),
               ('pinned_d_adj_resolved',d_eff-(s*( (d_eff/np.where(s==0,1,s)) )).sum()*0)):
    pass
d_un=d_eff.copy()
d_pn=d_eff - s*(d_eff.sum()/s.sum())     # d_adj re-solve removes the mean shift
for tag,dd in (('unpinned',d_un),('pinned',d_pn)):
    KG=K*float((dd*dd).mean()); KL=-K*float((dd*r).mean())
    res[tag]=dict(K_G=KG,K_L=KL,u_star=KL/KG,max_gain_calframe_pts=KL*KL/KG)
json.dump(res,open(O+'s4_betaf.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
