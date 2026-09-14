import json, numpy as np
O='out/'
K=100000.0/0.2498069
p=np.load(O+'offline_CW.npy'); y=np.load(O+'labels.npy')
C=np.load(O+'exp112_calcols.npz',allow_pickle=True)
bl=json.load(open('<SCRATCH>/cw/pkg/blend.json',encoding='utf-8'))
s=p*(1-p); r=p-y
def pin(v): return s*(v-(s*v).sum()/s.sum())
def solve(v,label,lb_gain=None,u_lb=1.0):
    d=pin(v); KG=K*float((d*d).mean()); KL=-K*float((d*r).mean())
    o=dict(label=label,K_G_calframe=KG,K_L_calframe=KL,
           u_star_calframe=KL/KG,max_gain_calframe_pts=KL*KL/KG)
    if lb_gain is not None:
        KLlb=(lb_gain+KG*u_lb*u_lb)/(2*u_lb)
        o.update(LB_gain_observed=lb_gain,u_of_LB_point=u_lb,
                 K_L_implied_by_LB=KLlb,
                 transfer_ratio_L_cal_to_LB=KLlb/KL,
                 calframe_gain_at_that_step=2*KL*u_lb-KG*u_lb*u_lb,
                 calframe_overreads_gain_by=(2*KL*u_lb-KG*u_lb*u_lb)/lb_gain,
                 u_star_LB=KLlb/KG, shipped_pct_of_vertex=100*u_lb/(KLlb/KG),
                 max_gain_LB_pts=KLlb*KLlb/KG,
                 residual_LB_pts=KLlb*KLlb/KG-lb_gain)
    return o
res={}
# --- 앵커 2: CS3 카운트 축 (현 기반 LB 점 +0.7866627052, 부모 CT) ---
th=bl['count_offsets']['theta']
tab=np.zeros((4,3))
for b_ in range(4):
    for s_ in range(3): tab[b_,s_]=th[f'{b_}-{s_}']
bb=C['balls_before'].astype(float); ss=C['strikes_before'].astype(float)
ok=np.isfinite(bb)&np.isfinite(ss)&(bb==np.floor(bb))&(ss==np.floor(ss))&(bb>=0)&(bb<=3)&(ss>=0)&(ss<=2)
v_cnt=np.zeros(len(p)); v_cnt[ok]=tab[bb[ok].astype(int),ss[ok].astype(int)]
res['anchor_count_CS3']=solve(v_cnt,'count_offsets 표 전체 (u=1 = 출하 걸음)',0.7866627052,1.0)
res['anchor_count_CS3']['n_rows_in_grid']=int(ok.sum())
# --- CW 타자 축 (LB 미측정) ---
bt=bl['bat_offsets']; bk=np.array([int(x) for x in bt['theta']],dtype=np.int64)
bv=np.array([float(bt['theta'][x]) for x in bt['theta']])
import pandas as pd
bid=pd.to_numeric(pd.Series(C['batter_id']),errors='coerce').to_numpy(float)
bok=np.isfinite(bid)&(bid==np.floor(bid))
pos=pd.Index(bk).get_indexer(np.where(bok,bid,-1).astype(np.int64))
v_bat=np.where(pos>=0,bv[np.maximum(pos,0)],bt['theta_miss']); v_bat=np.where(bok,v_bat,0.0)
res['CW_bat_axis']=solve(v_bat,'bat_offsets 표 전체 (u=1 = 출하 걸음, 부호 반전판)')
res['CW_bat_axis']['n_hit']=int((pos>=0).sum()); res['CW_bat_axis']['n_miss']=int(((pos<0)&bok).sum())
print(json.dumps(res,ensure_ascii=False,indent=1))
json.dump(res,open(O+'s3_anchors.json','w'),ensure_ascii=False,indent=1)
