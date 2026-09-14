import json, numpy as np
O='out/'
K = 100000.0/0.2498069
p  = np.load(O+'offline_CW.npy')          # shipped CW on the calibration frame
y  = np.load(O+'labels.npy')
lg_mr = np.load(O+'ftmr_CW_0.npy')
lg_wo = np.load(O+'ftwo_CW_0.npy')
mu_mr = -0.4543162230525402
mu_wo = -2.009182627386164
s = p*(1.0-p)                              # dp/dz
r = p - y

def axis(v, name):
    """v = dz/dtheta.  d_adj re-solve pins mean(p) -> project out the s-weighted mean."""
    raw = s*v
    vbar = raw.sum()/s.sum()               # weighted mean of v with weight s
    d_un = raw                             # unpinned
    d_pn = s*(v - vbar)                    # pinned (d_adj re-solved)
    out={}
    for tag,d in (('unpinned',d_un),('pinned',d_pn)):
        KG = K*float((d*d).mean())
        KL = -K*float((d*r).mean())
        out[tag]=dict(K_G=KG, K_L=KL, u_star=KL/KG, max_gain_pts=KL*KL/KG)
    out['name']=name
    out['mean_delta_unpinned']=float(d_un.mean())
    return out

res={}
res['K_pts_per_brier']=K
res['n']=int(len(p))
res['frame']="build_AC/work/tr2024 (train season==2024, 253,507행) — 우리 규약 캘프레임 (ins 앵커 lt2024 스왑 · lookup <=2023 키 1,202 제한).  벡터는 build_CW/work 의 출하 빌드 산출물"
res['ft_c'] = axis(lg_wo-mu_wo, 'failtype_offset.c  (way-off 로짓)')
res['ft_b'] = axis(lg_mr-mu_mr, 'failtype_offset.b  (mid/rev 로짓)  ★양성 대조 — LB 로 눈금이 잡힌 축★')
# shipped b step
b_ship = -0.03645076320272027
for tag in ('unpinned','pinned'):
    g = res['ft_b'][tag]
    g['gain_at_shipped_b_pts'] = 2*g['K_L']*b_ship - g['K_G']*b_ship*b_ship
    g['shipped_b_pct_of_vertex'] = 100.0*b_ship/g['u_star']
res['rho_b_c'] = float(np.corrcoef(s*(lg_mr-mu_mr), s*(lg_wo-mu_wo))[0,1])
# pinned correlation (what matters for joint loading)
db = s*((lg_mr-mu_mr) - (s*(lg_mr-mu_mr)).sum()/s.sum())
dc = s*((lg_wo-mu_wo) - (s*(lg_wo-mu_wo)).sum()/s.sum())
res['rho_b_c_pinned'] = float(np.corrcoef(db,dc)[0,1])
res['brier_shipped'] = float(((p-y)**2).mean())
res['score_shipped_cal_frame'] = 100000.0*(1.0-((p-y)**2).mean()/0.2498069)
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(O+'s1_failtype.json','w'), ensure_ascii=False, indent=1)
