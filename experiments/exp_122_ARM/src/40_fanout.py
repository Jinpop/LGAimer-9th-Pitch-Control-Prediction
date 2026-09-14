#!/usr/bin/env python3
"""exp_122 — 새 팔의 ★성향★을 8가지로 흔들어 혼합 이득을 한 번에 잰다.

exp_120 이 답한 것: 원열 47개 CatBoost(depth6) 는 rho 0.844, 최적 w 가 ★음수★.
음수 w = 팔을 빼는 게 낫다 = 새 정보가 아니라 척도/잡음 보정.

여기서 겨누는 것은 하나: ★w_opt 가 양수로 나오는 팔★.
열은 이미 소진됐다(v6 미사용 15열 중 진짜 새 정보는 outs_before 뿐).
그러므로 흔드는 것은 모형의 성향이다 — 깊이·정규화·부분공간·모형족·id 취급.

사전 등록(결과 보기 전에 고정):
  채택 문턱 = 혼합 이득 >= +9.31점 (exp_120 과 동일)
  주목 조건 = w_opt > 0 이고 혼합 이득 >= +3.0
  둘 다 미달이면 새 팔 계열 종결.
"""
import numpy as np, pandas as pd, json, os, sys, time
from catboost import CatBoostClassifier, Pool
from scipy.special import logit as _lg, expit
from scipy.optimize import minimize_scalar
H='/home/<user>/SungjinPark'; R=f'{H}/exp122'; os.makedirs(R,exist_ok=True)
tc=[c.strip() for c in open(f'{H}/data/test.csv').readline().replace('﻿','').strip().split(',')]
RAW=[c for c in tc if c!='row_id']
CATS=['top_bottom','game_type','base_state','pitcher_hand','batter_hand']
IDS=['pitcher_id','batter_id']
UNUSED=['top_bottom','outs_before','run_bot_before','score_diff_home','runner_on_1b',
        'runner_on_2b','runner_on_3b','num_runners_on','away_win_expectancy',
        'pitcher_id','batter_id','pitcher_hand','batter_hand',
        'asof_pitcher_pitchmix_n','asof_pitcher_offspeed_rate']
COUNT=['balls_before','strikes_before','inning']

def load(seasons):
    tr=pd.read_csv(f'{H}/data/train.csv'); tr.columns=[c.replace('﻿','') for c in tr.columns]
    tr=tr[tr.season.isin(seasons)].copy().reset_index(drop=True)
    return tr, tr['control_success'].to_numpy(np.float64)

t0=time.time()
tr_tr,ystr=load(list(range(2019,2024)))
tr_va,ysva=load([2024])
print(f'학습 {len(tr_tr)} · 검증 {len(tr_va)}',flush=True)

# ---- 정직면 정렬 (exp_120 과 동일 검증)
rid112=pd.Series(np.load(f'{H}/exp112_rid.npy',allow_pickle=True)).astype(str).values
po=np.load(f'{H}/exp111/p_oot.npy'); assert len(rid112)==len(po)
order=pd.Index(tr_va['row_id'].astype(str)).get_indexer(pd.Index(rid112))
assert (order>=0).all()
y=ysva[order]
assert abs(((po-y)**2).mean()-0.2474119208667189)<1e-12, 'p_oot 정렬 검증 실패'
V=y.mean()*(1-y.mean()); BR0=((po-y)**2).mean(); base=100000*(1-BR0/V)
zo=_lg(np.clip(po,1e-9,1-1e-9))
print(f'배포 블렌드 정직면 기준선 {base:.4f}점',flush=True)

def build(cols, cats, catify_id):
    Xt=tr_tr[cols].copy(); Xv=tr_va[cols].copy()
    cs=list(cats)+ (["pitcher_id","batter_id"] if catify_id else [])
    cs=[c for c in cs if c in cols]
    for c in cs: Xt[c]=Xt[c].astype(str); Xv[c]=Xv[c].astype(str)
    return Xt,Xv,[cols.index(c) for c in cs]

DESIGNS=[
 ('D1_raw47_d6',      RAW, dict(depth=6,  iterations=1000, learning_rate=0.06, l2_leaf_reg=6.0), False),
 ('D2_raw47_d10',     RAW, dict(depth=10, iterations=2000, learning_rate=0.05, l2_leaf_reg=3.0), False),
 ('D3_raw47_d4_i300', RAW, dict(depth=4,  iterations=300,  learning_rate=0.06, l2_leaf_reg=20.0),False),
 ('D4_unused15',      UNUSED, dict(depth=6, iterations=1000, learning_rate=0.06, l2_leaf_reg=6.0),False),
 ('D5_unused15_cnt',  UNUSED+COUNT, dict(depth=8, iterations=1500, learning_rate=0.06, l2_leaf_reg=6.0),False),
 ('D6_raw47_catid',   RAW, dict(depth=6,  iterations=1000, learning_rate=0.06, l2_leaf_reg=6.0), True),
 ('D7_raw47_rsm03',   RAW, dict(depth=6,  iterations=1500, learning_rate=0.06, l2_leaf_reg=6.0, rsm=0.3), False),
 ('D8_raw47_lang',    RAW, dict(depth=8,  iterations=1500, learning_rate=0.04, l2_leaf_reg=1.0,
                                langevin=True, diffusion_temperature=10000), False),
]
COMMON=dict(random_seed=42, task_type='GPU', devices='0', verbose=False, allow_writing_files=False)
yb=(ystr>0.5).astype(int)
out={'baseline_oot_pts':float(base),'threshold_prereg':9.31,'watch_rule':'w_opt>0 and gain>=3.0','arms':{}}
Z={}
for nm, cols, hp, catid in DESIGNS:
    t=time.time()
    try:
        Xt,Xv,ci=build(cols,CATS,catid)
        m=CatBoostClassifier(loss_function='Logloss',**hp,**COMMON)
        m.fit(Pool(Xt,yb,cat_features=ci))
        p=m.predict_proba(Pool(Xv,cat_features=ci))[:,1]
        m.save_model(f'{R}/{nm}.cbm')
    except Exception as e:
        print(f'  {nm} 실패: {e}',flush=True); out['arms'][nm]={'error':str(e)}; continue
    pv=p[order]; z=_lg(np.clip(pv,1e-9,1-1e-9)); z=z-z.mean(); Z[nm]=z
    rho=float(np.corrcoef(z,zo)[0,1])
    f=lambda w: ((expit(zo+w*z)-y)**2).mean()
    r=minimize_scalar(f,bounds=(-1.0,2.0),method='bounded',options={'xatol':1e-6})
    rp=minimize_scalar(f,bounds=(0.0,2.0),method='bounded',options={'xatol':1e-6})
    out['arms'][nm]=dict(n_feat=len(cols),
        solo_pts=float(100000*(1-((pv-y)**2).mean()/V)),
        rho_logit=rho, w_opt=float(r.x), blend_gain_pts=float(100000*(BR0-r.fun)/V),
        w_pos=float(rp.x), gain_wpos=float(100000*(BR0-rp.fun)/V), sec=round(time.time()-t,1))
    a=out['arms'][nm]
    print(f"  {nm:18s} {a['sec']:6.1f}s  단독 {a['solo_pts']:8.2f}  rho {rho:+.4f}  "
          f"w {a['w_opt']:+.4f}  이득 {a['blend_gain_pts']:+.3f}  (w>=0: {a['gain_wpos']:+.3f})",flush=True)

ok={k:v for k,v in out['arms'].items() if 'error' not in v}
out['best'] = max(ok, key=lambda k: ok[k]['blend_gain_pts']) if ok else None
out['any_positive_w'] = [k for k,v in ok.items() if v['w_opt']>0]
out['verdict'] = ('채택' if ok and ok[out['best']]['blend_gain_pts']>=9.31 else
                  '주목' if any(v['w_opt']>0 and v['blend_gain_pts']>=3.0 for v in ok.values()) else '종결')
if len(Z)>1:
    ks=list(Z); out['pairwise_rho']={f'{a}|{b}':float(np.corrcoef(Z[a],Z[b])[0,1])
                                     for i,a in enumerate(ks) for b in ks[i+1:]}
out['sec']=round(time.time()-t0,1)
json.dump(out,open(f'{R}/fanout.json','w'),ensure_ascii=False,indent=1)
print('\n판정:',out['verdict'],'| 양수 w:',out['any_positive_w'])
print('DONE_FANOUT')
