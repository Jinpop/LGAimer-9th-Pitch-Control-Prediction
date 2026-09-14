#!/usr/bin/env python3
"""exp_123 — 모형족을 바꾼다. exp_122 는 8설계 전부 CatBoost 였다.

exp_122 가 남긴 빈 사분면: "전 열 유지 + 다른 알고리즘 → 낮은 rho".
  D4/D5 는 열을 뺏어 rho 를 낮췄고 단독이 -259/-588 로 망가졌다.
  D1/D3 는 단독은 좋은데 rho 0.84 로 붙어 있다.
배깅(RF/ExtraTrees)은 부스팅과 성향이 근본적으로 다르고,
XGBoost 는 현 블렌드(lgb/cat/nn/leaf)에 없는 네 번째 GBM 구현이다.

사전 등록(결과 보기 전 고정, exp_122 와 동일):
  채택 = 혼합 이득 >= +9.31점
  주목 = w_opt > 0 이고 이득 >= +3.0
  둘 다 미달이면 모형족 갈래도 종결.
추가로 최고 신규 팔 + exp_122 D1(CatBoost) 을 ★동시★ 투입한 2팔 이득도 잰다.
"""
import numpy as np, pandas as pd, json, os, sys, time
from scipy.special import logit as _lg, expit
from scipy.optimize import minimize_scalar, minimize
H='/home/<user>/SungjinPark'; R=f'{H}/exp123'; os.makedirs(R,exist_ok=True)
tc=[c.strip() for c in open(f'{H}/data/test.csv').readline().replace('﻿','').strip().split(',')]
RAW=[c for c in tc if c!='row_id']
CATS=['top_bottom','game_type','base_state','pitcher_hand','batter_hand']

def load(seasons):
    tr=pd.read_csv(f'{H}/data/train.csv'); tr.columns=[c.replace('﻿','') for c in tr.columns]
    tr=tr[tr.season.isin(seasons)].copy().reset_index(drop=True)
    return tr, tr['control_success'].to_numpy(np.float64)

t0=time.time()
tr_tr,ystr=load(list(range(2019,2024))); tr_va,ysva=load([2024])
print(f'학습 {len(tr_tr)} · 검증 {len(tr_va)}',flush=True)
rid112=pd.Series(np.load(f'{H}/exp112_rid.npy',allow_pickle=True)).astype(str).values
po=np.load(f'{H}/exp111/p_oot.npy')
order=pd.Index(tr_va['row_id'].astype(str)).get_indexer(pd.Index(rid112)); assert (order>=0).all()
y=ysva[order]
assert abs(((po-y)**2).mean()-0.2474119208667189)<1e-12,'p_oot 정렬 검증 실패'
V=y.mean()*(1-y.mean()); BR0=((po-y)**2).mean(); base=100000*(1-BR0/V)
zo=_lg(np.clip(po,1e-9,1-1e-9))
print(f'기준선 {base:.4f}점',flush=True)
yb=(ystr>0.5).astype(int)

# 범주형: 서수 부호화(전 모형 공통) + XGB 용 category dtype
Xt_num=tr_tr[RAW].copy(); Xv_num=tr_va[RAW].copy()
for c in CATS:
    cats=pd.Index(sorted(set(Xt_num[c].astype(str))|set(Xv_num[c].astype(str))))
    Xt_num[c]=cats.get_indexer(Xt_num[c].astype(str)); Xv_num[c]=cats.get_indexer(Xv_num[c].astype(str))
Xt_num=Xt_num.astype(np.float32).fillna(-999.0); Xv_num=Xv_num.astype(np.float32).fillna(-999.0)

RES={}
def score(nm,pv,sec):
    z=_lg(np.clip(pv,1e-9,1-1e-9)); z=z-z.mean()
    rho=float(np.corrcoef(z,zo)[0,1])
    f=lambda w: ((expit(zo+w*z)-y)**2).mean()
    r=minimize_scalar(f,bounds=(-1.0,2.0),method='bounded',options={'xatol':1e-6})
    rp=minimize_scalar(f,bounds=(0.0,2.0),method='bounded',options={'xatol':1e-6})
    RES[nm]=dict(solo_pts=float(100000*(1-((pv-y)**2).mean()/V)),rho_logit=rho,
                 w_opt=float(r.x),blend_gain_pts=float(100000*(BR0-r.fun)/V),
                 w_pos=float(rp.x),gain_wpos=float(100000*(BR0-rp.fun)/V),sec=round(sec,1))
    a=RES[nm]; print(f"  {nm:16s} {a['sec']:6.1f}s  단독 {a['solo_pts']:8.2f}  rho {rho:+.4f}  "
        f"w {a['w_opt']:+.4f}  이득 {a['blend_gain_pts']:+.3f}  (w>=0: {a['gain_wpos']:+.3f})",flush=True)
    return z

Zs={}
# ---------- XGBoost
import xgboost as xgb
for nm,hp in [('E1_xgb_d6',  dict(max_depth=6, eta=0.06, num_boost_round=1000)),
              ('E2_xgb_deep',dict(max_depth=12,eta=0.04, num_boost_round=1200,
                                  colsample_bytree=0.3, subsample=0.6, min_child_weight=50))]:
    t=time.time(); nr=hp.pop('num_boost_round')
    p={**hp,'objective':'binary:logistic','tree_method':'hist','device':'cuda',
       'eval_metric':'logloss','seed':42,'nthread':16}
    try:
        bst=xgb.train(p, xgb.DMatrix(Xt_num,label=yb), num_boost_round=nr)
        pv=bst.predict(xgb.DMatrix(Xv_num))[order]; bst.save_model(f'{R}/{nm}.json')
        Zs[nm]=score(nm,pv,time.time()-t)
    except Exception as e: print(f'  {nm} 실패: {e}',flush=True); RES[nm]={'error':str(e)}

# ---------- 배깅 계열
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
for nm,mk in [('E3_rf',   lambda: RandomForestClassifier(n_estimators=300,max_depth=14,
                   max_features=0.3,min_samples_leaf=200,n_jobs=38,random_state=42)),
              ('E4_et',   lambda: ExtraTreesClassifier(n_estimators=400,max_depth=18,
                   max_features=0.3,min_samples_leaf=100,n_jobs=38,random_state=42)),
              ('E5_hgb',  lambda: HistGradientBoostingClassifier(max_iter=600,learning_rate=0.06,
                   max_depth=8,l2_regularization=1.0,random_state=42))]:
    t=time.time()
    try:
        m=mk(); m.fit(Xt_num.values,yb)
        pv=m.predict_proba(Xv_num.values)[:,1][order]
        Zs[nm]=score(nm,pv,time.time()-t)
    except Exception as e: print(f'  {nm} 실패: {e}',flush=True); RES[nm]={'error':str(e)}

# ---------- exp_122 D1(CatBoost) 재적재해 짝 rho + 2팔 동시 투입
try:
    from catboost import CatBoostClassifier, Pool
    Xv_c=tr_va[RAW].copy()
    for c in CATS: Xv_c[c]=Xv_c[c].astype(str)
    cb=CatBoostClassifier(); cb.load_model(f'{H}/exp122/D1_raw47_d6.cbm')
    pv=cb.predict_proba(Pool(Xv_c,cat_features=[RAW.index(c) for c in CATS]))[:,1][order]
    zd1=_lg(np.clip(pv,1e-9,1-1e-9)); zd1=zd1-zd1.mean()
    RES['D1_reload']=dict(solo_pts=float(100000*(1-((pv-y)**2).mean()/V)),
                          rho_logit=float(np.corrcoef(zd1,zo)[0,1]))
    ok={k:v for k,v in RES.items() if 'error' not in v and k in Zs}
    if ok:
        b=max(ok,key=lambda k:RES[k]['blend_gain_pts']); zb=Zs[b]
        g=lambda w: ((expit(zo+w[0]*zb+w[1]*zd1)-y)**2).mean()
        r2=minimize(g,[0.0,0.0],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-12})
        RES['JOINT_best_plus_D1']=dict(best=b,rho_best_vs_D1=float(np.corrcoef(zb,zd1)[0,1]),
            w=[float(r2.x[0]),float(r2.x[1])],gain_pts=float(100000*(BR0-r2.fun)/V))
        print('  2팔 동시:',json.dumps(RES['JOINT_best_plus_D1'],ensure_ascii=False),flush=True)
except Exception as e: print('  D1 재적재 실패:',e,flush=True)

ok={k:v for k,v in RES.items() if 'blend_gain_pts' in v}
out=dict(baseline_oot_pts=float(base),threshold_prereg=9.31,watch_rule='w_opt>0 and gain>=3.0',
         arms=RES, best=(max(ok,key=lambda k:ok[k]['blend_gain_pts']) if ok else None),
         any_positive_w=[k for k,v in ok.items() if v['w_opt']>0])
out['verdict']=('채택' if ok and ok[out['best']]['blend_gain_pts']>=9.31 else
                '주목' if any(v['w_opt']>0 and v['blend_gain_pts']>=3.0 for v in ok.values()) else '종결')
out['sec']=round(time.time()-t0,1)
json.dump(out,open(f'{R}/family.json','w'),ensure_ascii=False,indent=1)
print('\n판정:',out['verdict'],'| 최고:',out['best'],'| 양수 w:',out['any_positive_w'])
print('DONE_FAMILY')
