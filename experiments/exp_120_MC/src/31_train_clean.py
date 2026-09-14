#!/usr/bin/env python3
"""exp_120 3b — ★표 유래 열을 전부 빼고★ 원열 47개로만 다시. 그리고 혼합 이득을 실제로 잰다.

3단계에서 76열을 쓰니 sp_*(<=2024 라벨 EB 표)·te 계열이 2024 라벨을 새게 했다
(단일 팔이 배포 블렌드 정직면을 239점 이겼다 = 불가능).
여기서는 test.csv 원열 47개만 쓴다 — 동결표 의존 0, 누출 경로 0.
"""
import numpy as np, pandas as pd, json, os, sys, time
from catboost import CatBoostClassifier, Pool
from scipy.special import logit as _lg, expit
from scipy.optimize import minimize_scalar
H='/home/<user>/SungjinPark'; R=f'{H}/exp120'; K=400309.19882517256
sys.path.insert(0,f'{H}/build_AI/src'); import labels as L
tc=[c.strip() for c in open(f'{H}/data/test.csv').readline().replace('﻿','').strip().split(',')]
RAW=[c for c in tc if c!='row_id']
CATS=['top_bottom','game_type','base_state','pitcher_hand','batter_hand']
print('원열 피처',len(RAW),'개 (표 유래 0)')

def load(tag, seasons):
    tr=pd.read_csv(f'{H}/data/train.csv'); tr.columns=[c.replace('﻿','') for c in tr.columns]
    tr=tr[tr.season.isin(seasons)].copy().reset_index(drop=True)
    L.recover(tr, verbose=False)
    y5=np.full(len(tr),-1,np.int64)
    for i,c in enumerate(L.CELLS): y5[tr[c].to_numpy().astype(bool)]=i
    X=tr[RAW].copy()
    for c in CATS: X[c]=X[c].astype(str)
    return tr, X, y5, tr['control_success'].to_numpy(np.float64)

t0=time.time()
tr_tr,Xtr,y5tr,ystr=load('le2023',list(range(2019,2024)))
tr_va,Xva,y5va,ysva=load('y2024',[2024])
print(f'학습 {len(Xtr)} · 검증 {len(Xva)}')
ci=[RAW.index(c) for c in CATS]
Pva=Pool(Xva,cat_features=ci)
COMMON=dict(iterations=1000,learning_rate=0.06,depth=6,l2_leaf_reg=6.0,random_seed=42,
            task_type='GPU',devices='0',verbose=500,allow_writing_files=False)
P={}
for nm,pool,loss in [('A_binary',Pool(Xtr,(ystr>0.5).astype(int),cat_features=ci),'Logloss'),
                     ('B_multi', Pool(Xtr,y5tr,cat_features=ci),'MultiClass')]:
    print(f'\n=== {nm} ({loss})'); t=time.time()
    m=CatBoostClassifier(loss_function=loss,**COMMON); m.fit(pool)
    pr=m.predict_proba(Pva)
    p=pr[:,1] if loss=='Logloss' else pr[:,L.CELLS.index('is_success')]
    m.save_model(f'{R}/clean_{nm}.cbm'); P[nm]=p
    V=ysva.mean()*(1-ysva.mean()); br=((p-ysva)**2).mean()
    print(f'  {nm}: {time.time()-t:.1f}초  단독 {100000*(1-br/V):.2f}점  mean(p)={p.mean():.5f}')

# ---- 정직면 혼합 (p_oot 와 row_id 정렬)
rid112=pd.Series(np.load(f'{H}/exp112_rid.npy',allow_pickle=True)).astype(str).values \
       if os.path.exists(f'{H}/exp112_rid.npy') else None
po=np.load(f'{H}/exp111/p_oot.npy')
assert rid112 is not None and len(rid112)==len(po), 'exp112 row_id 배열이 필요하다'
order=pd.Index(tr_va['row_id'].astype(str)).get_indexer(pd.Index(rid112))
assert (order>=0).all(), 'row_id 정렬 실패'
y=ysva[order]
assert abs(((po-y)**2).mean()-0.2474119208667189)<1e-12, 'p_oot 정렬 검증 실패'
V=y.mean()*(1-y.mean()); base=100000*(1-((po-y)**2).mean()/V)
zo=_lg(np.clip(po,1e-9,1-1e-9))
print(f'\n배포 블렌드 정직면 기준선 {base:.4f}점')
out={'features':len(RAW),'baseline_oot_pts':float(base),'threshold_prereg':9.31,'arms':{}}
for nm,p in P.items():
    pv=p[order]; z=_lg(np.clip(pv,1e-9,1-1e-9)); z=z-z.mean()
    rho=float(np.corrcoef(z,zo)[0,1])
    f=lambda w: ((expit(zo+w*z)-y)**2).mean()
    r=minimize_scalar(f,bounds=(-1.0,2.0),method='bounded',options={'xatol':1e-6})
    gain=100000*(((po-y)**2).mean()-r.fun)/V
    solo=100000*(1-((pv-y)**2).mean()/V)
    out['arms'][nm]=dict(solo_pts=float(solo),rho_logit=rho,w_opt=float(r.x),blend_gain_pts=float(gain))
    print(f'  {nm}: 단독 {solo:8.2f}  rho {rho:+.4f}  최적 w {r.x:+.4f}  혼합 이득 {gain:+.3f}점')
out['B_minus_A_solo']=out['arms']['B_multi']['solo_pts']-out['arms']['A_binary']['solo_pts']
out['verdict']=('채택' if out['arms']['B_multi']['blend_gain_pts']>=9.31 else '문턱 미달')
out['sec']=round(time.time()-t0,1)
json.dump(out,open(f'{R}/clean_result.json','w'),ensure_ascii=False,indent=1)
print('\n'+json.dumps(out,ensure_ascii=False,indent=1)); print('DONE_CLEAN')
