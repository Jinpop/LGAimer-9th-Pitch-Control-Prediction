#!/usr/bin/env python3
"""exp_124 — 결합 규칙을 넓힌다. exp_122/123 은 전역 스칼라 하나로만 쟀다.

질문: "팔에 정보가 없다" 인가 "스칼라 하나로 못 꺼낸다" 인가?
S184: 모델이 구조적으로 틀리는 유일한 자리는 카운트다 -> 세그먼트 가중이 핵심 후보.
S190: 표본 안만 보면 속는다 -> ★전부 투수 id 2겹 교차적합★, 표본 안 값도 병기.

사전 등록: 교차적합 이득이 전역 스칼라 대비 3배 이상이면 '계열 재개', 아니면 종결 확정.
"""
import numpy as np, pandas as pd, json, time
from scipy.special import logit as _lg, expit
from scipy.optimize import minimize_scalar, minimize
from catboost import CatBoostClassifier, Pool
H='/home/<user>/SungjinPark'
tc=[c.strip() for c in open(f'{H}/data/test.csv').readline().replace('﻿','').strip().split(',')]
RAW=[c for c in tc if c!='row_id']; CATS=['top_bottom','game_type','base_state','pitcher_hand','batter_hand']
tr=pd.read_csv(f'{H}/data/train.csv'); tr.columns=[c.replace('﻿','') for c in tr.columns]
va=tr[tr.season==2024].reset_index(drop=True)
rid=pd.Series(np.load(f'{H}/exp112_rid.npy',allow_pickle=True)).astype(str).values
po=np.load(f'{H}/exp111/p_oot.npy')
o=pd.Index(va['row_id'].astype(str)).get_indexer(pd.Index(rid)); assert (o>=0).all()
va=va.iloc[o].reset_index(drop=True)
y=va['control_success'].to_numpy(float)
V=y.mean()*(1-y.mean()); BR0=((po-y)**2).mean(); zo=_lg(np.clip(po,1e-9,1-1e-9))
print(f'기준선 {100000*(1-BR0/V):.4f}점  n={len(y)}',flush=True)

Xv=va[RAW].copy()
for c in CATS: Xv[c]=Xv[c].astype(str)
P=Pool(Xv,cat_features=[RAW.index(c) for c in CATS])
ARMS={}
for nm,f in [('D1_cat',f'{H}/exp122/D1_raw47_d6.cbm'),('D6_catid',f'{H}/exp122/D6_raw47_catid.cbm')]:
    m=CatBoostClassifier(); m.load_model(f)
    if nm=='D6_catid':
        X2=va[RAW].copy()
        for c in CATS+['pitcher_id','batter_id']: X2[c]=X2[c].astype(str)
        p=m.predict_proba(Pool(X2,cat_features=[RAW.index(c) for c in CATS+['pitcher_id','batter_id']]))[:,1]
    else: p=m.predict_proba(P)[:,1]
    z=_lg(np.clip(p,1e-9,1-1e-9)); ARMS[nm]=z-z.mean()
print('팔 적재:',list(ARMS),flush=True)

fold=(pd.util.hash_pandas_object(va['pitcher_id'],index=False).to_numpy()%2).astype(int)
print('겹 크기',np.bincount(fold),flush=True)
cell=(va['balls_before'].clip(0,3)*3+va['strikes_before'].clip(0,2)).to_numpy()  # 12칸
def pts(br): return 100000*(BR0-br)/V

def eval_rule(fit, apply_):
    """fit(idx)->theta ; apply_(theta, idx)->p  ; 표본안/교차적합 둘 다"""
    th=fit(np.arange(len(y))); ins=pts(((apply_(th,np.arange(len(y)))-y)**2).mean())
    pv=np.empty(len(y))
    for k in (0,1):
        trn=np.where(fold!=k)[0]; tst=np.where(fold==k)[0]
        pv[tst]=apply_(fit(trn),tst)
    return ins, pts(((pv-y)**2).mean())

OUT={}
for nm,z in ARMS.items():
    # R1 전역 스칼라
    def f1(idx):
        g=lambda w: ((expit(zo[idx]+w*z[idx])-y[idx])**2).mean()
        return minimize_scalar(g,bounds=(-1.,2.),method='bounded',options={'xatol':1e-7}).x
    a1=eval_rule(f1, lambda th,idx: expit(zo[idx]+th*z[idx]))
    # R2 카운트 12칸 세그먼트 가중
    def f2(idx):
        w=np.zeros(12)
        for c in range(12):
            m=idx[cell[idx]==c]
            if len(m)<500: continue
            g=lambda v: ((expit(zo[m]+v*z[m])-y[m])**2).mean()
            w[c]=minimize_scalar(g,bounds=(-1.,2.),method='bounded',options={'xatol':1e-7}).x
        return w
    a2=eval_rule(f2, lambda th,idx: expit(zo[idx]+th[cell[idx]]*z[idx]))
    # R3 2차 스태커  zo + a*z + b*z*zo + c*z^2
    def f3(idx):
        B=np.c_[z[idx],z[idx]*zo[idx],z[idx]**2]
        g=lambda t: ((expit(zo[idx]+B@t)-y[idx])**2).mean()
        return minimize(g,np.zeros(3),method='Nelder-Mead',
                        options={'xatol':1e-8,'fatol':1e-14,'maxiter':4000}).x
    a3=eval_rule(f3, lambda th,idx: expit(zo[idx]+np.c_[z[idx],z[idx]*zo[idx],z[idx]**2]@th))
    OUT[nm]={'R1_global':{'in':a1[0],'cv':a1[1]},'R2_count12':{'in':a2[0],'cv':a2[1]},
             'R3_quad':{'in':a3[0],'cv':a3[1]}}
    for k,v in OUT[nm].items():
        print(f"  {nm:9s} {k:12s} 표본안 {v['in']:+7.3f}  교차적합 {v['cv']:+7.3f}",flush=True)

best_cv=max(max(v[k]['cv'] for k in v) for v in OUT.values())
g1=max(v['R1_global']['cv'] for v in OUT.values())
OUT['_summary']={'best_cv':best_cv,'global_cv':g1,'ratio':(best_cv/g1 if g1>0 else None),
                 'verdict':'계열 재개' if g1>0 and best_cv>=3*g1 else '종결 확정'}
print('\n',json.dumps(OUT['_summary'],ensure_ascii=False))
json.dump(OUT,open(f'{H}/exp124_combrule.json','w'),ensure_ascii=False,indent=1)
print('DONE_COMB')
