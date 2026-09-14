#!/usr/bin/env python3
"""exp_120 3단계 — 새 팔 후보(5클래스 다항) vs 대조군(이항) 을 정직면에서 겨룬다.

사전 등록 (결과 보기 전 확정, S8):
  채택 문턱   정직면 혼합 이득 >= +9.31  (S187, 딥 슬롯 밖 새 팔)
  즉시 폐기   최적 w == 0
  대조군      같은 행·같은 열·같은 GPU 설정의 ★이항★ 팔 (S135 — 무변형 복제본을 사다리 맨 아래에)
  ⚠️ 알려진 편의: 프레임의 동결표가 <=2024 라 <=2023 학습 행에 미래가 조금 샌다.
     그 편의는 후보와 대조군에 ★공통★ 이므로 둘의 ★차이★ 로만 읽는다.
"""
import numpy as np, pandas as pd, json, os, sys, time
from catboost import CatBoostClassifier, Pool
from scipy.special import logit as _lg, expit
H='/home/<user>/SungjinPark'; R=f'{H}/exp120'
K=400309.19882517256
meta=json.load(open(f'{H}/build_EA/pkg/PAR/model/meta.json',encoding='utf-8'))
BR=meta['branches']['R']; FEAT=BR['base_features']; CATS=BR['cat_features']
sys.path.insert(0,f'{H}/build_AI/src'); import labels as L

def build(tag):
    raw=pd.read_csv(f'{R}/frames/{tag}/test.csv',encoding='utf-8-sig')
    raw.columns=[c.replace('﻿','') for c in raw.columns]
    x29=pd.read_csv(f'{R}/aa29_{tag}.csv',encoding='utf-8-sig')
    x29.columns=[c.replace('﻿','') for c in x29.columns]
    assert list(x29['row_id'])==list(raw['row_id']), f'{tag}: aa29 행 순서 불일치'
    df=pd.concat([raw.reset_index(drop=True), x29.drop(columns=['row_id']).reset_index(drop=True)],axis=1)
    miss=[c for c in FEAT if c not in df.columns]
    assert not miss, f'{tag}: 빠진 열 {miss}'
    X=df[FEAT].copy()
    for c in CATS: X[c]=X[c].astype(str)
    return df, X

def cells(tag):
    """train.csv 원본에서 5셀 라벨을 복원한다 (labels.recover)."""
    tr=pd.read_csv(f'{H}/data/train.csv'); tr.columns=[c.replace('﻿','') for c in tr.columns]
    tr=tr[tr.season<=2023] if tag=='le2023' else tr[tr.season==2024]
    L.recover(tr, verbose=False)
    y5=np.full(len(tr),-1,dtype=np.int64)
    for i,c in enumerate(L.CELLS): y5[tr[c].to_numpy().astype(bool)]=i
    return tr['row_id'].to_numpy(), y5, tr['control_success'].to_numpy(np.float64)

t0=time.time()
print('프레임 조립...'); dtr,Xtr=build('le2023'); dva,Xva=build('y2024')
rid_tr,y5_tr,ys_tr=cells('le2023'); rid_va,y5_va,ys_va=cells('y2024')
assert list(rid_tr)==list(dtr['row_id']) and list(rid_va)==list(dva['row_id']), '행 순서 불일치'
ok_tr=y5_tr>=0; ok_va=y5_va>=0
print(f'  학습 {len(Xtr)}행(5셀 복원 {ok_tr.sum()}) · 검증 {len(Xva)}행(복원 {ok_va.sum()}) · 열 {len(FEAT)}')
print('  5셀 분포(학습):',dict(zip(L.CELLS,np.bincount(y5_tr[ok_tr],minlength=5))))
ci=[FEAT.index(c) for c in CATS]
Ptr5=Pool(Xtr[ok_tr],y5_tr[ok_tr],cat_features=ci); Ptr2=Pool(Xtr,(ys_tr>0.5).astype(int),cat_features=ci)
Pva=Pool(Xva,cat_features=ci)
COMMON=dict(iterations=1000,learning_rate=0.06,depth=6,l2_leaf_reg=6.0,
            random_seed=42,task_type='GPU',devices='0',verbose=200,allow_writing_files=False)
res={}
for nm,pool,loss in [('A_binary',Ptr2,'Logloss'),('B_multi',Ptr5,'MultiClass')]:
    print(f'\n=== {nm} ({loss}) 학습...'); t=time.time()
    m=CatBoostClassifier(loss_function=loss,**COMMON); m.fit(pool)
    p=m.predict_proba(Pva)
    ps=p[:,1] if loss=='Logloss' else p[:,list(L.CELLS).index('is_success')]
    m.save_model(f'{R}/{nm}.cbm')
    br=((ps-ys_va)**2).mean(); V=ys_va.mean()*(1-ys_va.mean())
    res[nm]=dict(p=ps, secs=round(time.time()-t,1), brier=float(br), pts=float(100000*(1-br/V)))
    print(f'  {nm}: {res[nm]["secs"]}초  단독 {res[nm]["pts"]:.2f}점  mean(p)={ps.mean():.5f}')
np.save(f'{R}/p_A.npy',res['A_binary']['p']); np.save(f'{R}/p_B.npy',res['B_multi']['p'])
# ---- 정직면 혼합
po=np.load(f'{H}/exp111/p_oot.npy')
assert len(po)==len(ys_va), (len(po),len(ys_va))
print('\n⚠️ p_oot 행 순서는 exp112 calcols 기준이다 — row_id 로 정렬 대조한다')
rid112=pd.Series(np.load(f'{H}/../SungjinPark/exp120/rid112.npy',allow_pickle=True)).astype(str).values \
       if os.path.exists(f'{R}/rid112.npy') else None
out={'sec_total':round(time.time()-t0,1),'n_train':int(len(Xtr)),'n_valid':int(len(Xva)),
     'threshold_prereg':9.31,'arms':{k:{kk:vv for kk,vv in v.items() if kk!='p'} for k,v in res.items()}}
json.dump(out,open(f'{R}/train_result.json','w'),ensure_ascii=False,indent=1)
print('\n'+json.dumps(out,ensure_ascii=False,indent=1))
print('DONE_TRAIN')
