#!/usr/bin/env python3
"""exp_120 1단계 — 새 팔(5클래스 다항) 학습용 test 형태 프레임을 만든다.

패키지를 이 프레임 위에서 한 번 돌리면 mate/output/aa29.csv 가 나오고,
그 29열 + 원본 47열 = AA R갈래 base 76열이 된다.
"""
import pandas as pd, numpy as np, os, sys
H='/home/<user>/SungjinPark'; OUT=f'{H}/exp120/frames'
os.makedirs(OUT, exist_ok=True)
tr=pd.read_csv(f'{H}/data/train.csv'); tr.columns=[c.replace('﻿','') for c in tr.columns]
tc=[c.strip() for c in open(f'{H}/data/test.csv').readline().replace('﻿','').strip().split(',')]
print('test 열',len(tc),'| train 행',len(tr))
for tag, mask in [('le2023', tr.season<=2023), ('y2024', tr.season==2024)]:
    d=tr[mask].copy()
    f=os.path.join(OUT,tag); os.makedirs(f,exist_ok=True)
    d[tc].to_csv(os.path.join(f,'test.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame({'row_id':d['row_id'].values,'control_success':0.5}).to_csv(
        os.path.join(f,'sample_submission.csv'), index=False, encoding='utf-8-sig')
    d[['row_id','season','control_success']+[c for c in tr.columns if c.startswith('asof_pitcher_') and c.endswith('_rate')]+['asof_pitcher_n','pitcher_id']].to_parquet(
        os.path.join(f,'meta.parquet'))
    print(f'  {tag}: {len(d)}행 -> {f}')
print('DONE')
