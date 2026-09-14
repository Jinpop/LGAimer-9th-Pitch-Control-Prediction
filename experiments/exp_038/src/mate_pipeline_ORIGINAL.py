"""제출용 피처 생성 경로 (script.py가 그대로 사용).

학습 때 쓴 경로와 결과가 비트 동일해야 한다. 로컬에서만 맞고 test에서
어긋나는 종류의 결함을 막기 위해 이 모듈 하나로 양쪽을 통일한다.
"""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

TYPES = ['curve', 'cutter', 'fastball', 'offspeed', 'sinker', 'slider']
PC = [f'p6_{t}' for t in TYPES]
PM_FEATS = ['inning', 'top_bottom', 'balls_before', 'strikes_before',
            'outs_before', 'batter_hand', 'pitcher_hand', 'pitcher_id']
PM_CATS = ['top_bottom', 'batter_hand', 'pitcher_hand', 'pitcher_id']
COMBO = {f'{p}_{b}_{d}': i for i, (p, b, d) in enumerate(
    [(p, b, d) for p in (1, 2) for b in (1, 2) for d in range(-1, 6)])}
K_SIT, K_TE48, K_CNT = 80.0, 300.0, 500.0
EPS = 1e-3


# ---------------- 테이블 생성 (학습 데이터에서) ----------------

def build_tables(df_hist, outdir):
    """df_hist: 대상 시즌보다 앞선 train 행 전체 (control_success 포함)"""
    os.makedirs(outdir, exist_ok=True)
    T = 'control_success'
    base = df_hist[T].mean()

    cs = df_hist.balls_before.astype(str) + '_' + df_hist.strikes_before.astype(str)
    mc = (df_hist.pitcher_hand.astype(int).astype(str) + '_' +
          df_hist.batter_hand.astype(int).astype(str) + '_' + cs)

    g = df_hist[T].groupby(cs).agg(['sum', 'size'])
    ((g['sum'] + K_CNT * base) / (g['size'] + K_CNT)).rename('v').to_csv(f'{outdir}/cnt_te.csv')

    g = df_hist[T].groupby(mc).agg(['sum', 'size'])
    ((g['sum'] + K_TE48 * base) / (g['size'] + K_TE48)).rename('v').to_csv(f'{outdir}/te48.csv')

    d = df_hist.copy()
    d['runner'] = (d.num_runners_on > 0).astype(int)
    d['scoring'] = (d.num_runners_on >= 2).astype(int)
    overall = d.groupby('pitcher_id')[T].agg(['mean', 'size'])
    overall.columns = ['p_all', 'p_n']
    overall.to_csv(f'{outdir}/pitcher_overall.csv')
    for name, keys in [('cnt', ['balls_before', 'strikes_before']),
                       ('run', ['runner']), ('sc', ['scoring'])]:
        gg = d.groupby(['pitcher_id'] + keys)[T].agg(['mean', 'size']).join(overall['p_all'])
        w = gg['size'] / (gg['size'] + K_SIT)
        gg[f'sit_{name}'] = w * gg['mean'] + (1 - w) * gg['p_all']
        gg[f'sit_{name}_dev'] = gg[f'sit_{name}'] - gg['p_all']
        gg[[f'sit_{name}', f'sit_{name}_dev']].reset_index().to_csv(
            f'{outdir}/sit_{name}.csv', index=False)
    json.dump({'base': float(base)}, open(f'{outdir}/meta.json', 'w'))


def build_pitch_model(tm_hist, outdir):
    """tm_hist: 대상 시즌보다 앞선 trackman 행 (pitcher_id 매핑 완료, lab 포함)"""
    y = tm_hist.lab.map({t: i for i, t in enumerate(TYPES)}).values
    X = tm_hist[PM_FEATS].copy()
    lv = {c: sorted(X[c].dropna().unique().tolist()) for c in PM_CATS}
    for c in PM_CATS:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    p = {'objective': 'multiclass', 'num_class': 6, 'learning_rate': 0.08,
         'num_leaves': 63, 'min_data_in_leaf': 300, 'feature_fraction': 0.9,
         'bagging_fraction': 0.9, 'bagging_freq': 1, 'seed': 42, 'verbose': -1}
    m = lgb.train(p, lgb.Dataset(X, y, categorical_feature=PM_CATS), 250)
    m.save_model(f'{outdir}/pitchmodel6.txt')
    json.dump({k: [int(x) if not isinstance(x, str) else x for x in v]
               for k, v in lv.items()}, open(f'{outdir}/pmcat6.json', 'w'))
    return m, lv


def build_anom_table(df_rows_with_p6, outdir):
    """카운트별 평균 구종확률 (의외성 비율의 분모)"""
    cs = (df_rows_with_p6.balls_before.astype(str) + '_' +
          df_rows_with_p6.strikes_before.astype(str))
    df_rows_with_p6.groupby(cs)[PC].mean().to_csv(f'{outdir}/anom_cnt.csv')


# ---------------- 피처 적용 (학습/추론 공통) ----------------

def apply_features(df, tdir):
    m = lgb.Booster(model_file=f'{tdir}/pitchmodel6.txt')
    lv = json.load(open(f'{tdir}/pmcat6.json'))
    X = df[PM_FEATS].copy()
    for c in PM_CATS:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    P = m.predict(X)
    out = pd.DataFrame(P, columns=PC, index=df.index)

    cs = df.balls_before.astype(str) + '_' + df.strikes_before.astype(str)
    mc = (df.pitcher_hand.astype(int).astype(str) + '_' +
          df.batter_hand.astype(int).astype(str) + '_' + cs)

    cnt = pd.read_csv(f'{tdir}/cnt_te.csv', index_col=0)['v']
    te48 = pd.read_csv(f'{tdir}/te48.csv', index_col=0)['v']
    out['idea_cnt_te'] = cs.map(cnt).values
    out['i3_te48'] = mc.map(te48).values

    dom = np.nanargmax(np.nan_to_num(P, nan=-1), axis=1)
    key = (df.pitcher_hand.astype(int).astype(str) + '_' +
           df.batter_hand.astype(int).astype(str) + '_' +
           pd.Series(dom, index=df.index).astype(str))
    out['idea_matchup_pitch'] = key.map(COMBO).astype(float).values

    ac = pd.read_csv(f'{tdir}/anom_cnt.csv', index_col=0)
    ref = ac.reindex(cs)[PC].values
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = P / np.clip(ref, EPS, None)
    for i, t in enumerate(TYPES):
        out[f'i3_anom_{t}'] = ratio[:, i]
    out['i3_anom_l1'] = np.nansum(np.abs(P - ref), axis=1)

    tmp = df[['pitcher_id']].copy()
    tmp['runner'] = (df.num_runners_on > 0).astype(int)
    tmp['scoring'] = (df.num_runners_on >= 2).astype(int)
    tmp['balls_before'] = df.balls_before.values
    tmp['strikes_before'] = df.strikes_before.values
    for name, keys in [('cnt', ['balls_before', 'strikes_before']),
                       ('run', ['runner']), ('sc', ['scoring'])]:
        tab = pd.read_csv(f'{tdir}/sit_{name}.csv')
        s = tmp[['pitcher_id'] + keys].merge(tab, on=['pitcher_id'] + keys, how='left')
        out[f'sit_{name}'] = s[f'sit_{name}'].values
        out[f'sit_{name}_dev'] = s[f'sit_{name}_dev'].values
    return out
