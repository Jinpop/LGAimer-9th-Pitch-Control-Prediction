"""in-season 분해 피처 — asof_* 를 '이전 시즌까지 커리어' 와 '당시즌 누적' 으로 분리.

원리
----
`asof_pitcher_n` 은 시즌 경계에서 끊기지 않는 **커리어 누적**이다
(train 의 투수-시즌 전이 1330건 전부 min(다음시즌) - max(이전시즌) = 1 로 검증).
따라서 어떤 행의 자기 값에서 "그 선수의 직전 시즌 종료 시점 누적치"를 빼면
**그 행 시점까지의 당시즌 성적**이 복원된다.

앵커(직전 시즌 종료 누적치)는 train.csv 로만 만든다. 2025 평가행의 앵커는
2024 시즌 종료 시점이고 그건 train 안에 있다.

규칙 준수
--------
- 각 행의 출력은 (그 행의 asof_* 값) + (train 유래 동결 앵커 테이블) 만의 함수다.
- 다른 평가 행을 일절 참조하지 않는다 → 규칙 4항(행 독립성) 준수.
- `asof_*` 는 주최측이 "공식 입력 피처이므로 사용할 수 있습니다" 라고 명시한 컬럼이다.
- 평가 데이터의 분포·순서·빈도를 쓰지 않는다.
"""
import numpy as np
import pandas as pd

PR = ['success', 'reverse', 'middle', 'ball', 'strike']
BR = ['success', 'middle']
K = 50.0


def _counts(n, rate):
    """정수 누적 카운트 복원. rate 는 6자리 반올림이라 n<10^6 에서 round 가 정확."""
    return np.round(np.asarray(n, np.float64) * np.nan_to_num(np.asarray(rate, np.float64)))


def build_anchor(df_hist):
    """df_hist: 대상 시즌보다 앞선 train 행 전체. 반환: 동결 앵커 dict."""
    n = df_hist['asof_pitcher_n'].to_numpy(np.float64)
    bn = df_hist['asof_batter_n'].to_numpy(np.float64)
    t = pd.DataFrame({'pitcher_id': df_hist.pitcher_id.to_numpy(),
                      'batter_id': df_hist.batter_id.to_numpy(), 'n': n, 'bn': bn})
    for r in PR:
        t['c_' + r] = _counts(n, df_hist['asof_pitcher_' + r + '_rate'])
    for r in BR:
        t['b_' + r] = _counts(bn, df_hist['asof_batter_' + r + '_rate'])
    p = t.loc[t.groupby('pitcher_id')['n'].idxmax()].set_index('pitcher_id')
    b = t.loc[t.groupby('batter_id')['bn'].idxmax()].set_index('batter_id')
    lg = {r: float(np.nansum(t['c_' + r]) / max(np.nansum(t['n']), 1)) for r in PR}
    lgb_ = {r: float(np.nansum(t['b_' + r]) / max(np.nansum(t['bn']), 1)) for r in BR}
    return {'p_ids': p.index.to_numpy(), 'p_n': p['n'].to_numpy(),
            'p_c': np.stack([p['c_' + r].to_numpy() for r in PR], 1),
            'b_ids': b.index.to_numpy(), 'b_n': b['bn'].to_numpy(),
            'b_c': np.stack([b['b_' + r].to_numpy() for r in BR], 1),
            'lg': np.array([lg[r] for r in PR]), 'lgb': np.array([lgb_[r] for r in BR])}


def apply(df, A):
    """행별 lookup + 산술만. 정렬·집계·분위수·rolling 일절 없음."""
    out = pd.DataFrame(index=df.index)
    n = df['asof_pitcher_n'].to_numpy(np.float64)
    pos = pd.Index(A['p_ids']).get_indexer(df['pitcher_id'].to_numpy())
    kn = pos >= 0
    safe = np.where(kn, pos, 0)
    an = np.where(kn, A['p_n'][safe], 0.0)                     # 미지 투수 = 신인 = 앵커 0
    ins_n = np.maximum(n - an, 0.0)
    out['ins_p_n'] = ins_n
    for i, r in enumerate(PR):
        ac = np.where(kn, A['p_c'][safe, i], 0.0)
        prior = np.where(an > 0, ac / np.maximum(an, 1.0), A['lg'][i])
        k = np.maximum(_counts(n, df['asof_pitcher_' + r + '_rate']) - ac, 0.0)
        out['ins_p_' + r] = (k + K * prior) / (ins_n + K)
        out['ins_p_' + r + '_dev'] = out['ins_p_' + r] - prior
    bn = df['asof_batter_n'].to_numpy(np.float64)
    pos = pd.Index(A['b_ids']).get_indexer(df['batter_id'].to_numpy())
    kn = pos >= 0
    safe = np.where(kn, pos, 0)
    abn = np.where(kn, A['b_n'][safe], 0.0)
    ins_bn = np.maximum(bn - abn, 0.0)
    out['ins_b_n'] = ins_bn
    for i, r in enumerate(BR):
        ac = np.where(kn, A['b_c'][safe, i], 0.0)
        prior = np.where(abn > 0, ac / np.maximum(abn, 1.0), A['lgb'][i])
        k = np.maximum(_counts(bn, df['asof_batter_' + r + '_rate']) - ac, 0.0)
        out['ins_b_' + r] = (k + K * prior) / (ins_bn + K)
        out['ins_b_' + r + '_dev'] = out['ins_b_' + r] - prior
    return out.astype('float32')


def save(A, path):
    np.savez(path, **A)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
