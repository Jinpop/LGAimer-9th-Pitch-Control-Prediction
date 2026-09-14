"""exp_060 stage1 — v6 파생 44열(테이블 21 + 비율 22 + p6는 테이블 안에 포함)을
연도별 as-of로 생성. 사용법: python prep_v6.py <YEAR>

규율: hist = train[season < Y] 로만 테이블·구종모델·anom 분모를 만든다
(sp_def README §2와 같은 연도 루프. cnt_te/te48/sit은 타깃 인코딩이라 causal 필수,
 구종모델·anom은 타깃 미포함이지만 같은 규율로 맞춘다 — valid 2024가 완전 as-of가 된다).
2019(선행 시즌 없음): 테이블 파생 전부 NaN, 원시열만으로 계산 가능한 비율은 계산.
코드는 출하본 pipeline.py(build_BH/pkg_BH/mate, sha a3181a10…)를 그대로 import.
"""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")

import numpy as np
import pandas as pd

B = os.path.expanduser("~/SungjinPark/exp060_nn")
sys.path.insert(0, B)
import pipeline as P  # noqa: E402  (출하본 사본)
import lightgbm as lgb  # noqa: E402

Y = int(sys.argv[1])
t0 = time.time()

tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"),
                 encoding="utf-8-sig")
cur = tr[tr.season == Y]
hist = tr[tr.season < Y]
print(f"Y={Y} cur {len(cur)} hist {len(hist)}", flush=True)

FEAT_COLS = (P.PC + ["idea_cnt_te", "i3_te48", "idea_matchup_pitch"]
             + [f"i3_anom_{t}" for t in P.TYPES] + ["i3_anom_l1"]
             + ["sit_cnt", "sit_cnt_dev", "sit_run", "sit_run_dev",
                "sit_sc", "sit_sc_dev"])

if len(hist) == 0:
    # 2019 — 테이블 파생 전부 NaN, base_out NaN으로 비율 계산(원시 전용 비율만 값이 남는다)
    feats = pd.DataFrame(np.nan, index=cur.index, columns=FEAT_COLS)
    ratios = P.apply_ratio_features(cur, feats)
    t_fit = 0.0
else:
    tdir = f"{B}/work/tables_{Y}"
    P.build_tables(hist, tdir)

    tm = pd.read_parquet(f"{B}/work/tm_mapped.parquet")
    tm_hist = tm[tm.season < Y]
    t1 = time.time()
    m, lv = P.build_pitch_model(tm_hist, tdir)
    t_fit = time.time() - t1
    print(f"Y={Y} pitch model fit {t_fit/60:.2f}min on {len(tm_hist)} tm rows",
          flush=True)

    # anom 분모: hist 행에 p6를 적용해 카운트별 평균 (as-of: hist·<Y 모델만 사용)
    Xh = hist[P.PM_FEATS].copy()
    for c in P.PM_CATS:
        Xh[c] = pd.Categorical(Xh[c], categories=lv[c])
    Ph = m.predict(Xh)
    hp = hist[["balls_before", "strikes_before"]].copy()
    for i, t in enumerate(P.TYPES):
        hp[f"p6_{t}"] = Ph[:, i]
    P.build_anom_table(hp, tdir)

    feats = P.apply_features(cur, tdir)
    ratios = P.apply_ratio_features(cur, feats)

out = pd.concat([feats, ratios], axis=1)
assert out.index.equals(cur.index)
out.astype("float32").to_parquet(f"{B}/work/v6_{Y}.parquet")
print(f"Y={Y} done cols {out.shape[1]} elapsed {(time.time()-t0)/60:.2f}min "
      f"(fit {t_fit/60:.2f}min)", flush=True)
