"""feat2.parquet — ins_* 를 팀 배포 모듈 inseason.py 로 재생성.

왜 다시 만드는가
  1차(feats.py) 자체 구현에 두 결함이 있었다
    (a) 앵커 n 에만 +1 하고 카운트에는 안 더해 off-by-one 편향 (dev 평균 -0.0179 vs 정본 -0.0321)
    (b) ins_b_* 가 전부 NaN
  배포 추론은 어차피 inseason.py 를 타므로, 학습도 같은 코드로 만드는 것이 유일하게 안전하다
  (인수인계 §9 '학습·추론이 같은 코드를 타도록 통일' 규율).

as-of 규율: 연도 Y 의 앵커는 season < Y 행으로만 만든다.
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("v16"))
import inseason
import feats as FT

t0 = time.time()
df = pd.read_parquet("feat.parquet")
old = [c for c in df.columns if c.startswith("ins_")]
raw = pd.read_csv("data/data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "batter_id", "asof_pitcher_n",
                          "asof_batter_n"] + FT.P_RATES + FT.B_RATES)
print(f"[load] {df.shape} old ins {len(old)} {time.time()-t0:.0f}s", flush=True)

parts = []
for Y in sorted(raw.season.unique()):
    cur = raw[raw.season == Y]
    hist = raw[raw.season < Y]
    if len(hist) == 0:
        # 2019: 앞선 시즌이 없다. 존재하지 않는 id 하나로 더미 앵커를 만들어
        # 전 행이 '앵커 없음'(= 신인 취급, ins = 커리어 전체)으로 축퇴하게 한다.
        d = raw.iloc[:1].copy()
        d["pitcher_id"] = -1
        d["batter_id"] = -1
        A = inseason.build_anchor(d)
    else:
        A = inseason.build_anchor(hist)
    I = inseason.apply(cur, A)
    I.index = cur.index
    parts.append(I)
    print(f"  [{Y}] {len(cur):,} rows  {time.time()-t0:.0f}s", flush=True)
INS = pd.concat(parts).reindex(df.index)
print("[nan]", {c: int(INS[c].isna().sum()) for c in INS.columns if INS[c].isna().any()})
INS = INS.astype(np.float32)

out = pd.concat([df.drop(columns=old), INS], axis=1)
out.to_parquet("feat2.parquet", index=False)
print(f"[save] feat2.parquet {out.shape}  ins {list(INS.columns)}  {time.time()-t0:.0f}s")
